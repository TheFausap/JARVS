"""Meta-dreaming: a bounded *self-referential* pass that improves JARVS's own CRITIC,
not just its replies.

Motivation (Yin et al. 2024, "Gödel Agent"): a learning stack that only refines its
replies always does so inside a FIXED evaluation rubric. The Gödel-agent insight is
"improve the optimizer, not only the policy." This module takes the one safe, bounded
step in that direction — it lets the engine refine the thing that *drives* its
learning, the critic's prompt, rather than the replies the critic judges. Ported from
ENGRAM2's ``meta_dreaming.py`` and re-pointed from fiction-prose craft to
assistant-answer craft.

Crucially, and unlike a literal Gödel Agent, it NEVER modifies code. It edits a prompt
only within a CLOSED allowlist of mechanical critique-focus areas
(:data:`CRITIQUE_FOCUS_AREAS`). The model can select and instantiate from that
taxonomy; it cannot invent new instructions for the critic. The self-referential loop:

    critic prompt  ->  critiques  ->  logged lessons  ->  meta-dreaming  -> …
         ^                                                      |
         +------------------  focus directive  -----------------+

How it works:

  * It reads the critic's append-only critique log, keeping its OWN offset.
  * A background LLM pass clusters recurring LESSON notes into WEAKNESS CLASSES and
    maps each to exactly one allowlisted focus area, citing the source notes it drew
    from (bounded to the real batch, so it cannot invent corroboration).
  * Deterministic validation rejects an out-of-allowlist area, or a focus note that
    names a known identity, carries personality/tone vocabulary, or smells like a
    prompt injection.
  * A focus area is INJECTED into the critic prompt only once corroborated by
    >= ``promote_min_pairs`` DISTINCT sessions — so one conversation's quirk can't
    reshape every future critic.

Best-effort throughout: a failure here never disturbs a live reply.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable

from .dream_common import (
    sanitize_rule,
    rule_similarity,
    _words,
    _TRAIT_DIRECTIVE_WORDS,
)


# The closed allowlist of critique-focus areas. The LLM must map every recurring
# weakness onto one of these keys; anything else is rejected. Each is a purely
# MECHANICAL concern that transfers across ANY topic — none is about who JARVS IS
# (tone/persona). The value is the canonical, neutral phrasing injected into the critic.
CRITIQUE_FOCUS_AREAS: dict[str, str] = {
    "directness":
        "whether the reply answers the actual question first, before any elaboration",
    "concision":
        "whether the reply stays as brief as the task allows rather than padding it",
    "relevance":
        "whether every part of the reply bears on what was actually asked",
    "epistemic_rigor":
        "whether the reply makes only claims it can support and flags its uncertainty",
    "continuity_rigor":
        "whether the reply stays consistent with what was established earlier and in memory",
    "clarity":
        "whether the phrasing is unambiguous and the structure easy to follow",
    "actionability":
        "whether the reply gives concrete, usable next steps when the task calls for them",
    "scope_control":
        "whether the reply avoids unrequested tangents and scope creep",
    "format_fit":
        "whether formatting suits the medium (clean spoken prose vs. a written layout)",
    "calibration":
        "whether the reply's confidence matches the strength of its evidence",
}

DEFAULT_MIN_NEW = 16        # new critique lines before a meta pass fires
DEFAULT_PROMOTE_PAIRS = 3   # distinct sessions before an area is injected
_MAX_SPECIFIC_CHARS = 180
_MERGE_SIMILARITY = 0.72    # Jaccard overlap that refreshes an area's specific
_MAX_INJECTED_AREAS = 3     # keep the critic prompt focused, not a laundry list


@dataclass
class FocusArea:
    area: str                                   # key from CRITIQUE_FOCUS_AREAS
    specific: str = ""                          # distilled recurring-pattern note
    hits: int = 1
    contributing_pairs: list = field(default_factory=list)  # distinct session ids
    example_lessons: list = field(default_factory=list)     # a couple, for /show
    updated_at: float = 0.0

    @property
    def pair_count(self) -> int:
        return len(set(self.contributing_pairs))


_DISTILL_SYSTEM = """You are a reviewer reading many short critique notes an automated \
critic produced across DIFFERENT conversations and DIFFERENT topics. Each note is one \
weakness the critic flagged in a reply JARVS gave.

Your job is NOT to fix any reply. It is to decide which WEAKNESS CLASSES recur often \
enough that the critic should scrutinize every future reply for them harder.

Map each recurring class to EXACTLY ONE focus area, using its key verbatim:
{areas}

Rules:
- Only report a class supported by SEVERAL different notes.
- Ignore one-off notes, and anything tied to a specific user, topic, task, \
personality, mood, or tone.
- Cite the numbered notes each class came from.

Format each class on its own line, EXACTLY:
AREA_KEY | one-line description of the recurring pattern | SOURCES: n,n,n

AREA_KEY must be one of the keys above. The description must be general craft only \
(no names, no topic, no personality/tone words). Output at most {max_areas} lines. \
If nothing recurs, output only: NONE"""


class MetaDreamer:
    def __init__(self, *,
                 log_path: str,
                 store_path: str,
                 generate_fn: Callable[[str, str, bool], str],
                 known_names: Iterable[str] = (),
                 promote_min_pairs: int = DEFAULT_PROMOTE_PAIRS,
                 distill_min_new: int = DEFAULT_MIN_NEW,
                 max_areas_per_pass: int = 4):
        self._log_path = log_path
        self._store_path = store_path
        self._generate = generate_fn
        self._promote_min_pairs = max(1, int(promote_min_pairs))
        self._distill_min_new = max(1, int(distill_min_new))
        self._max_areas_per_pass = max(1, int(max_areas_per_pass))
        self._lock = threading.RLock()
        names = sorted({n.strip().lower() for n in known_names if n and n.strip()},
                       key=len, reverse=True)
        self._name_re = (re.compile(r"\b(" + "|".join(re.escape(n) for n in names)
                                    + r")\b", re.IGNORECASE) if names else None)
        self._store = self._load_store()

    # -- persistence -------------------------------------------------------- #

    def _empty_store(self) -> dict:
        return {"version": 1, "areas": [],
                "last_distillation": {"log_offset": 0, "timestamp": 0.0}}

    def _load_store(self) -> dict:
        try:
            with open(self._store_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return self._empty_store()
        if not isinstance(data, dict) or not isinstance(data.get("areas"), list):
            return self._empty_store()
        data.setdefault("version", 1)
        data.setdefault("last_distillation", {"log_offset": 0, "timestamp": 0.0})
        return data

    def _save_store(self):
        os.makedirs(os.path.dirname(self._store_path) or ".", exist_ok=True)
        tmp = f"{self._store_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._store, fh, indent=2)
        os.replace(tmp, self._store_path)

    # -- cadence (reads the critic's log at our own offset) ----------------- #

    def _new_lines(self) -> tuple:
        """Return (records, new_offset) for log content past OUR offset."""
        offset = int(self._store.get("last_distillation", {}).get("log_offset", 0))
        records = []
        try:
            with open(self._log_path, "r", encoding="utf-8") as fh:
                fh.seek(offset)
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                new_offset = fh.tell()
        except OSError:
            return [], offset
        return records, new_offset

    def should_distill(self) -> bool:
        with self._lock:
            records, _ = self._new_lines()
            return len(records) >= self._distill_min_new

    # -- validation --------------------------------------------------------- #

    def _passes_name_filter(self, text: str) -> bool:
        return not (self._name_re and self._name_re.search(text or ""))

    def _passes_trait_filter(self, text: str) -> bool:
        return not (_words(text) & _TRAIT_DIRECTIVE_WORDS)

    def _specific_ok(self, text: str) -> bool:
        if not text:
            return False
        return self._passes_name_filter(text) and self._passes_trait_filter(text)

    # -- the distillation sweep --------------------------------------------- #

    def distill(self) -> int:
        """Read new critique notes, cluster recurring weakness classes onto allowlisted
        focus areas via the LLM, validate, and merge survivors into the store with
        evidence-grounded session attribution. Returns the number of areas added or
        refreshed. Best-effort; never raises."""
        with self._lock:
            try:
                return self._distill_locked()
            except Exception:
                return 0

    def _distill_locked(self) -> int:
        records, new_offset = self._new_lines()
        if len(records) < self._distill_min_new:
            return 0

        numbered, pair_of, lesson_of = [], {}, {}
        for i, rec in enumerate(records, start=1):
            lesson = sanitize_rule(str(rec.get("lesson", "")))
            # Fold the critic's own axis verdicts in: a recurring epistemic /
            # continuity miss is a weakness class even when the LESSON was terse.
            extra = []
            for axis in ("epistemic", "continuity"):
                val = str(rec.get(axis, "ok")).strip()
                if val and val.lower() not in ("ok", "none", ""):
                    extra.append(f"[{axis}] {val}")
            note = "; ".join([p for p in (lesson, *extra) if p])
            if not note:
                continue
            numbered.append(f"{i}. {note}")
            pair_of[i] = str(rec.get("pair", "?"))
            lesson_of[i] = note

        # Too few usable notes: still advance the offset so we never re-scan them.
        if len(numbered) < 4:
            self._store["last_distillation"] = {"log_offset": new_offset,
                                                "timestamp": time.time()}
            self._save_store()
            return 0

        areas_help = "\n".join(f"- {k}: {v}" for k, v in CRITIQUE_FOCUS_AREAS.items())
        system = _DISTILL_SYSTEM.format(areas=areas_help,
                                        max_areas=self._max_areas_per_pass)
        raw = self._generate(system, "CRITIQUE NOTES:\n" + "\n".join(numbered), False)

        added = 0
        for area, specific, sources in self._parse_distilled(raw):
            if area not in CRITIQUE_FOCUS_AREAS:          # closed-allowlist gate
                continue
            if not self._specific_ok(specific):           # name/trait/injection
                continue
            pairs = {pair_of[s] for s in sources if s in pair_of}
            if not pairs:                                  # must be grounded
                continue
            examples = [lesson_of[s] for s in sources if s in lesson_of][:2]
            if self._merge_area(area, specific, pairs, examples):
                added += 1

        self._store["areas"].sort(
            key=lambda a: (len(set(a.get("contributing_pairs", []))),
                           a.get("hits", 0), a.get("updated_at", 0.0)),
            reverse=True)
        self._store["last_distillation"] = {"log_offset": new_offset,
                                            "timestamp": time.time()}
        self._save_store()
        return added

    @staticmethod
    def _parse_distilled(raw: str):
        """Yield (area_key, specific_text, [source_indices]) from model output."""
        if not raw or raw.strip().upper() == "NONE":
            return
        for line in raw.splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue
            area = parts[0].lower().replace(" ", "_").replace("-", "_")
            specific = sanitize_rule(parts[1])[:_MAX_SPECIFIC_CHARS].rstrip()
            if not specific:
                continue
            sources = []
            src_blob = parts[2] if len(parts) >= 3 else ""
            for n in re.findall(r"\d+", src_blob):
                try:
                    sources.append(int(n))
                except ValueError:
                    pass
            yield area, specific, sources

    def _merge_area(self, area: str, specific: str, pairs: set,
                    examples: list) -> bool:
        """One stored entry per area key: accumulate corroborating sessions, keep the
        freshest well-supported recurring-pattern phrasing."""
        for existing in self._store["areas"]:
            if existing.get("area") == area:
                merged = set(existing.get("contributing_pairs", [])) | pairs
                existing["contributing_pairs"] = sorted(merged)
                existing["hits"] = int(existing.get("hits", 1)) + 1
                existing["updated_at"] = time.time()
                # Refresh the specific only when it genuinely restates the same pattern
                # (keeps phrasing current without thrashing on noise).
                if (not existing.get("specific")
                        or rule_similarity(existing["specific"], specific) >= _MERGE_SIMILARITY):
                    existing["specific"] = specific
                ex = (existing.get("example_lessons", []) + examples)[:3]
                existing["example_lessons"] = ex
                return True
        self._store["areas"].append(asdict(FocusArea(
            area=area, specific=specific, hits=1, contributing_pairs=sorted(pairs),
            example_lessons=examples[:3], updated_at=time.time())))
        return True

    # -- injection ---------------------------------------------------------- #

    def _promoted(self) -> list:
        areas = [a for a in self._store.get("areas", [])
                 if len(set(a.get("contributing_pairs", []))) >= self._promote_min_pairs
                 and a.get("area") in CRITIQUE_FOCUS_AREAS]
        areas.sort(key=lambda a: (len(set(a.get("contributing_pairs", []))),
                                  a.get("hits", 0)), reverse=True)
        return areas[:_MAX_INJECTED_AREAS]

    def focus_directive(self) -> str:
        """Render the ``[CRITIQUE FOCUS]`` block for the self-critique prompt from
        PROMOTED areas only. Additive: it tells the critic what to scrutinize harder;
        it never relaxes the fixed axes. Empty until an area is promoted, so an
        untrained engine behaves exactly as before."""
        with self._lock:
            promoted = self._promoted()
        if not promoted:
            return ""
        lines = [
            "[CRITIQUE FOCUS]",
            "Across many conversations, replies have recurred on the weaknesses below. "
            "Scrutinize this reply especially hard for each. This ADDS emphasis and "
            "does NOT relax the axes above.",
        ]
        for a in promoted:
            canonical = CRITIQUE_FOCUS_AREAS[a["area"]]
            label = a["area"].replace("_", " ").upper()
            specific = a.get("specific", "")
            tail = f" Recurring pattern: {specific}" if specific else ""
            lines.append(f"- {label} — check {canonical}.{tail}")
        return "\n".join(lines)

    # -- introspection / control ------------------------------------------- #

    def snapshot(self) -> dict:
        with self._lock:
            areas = self._store.get("areas", [])
            promoted = sum(1 for a in areas
                           if len(set(a.get("contributing_pairs", []))) >= self._promote_min_pairs)
            records, _ = self._new_lines()
            return {
                "total_areas": len(areas),
                "promoted_areas": promoted,
                "promote_min_pairs": self._promote_min_pairs,
                "pending_lines": len(records),
                "distill_min_new": self._distill_min_new,
                "last_distillation": dict(self._store.get("last_distillation", {})),
                "areas": [
                    {"area": a.get("area"),
                     "pairs": len(set(a.get("contributing_pairs", []))),
                     "hits": a.get("hits", 0),
                     "specific": a.get("specific", ""),
                     "examples": a.get("example_lessons", [])}
                    for a in sorted(areas,
                                    key=lambda a: len(set(a.get("contributing_pairs", []))),
                                    reverse=True)
                ],
            }

    def reset(self):
        """Clear the focus store and rewind our offset so the log can be re-scanned."""
        with self._lock:
            self._store = self._empty_store()
            self._save_store()
