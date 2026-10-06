"""Model-native associative memory — JARVS's own typed links between its memories.

The vector stores already connect memories whose embeddings are near; those edges
say only "these two are similar". This module adds a second, qualitatively different
layer: edges JARVS draws **itself** during a periodic "dream" (consolidation) pass —
typed associations across memories a similarity metric would never connect. Each edge
carries:

    a, b     — the two memory ids it joins
    kind     — one of KINDS (analogy / causality / contrast / pattern / intuition)
    reason   — a one-line justification for the link
    weight   — reinforced when the same pair is drawn again
    source   — "llm" (dreamt)

Edges live in their own JSON store keyed by the unordered memory-id pair, separate
from the vector-store metadata, so recall can follow them alongside the cosine links.
Vendored from JCN3690.
"""

from __future__ import annotations

import json
import os
import re
import time

# The taxonomy echoes the "model-native pathways" reasoning forms: an analogy, a
# causal chain, a deliberate contrast, a recurring pattern, or an intuitive leap.
KINDS = ("analogy", "causality", "contrast", "pattern", "intuition")

WEIGHT_CAP = 3.0        # a repeatedly-drawn edge stops strengthening here
_REASON_MAX = 200


def _coerce_kind(raw) -> str:
    k = str(raw or "").strip().lower()
    if k in KINDS:
        return k
    # tolerate near-misses ("causal" -> "causality", "analogical" -> "analogy")
    for kind in KINDS:
        if k and (k.startswith(kind[:5]) or kind.startswith(k[:5])):
            return kind
    return "pattern"


class LinkGraph:
    """An undirected, typed graph of JARVS's self-drawn memory associations."""

    def __init__(self, path: str):
        self.path = path
        self.edges: dict[str, dict] = {}   # "a|b" (sorted) -> edge dict
        self._adj: dict[str, set] = {}     # memory id -> set of edge keys
        self.load()

    # -- persistence --------------------------------------------------------
    @staticmethod
    def _key(a: str, b: str) -> str:
        return "|".join(sorted((a, b)))

    def _index(self, key: str, edge: dict) -> None:
        self._adj.setdefault(edge["a"], set()).add(key)
        self._adj.setdefault(edge["b"], set()).add(key)

    def load(self) -> None:
        self.edges, self._adj = {}, {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        for e in data.get("edges", []):
            a, b = e.get("a"), e.get("b")
            if not a or not b or a == b:
                continue
            key = self._key(a, b)
            self.edges[key] = e
            self._index(key, e)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"edges": list(self.edges.values())}, f, indent=2)

    # -- mutation -----------------------------------------------------------
    def add_edge(self, a: str, b: str, kind: str, reason: str,
                 source: str = "llm", weight: float = 1.0, ts=None) -> bool:
        if not a or not b or a == b:
            return False
        kind = _coerce_kind(kind)
        reason = " ".join(str(reason or "").split())[:_REASON_MAX]
        ts = ts or time.time()
        key = self._key(a, b)
        e = self.edges.get(key)
        if e:  # drawn before — reinforce and refresh the rationale
            e["weight"] = round(min(e.get("weight", 1.0) + weight * 0.5, WEIGHT_CAP), 3)
            e["kind"] = kind
            if reason:
                e["reason"] = reason
            e["source"] = source
            e["updated"] = ts
            return True
        e = {"a": a, "b": b, "kind": kind, "reason": reason,
             "weight": round(weight, 3), "source": source,
             "created": ts, "updated": ts}
        self.edges[key] = e
        self._index(key, e)
        return True

    def prune(self, valid_ids) -> int:
        """Drop edges whose endpoints no longer exist (e.g. after an episodic prune)."""
        valid = set(valid_ids)
        removed = 0
        for key in list(self.edges):
            e = self.edges[key]
            if e["a"] not in valid or e["b"] not in valid:
                del self.edges[key]
                removed += 1
        if removed:
            self._adj = {}
            for key, e in self.edges.items():
                self._index(key, e)
            self.save()
        return removed

    def reset(self) -> None:
        self.edges, self._adj = {}, {}
        self.save()

    # -- queries ------------------------------------------------------------
    def neighbors(self, id_: str) -> list[dict]:
        out = []
        for key in self._adj.get(id_, ()):
            e = self.edges[key]
            other = e["b"] if e["a"] == id_ else e["a"]
            out.append({"other": other, "kind": e["kind"], "reason": e["reason"],
                        "weight": e.get("weight", 1.0), "source": e.get("source", "llm")})
        out.sort(key=lambda x: x["weight"], reverse=True)
        return out

    def degree(self, id_: str) -> int:
        return len(self._adj.get(id_, ()))

    def all_edges(self) -> list[dict]:
        return list(self.edges.values())

    def stats(self) -> dict:
        by_kind: dict[str, int] = {}
        for e in self.edges.values():
            by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1
        return {"count": len(self.edges), "by_kind": by_kind}


# ---------------------------------------------------------------------------
# The dream (consolidation) pass — JARVS weaving its own links across a sample of
# memories. The caller feeds in the candidate memories and a
# `generate(system, user) -> str` callable.
# ---------------------------------------------------------------------------

_DREAM_SYSTEM = """You are the memory-consolidation process of JARVS, a calm, precise \
AI assistant, running while idle. You are shown a numbered sample of JARVS's own \
stored memories. Your task is to draw NON-OBVIOUS associative links between them — \
the connections a mind makes for itself, not ones a similarity search would already \
find.

Link memories that are DIFFERENT on the surface but share a deeper thread. Prefer \
links across distant memories. Do NOT link two memories that merely restate the \
same fact.

Each link has a kind:
- analogy: two situations share a structure ("A is to B as C is to D").
- causality: one memory helps explain or caused another.
- contrast: they oppose or tension each other in a revealing way.
- pattern: they are two instances of the same recurring theme.
- intuition: a subtle, associative leap you cannot fully justify but sense.

Respond with ONLY a JSON array (no prose), each element:
{"a": <number>, "b": <number>, "kind": "<one kind>", "reason": "<max 15 words>"}
Draw at most %d links. If nothing worthy connects, return []."""

_DREAM_USER = """JARVS's memory sample:

%s

Draw the links now as a JSON array."""


def _extract_json_array(raw: str):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip(),
                  flags=re.MULTILINE).strip()
    try:
        val = json.loads(text)
        return val if isinstance(val, list) else None
    except Exception:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            val = json.loads(text[start:end + 1])
            return val if isinstance(val, list) else None
        except Exception:
            return None
    return None


def weave(graph: LinkGraph, candidates: list[dict], generate_fn,
          max_links: int = 8) -> dict:
    """Run one dream pass. `candidates` is a list of dicts with keys idx (1-based int),
    id (memory id), store ("episodic"/"semantic"), text. `generate_fn(system, user) ->
    str` performs the LLM call. Returns a summary and mutates `graph` in place."""
    if len(candidates) < 4:
        return {"added": 0, "considered": len(candidates)}

    by_idx = {c["idx"]: c for c in candidates}
    listing = "\n".join(f"[{c['idx']}] ({c['store']}) {c['text']}"
                        for c in candidates)
    raw = generate_fn(_DREAM_SYSTEM % max_links, _DREAM_USER % listing)
    links = _extract_json_array(raw) or []

    added = 0
    for item in links:
        if not isinstance(item, dict):
            continue
        try:
            ai, bi = int(item.get("a")), int(item.get("b"))
        except (TypeError, ValueError):
            continue
        if ai == bi or ai not in by_idx or bi not in by_idx:
            continue
        a_id, b_id = by_idx[ai]["id"], by_idx[bi]["id"]
        if graph.add_edge(a_id, b_id, item.get("kind"), item.get("reason")):
            added += 1
        if added >= max_links:
            break
    return {"added": added, "considered": len(candidates), "proposed": len(links)}
