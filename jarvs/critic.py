"""
critic.py — the self-critique pass (the thing meta-dreaming improves).

After a reply is delivered, the critic grades it on a FIXED set of mechanical axes and
appends one line to an append-only critique log. Two parties read that log:
  * nobody on the reply path (the critique is observational — it never blocks a reply);
  * :class:`~jarvs.meta_dreaming.MetaDreamer`, which distills recurring weaknesses in
    the logged LESSONs into a ``[CRITIQUE FOCUS]`` block.

That block is injected back into THIS critic's prompt via ``focus_fn``, closing the
self-referential loop: the critic's own rubric gets sharper over time, within a bounded
allowlist, without ever changing code or the fixed axes.

Best-effort: every method swallows its own failures so a bookkeeping pass can never
disturb a live turn.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Callable

from .dream_common import sanitize_rule

# The FIXED axes. Meta-dreaming may add EMPHASIS (what to scrutinize harder) but never
# removes or relaxes these.
_CRITIC_SYSTEM = """You are the self-critique process of JARVS, a precise AI assistant. \
You are shown the user's message and JARVS's reply. Grade the reply honestly and \
tersely on these axes:

- QUALITY (0-10): did it answer well — direct, relevant, clear, correctly scoped?
- EPISTEMIC: does it make only claims it could support, and flag uncertainty? Reply \
"ok" or name the specific overreach.
- CONTINUITY: is it consistent with the conversation so far and anything in context? \
Reply "ok" or name the specific inconsistency.
- LESSON: the single most useful mechanical improvement for NEXT time, as one general \
craft note (no restating this specific reply's content). Reply "none" if the reply was \
already good.

{focus}

Respond EXACTLY in this form, nothing else:
SCORE: <0-10>
EPISTEMIC: <ok | issue>
CONTINUITY: <ok | issue>
LESSON: <one line | none>"""

_CRITIC_USER = """USER MESSAGE:
{user}

JARVS REPLY:
{reply}

Grade it now."""


def _verdict(score: float) -> str:
    if score >= 7.0:
        return "ok"
    if score >= 4.0:
        return "needs_improvement"
    return "poor"


class Critic:
    def __init__(self, *, log_path: str, session_id: str,
                 generate_fn: Callable[[str, str, bool], str],
                 focus_fn: Callable[[], str] | None = None):
        self._log_path = log_path
        self._session = session_id
        self._generate = generate_fn
        self._focus_fn = focus_fn or (lambda: "")
        self._append_lock = threading.Lock()

    # -- the critique pass ---------------------------------------------------- #

    def critique(self, user_input: str, reply: str) -> dict:
        """Grade one reply and log it. Returns the parsed verdict dict (empty on
        failure). Never raises."""
        try:
            focus = self._focus_fn() or ""
        except Exception:  # noqa: BLE001
            focus = ""
        system = _CRITIC_SYSTEM.format(focus=focus or "")
        user = _CRITIC_USER.format(user=user_input.strip(), reply=reply.strip())
        try:
            raw = self._generate(system, user, False)
        except Exception:  # noqa: BLE001
            return {}
        parsed = self._parse(raw)
        if parsed:
            self._log(user_input, parsed)
        return parsed

    @staticmethod
    def _parse(raw: str) -> dict:
        if not raw:
            return {}
        fields: dict = {}
        for line in raw.splitlines():
            m = re.match(r"\s*(SCORE|EPISTEMIC|CONTINUITY|LESSON)\s*:\s*(.*)$",
                         line, re.IGNORECASE)
            if m:
                fields[m.group(1).upper()] = m.group(2).strip()
        if not fields:
            return {}
        try:
            score = float(re.findall(r"-?\d+(?:\.\d+)?", fields.get("SCORE", ""))[0])
        except (IndexError, ValueError):
            score = 5.0
        score = max(0.0, min(10.0, score))
        lesson = sanitize_rule(fields.get("LESSON", ""))
        return {
            "score": round(score, 1),
            "verdict": _verdict(score),
            "epistemic": (fields.get("EPISTEMIC", "ok") or "ok")[:200],
            "continuity": (fields.get("CONTINUITY", "ok") or "ok")[:200],
            "lesson": lesson,
        }

    # -- the append-only log (meta-dreaming's input) -------------------------- #

    def _log(self, user_input: str, parsed: dict) -> None:
        """Append one compact JSON line. ``pair`` is the session id — the diversity
        axis meta-dreaming uses so one conversation can't reshape the critic. A terse
        or "none" lesson with clean axes carries no signal, so it is skipped."""
        lesson = parsed.get("lesson", "")
        epi = parsed.get("epistemic", "ok")
        cont = parsed.get("continuity", "ok")
        has_axis_signal = (epi.lower() not in ("ok", "none", "")
                           or cont.lower() not in ("ok", "none", ""))
        if not lesson and not has_axis_signal:
            return
        record = {
            "t": round(time.time(), 1),
            "pair": self._session,
            "prompt": (user_input or "")[:200],
            "score": parsed.get("score", 0.0),
            "verdict": parsed.get("verdict", "unknown"),
            "lesson": lesson,
            "epistemic": epi,
            "continuity": cont,
        }
        try:
            os.makedirs(os.path.dirname(self._log_path) or ".", exist_ok=True)
            line = json.dumps(record, ensure_ascii=False)
            with self._append_lock:
                with open(self._log_path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except OSError:
            pass
