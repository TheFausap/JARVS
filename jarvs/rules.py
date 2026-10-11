"""
rules.py — standing rules JARVS holds for itself.

Durable, user-authored directives that are injected into the system head on every turn
(a ``[STANDING RULES]`` block), so the user can shape JARVS's behaviour in plain
language ("from now on, always check the web before answering about fast-moving tech")
and have it persist across sessions — distinct from a one-off memory fact, which is
recalled only when relevant.

One rule is also machine-actionable: ``web_verify``. When the user sets a rule about
checking the internet / verifying current information, the store flips this flag, and the
reply path lowers its bar for running a live web search (see
:func:`jarvs.web_access.should_search`). The directive text makes JARVS *know* the policy;
the flag makes the engine actually *act* on it.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid

# --- intent detection (pure, testable) ---------------------------------------- #
# Imperative-anchored, like the remember/schedule detectors: a request to adopt a
# standing rule, not a passing mention of the word "rule".

_RULE_RE = re.compile(
    r"(?:^|[.!?;,]\s*|\bplease\s+|\bjarvs[,:]?\s+)"
    r"(?:(?:make|set|add|create|put)\s+(?:it|that|this)?\s*(?:as\s+)?(?:a|an)?\s*"
    r"(?:standing\s+)?rule"                      # make/set/put it (as) a (standing) rule
    r"|as\s+a\s+standing\s+rule"
    r"|from now on|going forward|henceforth)",
    re.I)

# When a rule concerns checking the live web / verifying currency, flip web_verify.
_WEB_VERIFY_RE = re.compile(
    r"\b(?:web|internet|online|search|google|look (?:it|them|that) up|verify|"
    r"fact[- ]?check|up[- ]?to[- ]?date|current|latest|updated information|"
    r"before (?:you )?answer)\b", re.I)


def is_rule_request(text: str) -> bool:
    """True when the user is asking JARVS to adopt a durable standing rule."""
    return bool(_RULE_RE.search(text or ""))


def implies_web_verify(text: str) -> bool:
    """True when the rule is about checking the web / verifying current information."""
    return bool(_WEB_VERIFY_RE.search(text or ""))


class RulesStore:
    """Persisted standing directives plus the actionable ``web_verify`` flag."""

    def __init__(self, store_path: str):
        self._path = store_path
        self._lock = threading.RLock()
        self.directives: list[dict] = []     # [{id, text, ts}]
        self.web_verify: bool = False
        self._load()

    # -- persistence ---------------------------------------------------------- #

    def _load(self) -> None:
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return
        if isinstance(data, dict):
            self.directives = [d for d in data.get("directives", [])
                               if isinstance(d, dict) and d.get("text")]
            self.web_verify = bool(data.get("web_verify", False))

    def _save_locked(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
            tmp = f"{self._path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"directives": self.directives[-40:],
                           "web_verify": self.web_verify}, fh, indent=2)
            os.replace(tmp, self._path)
        except OSError:
            pass

    # -- mutations ------------------------------------------------------------ #

    def add(self, text: str) -> dict | None:
        text = " ".join((text or "").split()).strip()[:400]
        if len(text) < 4:
            return None
        with self._lock:
            # De-dupe near-identical directives (case-insensitive).
            for d in self.directives:
                if d["text"].casefold() == text.casefold():
                    return dict(d)
            rule = {"id": uuid.uuid4().hex[:8], "text": text, "ts": time.time()}
            self.directives.append(rule)
            del self.directives[:-40]
            self._save_locked()
            return dict(rule)

    def remove(self, rule_id: str) -> bool:
        with self._lock:
            before = len(self.directives)
            self.directives = [d for d in self.directives if d["id"] != rule_id]
            if len(self.directives) != before:
                self._save_locked()
                return True
            return False

    def set_web_verify(self, on: bool) -> None:
        with self._lock:
            self.web_verify = bool(on)
            self._save_locked()

    def clear(self) -> None:
        with self._lock:
            self.directives = []
            self.web_verify = False
            self._save_locked()

    # -- reads ---------------------------------------------------------------- #

    def list(self) -> list[dict]:
        with self._lock:
            return [dict(d) for d in self.directives]

    def block(self) -> str:
        """Render the ``[STANDING RULES]`` block for the system head, or "" if none."""
        with self._lock:
            if not self.directives:
                return ""
            lines = ["[STANDING RULES]",
                     "Durable directives the user has set; follow them every turn:"]
            lines += [f"- {d['text']}" for d in self.directives]
            return "\n".join(lines)

    def status(self) -> str:
        with self._lock:
            return (f"rules: {len(self.directives)} standing directive(s) · "
                    f"web-verify {'on' if self.web_verify else 'off'}")
