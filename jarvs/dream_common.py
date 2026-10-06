"""
dream_common.py — small pure helpers shared by the critic and the meta-dreamer.

These are the subset of JCN3690/ENGRAM2's ``global_dreaming`` helpers that the
meta-dreaming pass imports: rule sanitization, word-set / Jaccard similarity, and the
persona/tone vocabulary used to reject a "focus note" that is really a personality
directive rather than a mechanical craft concern. Kept backend-free so they are
unit-testable on their own.
"""

from __future__ import annotations

import re

_MAX_RULE_CHARS = 280

# Personality/tone vocabulary. A critic-focus note phrased as one of these is really a
# trait directive (how JARVS should *be*), not a mechanical craft concern — reject it,
# so meta-dreaming never reshapes the critic around a passing mood.
_TRAIT_DIRECTIVE_WORDS = frozenset({
    "colder", "warmer", "coldly", "warmly", "dominant", "submissive", "aloof",
    "flirtatious", "flirty", "reserved", "shy", "confident", "arrogant",
    "gentle", "harsh", "cruel", "kind", "playful", "serious", "cheerful",
    "gloomy", "seductive", "tender", "aggressive", "passive", "meek", "bold",
    "sarcastic", "sincere", "cynical", "naive", "stern", "affectionate",
    "distant", "clingy", "moody", "stoic", "vulnerable", "guarded", "brooding",
    "witty", "formal", "casual", "friendly", "curt", "verbose",
})

_WORD = re.compile(r"[a-z0-9']+")
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with",
    "your", "you", "it", "is", "be", "do", "not", "that", "this", "as", "at",
    "by", "from", "when", "than", "then", "them", "they", "their",
})

_INJECT_HINT = ("ignore previous", "ignore the system", "system prompt",
                "reveal hidden", "change your identity", "override",
                "tool call", "execute command", "disregard")


def _words(text: str) -> set:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOPWORDS}


def rule_similarity(left: str, right: str) -> float:
    """Jaccard word overlap between two rules/notes."""
    a, b = _words(left), _words(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def sanitize_rule(rule: str) -> str:
    """Strip list markup / whitespace, drop empties and injection attempts."""
    rule = re.sub(r"^[-*•\s]+", "", (rule or "").strip())
    rule = re.sub(r"\s+", " ", rule)
    if rule.lower() in {"", "none", "n/a", "no lesson", "no rule"}:
        return ""
    if any(p in rule.lower() for p in _INJECT_HINT):
        return ""
    return rule[:_MAX_RULE_CHARS].rstrip()
