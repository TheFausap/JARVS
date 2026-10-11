"""
self_model.py — JARVS's model of its OWN architecture (the self-memory).

A small, curated dossier (``self/*.md`` / ``*.txt``) describing how JARVS is built,
loaded, chunked by heading, embedded in-memory, and queried by cosine similarity. It
is kept deliberately OUTSIDE the episodic/semantic stores, so it can never leak into
ordinary recall: the spine consults it only when the user is actually asking JARVS
about itself (an *intent* gate, :func:`is_self_query`), and even then injects only the
chunks that clear a relevance threshold for the question (a *relevance* gate).

This is EXTERNAL self-documentation, not introspective access to the model's weights.
JARVS may read and quote it, but it describes JARVS; it is not JARVS perceiving its own
parameters. The module owns loading + chunking + similarity; the embedding function is
injected (JARVS reuses its memory embedder), as is the decision of when to consult it.

Ported from JCN3690's ``self_model.py``; the intent gate and ``self_block`` render are
the JARVS additions, matching how :class:`~jarvs.memory.Memory` renders its recall block.
"""

from __future__ import annotations

import os
import re

import numpy as np

from . import config

# A markdown ATX heading (``#`` .. ``######``) starts a new section/chunk.
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*\S)\s*$")

# --- intent gate (pure, testable) --------------------------------------------- #
# When the user is actually asking JARVS about itself. High precision: a bare greeting
# ("how are you?") must NOT trigger, but "how are you built?" / "what's your
# architecture?" must.

_SELF_STRONG_RE = re.compile(
    r"\b(?:"
    r"how (?:are|were|r) (?:you|jarvs) (?:built|made|designed|structured|architected|"
    r"implemented|created|put together)|"
    r"how (?:do|does) (?:you|jarvs) (?:work|function|operate)|"
    r"how (?:do|does) your \w+ (?:work|run|operate|function)|"
    r"how (?:do|does) (?:you|jarvs) (?:store|handle|keep|manage) (?:memor|context|facts)|"
    r"what('?s| is| are) your (?:architecture|design|internals?|structure|stack|"
    r"pipeline|memory|model|backend|setup|make[- ]?up|components?|capabilit|voice|"
    r"routines?|jobs?)|"
    r"(?:describe|explain|tell me about) (?:yourself|your (?:architecture|design|"
    r"internals?|memory|components?|capabilit|stack|pipeline|setup|build|model|voice|"
    r"routines?|jobs?|dream\w*|background))|"
    r"your (?:architecture|internals?|components?)|"
    r"what (?:model|llm|backend) (?:are you|do you (?:use|run (?:on)?)|runs you|powers you)|"
    r"what are your (?:capabilit|features|components?|parts)|"
    r"are you (?:conscious|sentient|an? (?:ai|llm|bot|language model|assistant))"
    r")", re.I)

# A bare identity question, allowed only when short and not "what are you DOING".
_SELF_IDENTITY_RE = re.compile(
    r"\b(?:what|who)\s+(?:are|r)\s+(?:you|jarvs)\b"
    r"(?!\s+(?:doing|up|going|working|thinking|able|sure|planning|gonna|going to))",
    re.I)


def is_self_query(text: str) -> bool:
    """True when the user is asking about JARVS itself (its build, architecture,
    memory, model, capabilities, or identity)."""
    t = (text or "").strip()
    if _SELF_STRONG_RE.search(t):
        return True
    return bool(_SELF_IDENTITY_RE.search(t)) and len(t.split()) <= 6


class SelfModel:
    """The self-dossier as an in-memory, cosine-searchable index."""

    def __init__(self, docs_dir: str, embed_fn, *, recall_k: int = 4,
                 min_sim: float = 0.22, chunk_chars: int = 1400):
        self.docs_dir = docs_dir
        self._embed = embed_fn
        self.recall_k = max(1, int(recall_k))
        self.min_sim = float(min_sim)
        self.chunk_chars = max(400, int(chunk_chars))
        self.chunks: list[dict] = []            # {heading, source, text}
        self._matrix: np.ndarray | None = None  # chunk vectors, embedded lazily
        self.load()

    # -- loading -------------------------------------------------------------- #

    def load(self) -> None:
        """(Re)read every dossier file. Cheap — chunk vectors are (re)built lazily on
        the next recall, so an edited dossier is picked up via ``/self reload`` without
        paying the embedding cost until JARVS is actually asked about itself."""
        self.chunks = []
        self._matrix = None
        if not os.path.isdir(self.docs_dir):
            return
        for name in sorted(os.listdir(self.docs_dir)):
            if not name.lower().endswith((".md", ".markdown", ".txt", ".text")):
                continue
            try:
                with open(os.path.join(self.docs_dir, name), "r", encoding="utf-8") as fh:
                    raw = fh.read()
            except OSError:
                continue
            self.chunks.extend(self._chunk_file(raw, name))

    def _chunk_file(self, raw: str, source: str) -> list[dict]:
        """Split a heading-delimited doc into (heading, body) chunks; an overlong
        section is further split on blank lines so no chunk exceeds chunk_chars."""
        sections: list[tuple[str, list[str]]] = []
        heading, body = source, []
        for line in raw.splitlines():
            match = _HEADING_RE.match(line)
            if match:
                if any(b.strip() for b in body):
                    sections.append((heading, body))
                heading, body = match.group(2).strip(), []
            else:
                body.append(line)
        if any(b.strip() for b in body):
            sections.append((heading, body))
        out: list[dict] = []
        for head, lines in sections:
            text = "\n".join(lines).strip()
            if not text:
                continue
            for piece in self._split_long(text):
                out.append({"heading": head, "source": source, "text": piece})
        return out

    def _split_long(self, text: str) -> list[str]:
        if len(text) <= self.chunk_chars:
            return [text]
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        pieces, current = [], ""
        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if current and len(candidate) > self.chunk_chars:
                pieces.append(current)
                current = paragraph
            else:
                current = candidate
        if current:
            pieces.append(current)
        final: list[str] = []                   # a single monster paragraph: hard-split
        for piece in pieces:
            if len(piece) <= self.chunk_chars:
                final.append(piece)
            else:
                final.extend(piece[i:i + self.chunk_chars]
                             for i in range(0, len(piece), self.chunk_chars))
        return final

    # -- embedding ------------------------------------------------------------ #

    def _ensure_embedded(self) -> None:
        if self._matrix is not None or not self.chunks:
            return
        vectors = [np.asarray(self._embed(f"{c['heading']}\n{c['text']}"),
                              dtype=np.float32) for c in self.chunks]
        self._matrix = np.vstack(vectors) if vectors else None

    # -- recall --------------------------------------------------------------- #

    def recall(self, query: str, top_k: int | None = None,
               min_sim: float | None = None) -> list[dict]:
        """Top chunks by cosine similarity to ``query`` that clear ``min_sim``. Returns
        [] when nothing is relevant — the relevance gate that keeps the self-model out
        of a turn it does not actually inform."""
        if not self.chunks or not (query or "").strip():
            return []
        top_k = self.recall_k if top_k is None else top_k
        min_sim = self.min_sim if min_sim is None else min_sim
        self._ensure_embedded()
        if self._matrix is None:
            return []
        q = np.asarray(self._embed(query), dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []
        row_norms = np.linalg.norm(self._matrix, axis=1)
        denom = row_norms * q_norm
        sims = np.where(denom > 0, (self._matrix @ q) / denom, 0.0)
        results: list[dict] = []
        for idx in np.argsort(-sims)[:max(1, top_k)]:
            score = float(sims[int(idx)])
            if score < min_sim:
                break
            chunk = self.chunks[int(idx)]
            results.append({"heading": chunk["heading"], "source": chunk["source"],
                            "text": chunk["text"], "sim": round(score, 3)})
        return results

    def self_block(self, query: str) -> str:
        """Render the ``[SELF]`` block for the system head. Empty when nothing relevant
        is recalled, so a turn that doesn't ask about JARVS is untouched."""
        hits = self.recall(query)
        if not hits:
            return ""
        lines = ["[SELF]",
                 "Reference notes on how you (JARVS) are built — external documentation "
                 "about yourself, not perception of your own weights. Use them only to "
                 "answer the current question about yourself; quote naturally, do not "
                 "recite verbatim:"]
        for h in hits:
            text = " ".join(h["text"].split())
            if len(text) > 700:
                text = text[:697].rstrip() + "…"
            lines.append(f"- ({h['heading']}) {text}")
        return "\n".join(lines)

    # -- introspection -------------------------------------------------------- #

    def stats(self) -> str:
        if not self.chunks:
            return f"self-model: no dossier found in '{self.docs_dir}/'"
        sources = sorted({c["source"] for c in self.chunks})
        return (f"self-model: {len(self.chunks)} chunk(s) from {len(sources)} file(s) "
                f"[{', '.join(sources)}]")
