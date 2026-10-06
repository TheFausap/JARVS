"""
memory.py — JARVS's external memory (the recall seam).

This is the capability the fixed base model cannot supply unaided: durable recall
across sessions. It composes three vendored pieces:

  * two :class:`~jarvs.vector_store.SQLiteVectorStore` collections —
    ``episodic`` (what was said, turn by turn) and ``semantic`` (distilled facts);
  * a sentence-transformers embedder (lazy-loaded, unit-normalized vectors so the
    stores' L2 distance is monotonic with cosine);
  * a :class:`~jarvs.assoc_memory.LinkGraph` of self-drawn associative edges.

It is **best-effort and degradable**: if sentence-transformers (or the model) is not
available, the whole layer disables itself — ``remember_*`` become no-ops and
``recall_block`` returns "" — so the chat spine keeps working without the dependency,
exactly as a reply must never be broken by a bookkeeping subsystem.

The recall block it renders is appended to the system head in
``JarvsSession.build_system``; the exchange recording hangs off the post-reply seam.
"""

from __future__ import annotations

import math
import time
import uuid

from . import config
from .assoc_memory import LinkGraph, weave
from .vector_store import SQLiteVectorStore


def _cosine_from_l2(dist: float) -> float:
    """For unit-norm vectors, ||a-b||^2 = 2(1 - cos). Invert to recover cosine."""
    return 1.0 - (dist * dist) / 2.0


class Memory:
    def __init__(self, *, data_dir=None, embed_model: str = "",
                 generate_fn=None):
        self.data_dir = str(data_dir or config.DATA_DIR)
        self.embed_model_name = embed_model or config.EMBED_MODEL
        # generate_fn(system, user) -> str, used only by the dream pass; optional.
        self._generate = generate_fn
        self._embedder = None
        self._dim = 0
        self.enabled = False
        self.episodic: SQLiteVectorStore | None = None
        self.semantic: SQLiteVectorStore | None = None
        self.links: LinkGraph | None = None
        self._init_error = ""
        if config.MEMORY_ENABLED:
            self._try_init()

    # -- lazy, degradable initialization -------------------------------------- #

    def _try_init(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:  # noqa: BLE001 — missing dep must not crash the spine
            self._init_error = f"sentence-transformers unavailable: {e}"
            return
        try:
            device = config.EMBED_DEVICE or None  # "" -> let the library choose
            self._embedder = SentenceTransformer(self.embed_model_name, device=device)
            # Method was renamed across ST versions; support both.
            dim_fn = (getattr(self._embedder, "get_embedding_dimension", None)
                      or self._embedder.get_sentence_embedding_dimension)
            self._dim = int(dim_fn())
            import os
            self.episodic = SQLiteVectorStore(
                os.path.join(self.data_dir, "episodic.sqlite3"), self._dim, "episodic")
            self.semantic = SQLiteVectorStore(
                os.path.join(self.data_dir, "semantic.sqlite3"), self._dim, "semantic")
            self.links = LinkGraph(os.path.join(self.data_dir, "links.json"))
            self.enabled = True
        except Exception as e:  # noqa: BLE001
            self._init_error = f"memory init failed: {e}"
            self.enabled = False

    def _embed(self, text: str) -> list[float]:
        # Unit-normalized so the stores' L2 ranking matches cosine and _cosine_from_l2
        # recovers a true cosine for the relevance floor.
        vec = self._embedder.encode([text], normalize_embeddings=True)[0]
        return [float(x) for x in vec]

    def _store_for(self, kind: str) -> SQLiteVectorStore | None:
        return self.semantic if kind == "semantic" else self.episodic

    # -- writes --------------------------------------------------------------- #

    def remember_exchange(self, user_text: str, assistant_text: str) -> str | None:
        """Store one conversation turn as an episodic memory. Returns its id, or None
        when memory is disabled / the write fails (never raises)."""
        user_text = (user_text or "").strip()
        assistant_text = (assistant_text or "").strip()
        if not self.enabled or not user_text:
            return None
        doc = f"User: {user_text}\nJARVS: {assistant_text}".strip()
        return self._add(self.episodic, doc, {"kind": "episodic", "ts": time.time()})

    def remember_fact(self, text: str, source: str = "user") -> str | None:
        """Store a distilled, durable fact as a semantic memory."""
        text = (text or "").strip()
        if not self.enabled or not text:
            return None
        return self._add(self.semantic, text,
                         {"kind": "semantic", "source": source, "ts": time.time()})

    def _add(self, store, doc: str, meta: dict) -> str | None:
        try:
            mid = uuid.uuid4().hex
            store.add(ids=[mid], documents=[doc],
                      embeddings=[self._embed(doc)], metadatas=[meta])
            return mid
        except Exception:  # noqa: BLE001
            return None

    # -- reads ---------------------------------------------------------------- #

    def recall(self, query: str, k: int | None = None) -> list[dict]:
        """Top memories across both stores for `query`, above the relevance floor,
        most-relevant first. Each item: {id, text, kind, score, ts}."""
        query = (query or "").strip()
        if not self.enabled or not query:
            return []
        k = config.RECALL_K if k is None else k
        try:
            qv = self._embed(query)
        except Exception:  # noqa: BLE001
            return []
        hits: list[dict] = []
        for store in (self.episodic, self.semantic):
            if store is None or store.count() == 0:
                continue
            try:
                res = store.query(query_embeddings=[qv], n_results=k,
                                  include=["documents", "metadatas"])
            except Exception:  # noqa: BLE001
                continue
            ids = (res.get("ids") or [[]])[0]
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
            for mid, doc, meta, dist in zip(ids, docs, metas, dists):
                score = _cosine_from_l2(float(dist))
                if score < config.RECALL_MIN:
                    continue
                hits.append({"id": mid, "text": doc, "kind": (meta or {}).get("kind"),
                             "score": round(score, 3), "ts": (meta or {}).get("ts")})
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:k]

    def recall_block(self, query: str) -> str:
        """Render the ``[MEMORY]`` block for the system head. Empty when nothing
        relevant is recalled, so an untrained JARVS behaves exactly as before."""
        hits = self.recall(query)
        if not hits:
            return ""
        lines = ["[MEMORY]",
                 "Relevant recollections from earlier (most relevant first). Use them "
                 "only if they bear on the current message; do not recite them:"]
        for h in hits:
            when = _ago(h.get("ts"))
            tag = "fact" if h["kind"] == "semantic" else when
            text = " ".join(h["text"].split())
            if len(text) > 400:
                text = text[:397].rstrip() + "…"
            lines.append(f"- ({tag}) {text}")
        return "\n".join(lines)

    # -- dream (associative consolidation) ------------------------------------ #

    def dream(self, sample: int = 24, max_links: int = 8) -> dict:
        """Draw non-obvious associative links across a recent sample of memories.
        Requires a generate_fn; no-op otherwise. Best-effort."""
        if not self.enabled or self._generate is None or self.links is None:
            return {"added": 0, "considered": 0, "skipped": "disabled"}
        candidates = []
        idx = 1
        for name, store in (("episodic", self.episodic), ("semantic", self.semantic)):
            got = store.get(include=["documents"], limit=sample)
            for mid, doc in zip(got.get("ids") or [], got.get("documents") or []):
                candidates.append({"idx": idx, "id": mid, "store": name,
                                   "text": " ".join((doc or "").split())[:200]})
                idx += 1
        try:
            summary = weave(self.links, candidates, self._generate, max_links=max_links)
            self.links.save()
            return summary
        except Exception as e:  # noqa: BLE001
            return {"added": 0, "considered": len(candidates), "error": str(e)}

    # -- introspection -------------------------------------------------------- #

    def status(self) -> str:
        if not self.enabled:
            return f"memory: off ({self._init_error or 'disabled'})"
        return (f"memory: on (episodic={self.episodic.count()} "
                f"semantic={self.semantic.count()} "
                f"links={self.links.stats()['count']} · {self.embed_model_name})")

    def stats(self) -> dict:
        if not self.enabled:
            return {"enabled": False, "error": self._init_error}
        return {"enabled": True, "dim": self._dim,
                "episodic": self.episodic.count(),
                "semantic": self.semantic.count(),
                "links": self.links.stats()}


def _ago(ts) -> str:
    try:
        delta = time.time() - float(ts)
    except (TypeError, ValueError):
        return "earlier"
    if delta < 90:
        return "just now"
    for unit, secs in (("d", 86400), ("h", 3600), ("m", 60)):
        if delta >= secs:
            return f"{int(delta // secs)}{unit} ago"
    return "just now"
