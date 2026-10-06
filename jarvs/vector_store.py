"""vector_store.py — a crash-proof, dependency-light vector collection.

A ``SQLiteVectorStore`` is a Chroma-compatible collection (same method names,
arguments and return shapes) backed by plain SQLite (WAL) for persistence and a
brute-force cosine KNN in NumPy over an in-memory matrix. At JARVS's scale (a few
thousand 384-dim vectors) that is sub-millisecond, and — crucially — every failure
mode is a catchable Python exception (``sqlite3.Error``, ``MemoryError``), never a
process-killing native segfault. Vendored from JCN3690.

Chroma-compatible surface (only what the memory layer uses):
    count() -> int
    add(ids=[...], documents=[...], embeddings=[...], metadatas=[...])
    update(ids=[i], documents=?, embeddings=?, metadatas=?)   # partial
    delete(ids=[...])
    get(ids=None, where=None, include=[...], limit=None)      # FLAT lists out
    query(query_embeddings=[v], n_results=k, where=None, include=[...])  # NESTED

Return-shape notes (matched to chroma so callers are unchanged):
  * get(...) -> {"ids": [...], "documents": [...]|None, "metadatas": [...]|None,
                 "embeddings": [...]|None}. A key is a real list only when its field
    is in `include`; otherwise the value is None (callers use dict.get()).
  * query(...) -> the same keys but each is a list-of-lists (one inner list per
    query embedding), plus "distances". Callers index [0].
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading

import numpy as np

# Chroma's default include when the caller passes none.
_DEFAULT_INCLUDE = ("documents", "metadatas")


class SQLiteVectorStore:
    """One collection (episodic or semantic) backed by a single SQLite file plus an
    in-memory NumPy matrix for cosine KNN. Thread-safe: every public method holds one
    reentrant lock, so concurrent access from the turn thread and the voice thread is
    serialized in Python — and can only ever raise, never segfault."""

    def __init__(self, path: str, dim: int, name: str = ""):
        self.path = path
        self.dim = int(dim)
        self.name = name or os.path.splitext(os.path.basename(path))[0]
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # check_same_thread=False is safe because every access is under self._lock.
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS memories ("
            " id TEXT PRIMARY KEY, document TEXT, metadata TEXT, embedding BLOB)")
        self._db.commit()
        # In-memory KNN cache, kept in sync with the table.
        self._ids: list[str] = []
        self._idx: dict[str, int] = {}
        self._mat: np.ndarray = np.zeros((0, self.dim), dtype=np.float32)
        self._norms: np.ndarray = np.zeros((0,), dtype=np.float32)
        self._load_cache()

    # ------------------------------------------------------------------ internals
    def _load_cache(self) -> None:
        rows = self._db.execute("SELECT id, embedding FROM memories").fetchall()
        self._ids = [r[0] for r in rows]
        self._idx = {i: n for n, i in enumerate(self._ids)}
        if rows:
            self._mat = np.stack([self._blob_to_vec(r[1]) for r in rows]).astype(
                np.float32, copy=False)
        else:
            self._mat = np.zeros((0, self.dim), dtype=np.float32)
        self._norms = np.linalg.norm(self._mat, axis=1) if len(self._mat) \
            else np.zeros((0,), dtype=np.float32)

    def _rebuild_matrix(self) -> None:
        """Recompute the KNN matrix from self._ids by reading their blobs. Called
        after a mutation changes membership (add/delete)."""
        if not self._ids:
            self._mat = np.zeros((0, self.dim), dtype=np.float32)
            self._norms = np.zeros((0,), dtype=np.float32)
            return
        vecs = []
        for i in self._ids:
            row = self._db.execute(
                "SELECT embedding FROM memories WHERE id=?", (i,)).fetchone()
            vecs.append(self._blob_to_vec(row[0]))
        self._mat = np.stack(vecs).astype(np.float32, copy=False)
        self._norms = np.linalg.norm(self._mat, axis=1)

    @staticmethod
    def _vec_to_blob(vec) -> bytes:
        return np.asarray(vec, dtype=np.float32).tobytes()

    def _blob_to_vec(self, blob) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32).reshape(-1)

    @staticmethod
    def _match_where(meta: dict, where: dict | None) -> bool:
        """Evaluate a chroma-style where filter against one metadata dict. Supports
        bare {key: value} equality, {key: {"$eq": value}}, and top-level
        {"$and": [...]} / {"$or": [...]}."""
        if not where:
            return True
        if "$and" in where:
            return all(SQLiteVectorStore._match_where(meta, c) for c in where["$and"])
        if "$or" in where:
            return any(SQLiteVectorStore._match_where(meta, c) for c in where["$or"])
        for key, cond in where.items():
            actual = (meta or {}).get(key)
            if isinstance(cond, dict):
                if "$eq" in cond and actual != cond["$eq"]:
                    return False
                if "$ne" in cond and actual == cond["$ne"]:
                    return False
            elif actual != cond:
                return False
        return True

    # ------------------------------------------------------------------- reads
    def count(self) -> int:
        with self._lock:
            return len(self._ids)

    def get(self, ids=None, where=None, include=None, limit=None) -> dict:
        include = list(include) if include is not None else list(_DEFAULT_INCLUDE)
        want_doc = "documents" in include
        want_meta = "metadatas" in include
        want_emb = "embeddings" in include
        with self._lock:
            if ids is not None:
                rows = []
                for i in ids:
                    r = self._db.execute(
                        "SELECT id, document, metadata, embedding FROM memories "
                        "WHERE id=?", (i,)).fetchone()
                    if r is not None:
                        rows.append(r)
            else:
                rows = self._db.execute(
                    "SELECT id, document, metadata, embedding FROM memories").fetchall()
            out_ids, out_doc, out_meta, out_emb = [], [], [], []
            for rid, doc, meta_json, blob in rows:
                meta = json.loads(meta_json) if meta_json else {}
                if not self._match_where(meta, where):
                    continue
                out_ids.append(rid)
                out_doc.append(doc)
                out_meta.append(meta)
                if want_emb:
                    out_emb.append(self._blob_to_vec(blob).tolist())
                if limit is not None and len(out_ids) >= limit:
                    break
        return {"ids": out_ids,
                "documents": out_doc if want_doc else None,
                "metadatas": out_meta if want_meta else None,
                "embeddings": out_emb if want_emb else None}

    def query(self, query_embeddings, n_results=10, where=None, include=None) -> dict:
        include = list(include) if include is not None else list(_DEFAULT_INCLUDE)
        want_doc = "documents" in include
        want_meta = "metadatas" in include
        want_emb = "embeddings" in include
        ids_out, doc_out, meta_out, emb_out, dist_out = [], [], [], [], []
        with self._lock:
            for q in query_embeddings:
                qv = np.asarray(q, dtype=np.float32).reshape(-1)
                order = self._knn_order(qv, where)[:n_results]
                r_ids, r_doc, r_meta, r_emb, r_dist = [], [], [], [], []
                for pos, dist in order:
                    rid = self._ids[pos]
                    r_ids.append(rid)
                    r_dist.append(dist)
                    if want_doc or want_meta:
                        row = self._db.execute(
                            "SELECT document, metadata FROM memories WHERE id=?",
                            (rid,)).fetchone()
                        r_doc.append(row[0] if row else "")
                        r_meta.append(json.loads(row[1]) if row and row[1] else {})
                    if want_emb:
                        r_emb.append(self._mat[pos].tolist())
                ids_out.append(r_ids)
                doc_out.append(r_doc)
                meta_out.append(r_meta)
                emb_out.append(r_emb)
                dist_out.append(r_dist)
        return {"ids": ids_out,
                "documents": doc_out if want_doc else None,
                "metadatas": meta_out if want_meta else None,
                "embeddings": emb_out if want_emb else None,
                "distances": dist_out}

    def _knn_order(self, qv: np.ndarray, where: dict | None):
        """Return [(row_index, l2_distance), ...] nearest-first. Cosine is used for
        ranking; the reported distance is L2 for shape compatibility (callers can
        recompute cosine from the embeddings)."""
        if len(self._mat) == 0:
            return []
        qn = np.linalg.norm(qv)
        if qn == 0:
            return []
        denom = self._norms * qn
        with np.errstate(divide="ignore", invalid="ignore"):
            sims = (self._mat @ qv) / denom
        sims = np.nan_to_num(sims, nan=-1.0)
        if where:  # mask out rows whose metadata fails the filter
            keep = np.array([self._match_where(self._meta_at(p), where)
                             for p in range(len(self._ids))])
            sims = np.where(keep, sims, -np.inf)
        n = min(len(sims), max(1, int((sims > -np.inf).sum())))
        top = np.argpartition(-sims, n - 1)[:n] if n < len(sims) else np.arange(len(sims))
        top = top[np.argsort(-sims[top])]
        out = []
        for p in top:
            if sims[p] == -np.inf:
                continue
            d = float(np.linalg.norm(self._mat[p] - qv))
            out.append((int(p), d))
        return out

    def _meta_at(self, pos: int) -> dict:
        row = self._db.execute("SELECT metadata FROM memories WHERE id=?",
                               (self._ids[pos],)).fetchone()
        # Single-column SELECT -> metadata is row[0]. (The upstream JCN3690 copy read
        # row[1] here, which crashed any query() that passed a `where` filter.)
        return json.loads(row[0]) if row and row[0] else {}

    # ------------------------------------------------------------------- writes
    def add(self, ids, documents, embeddings, metadatas) -> None:
        with self._lock:
            new_rows = []
            for i, d, e, m in zip(ids, documents, embeddings, metadatas):
                self._db.execute(
                    "INSERT OR REPLACE INTO memories (id, document, metadata, embedding)"
                    " VALUES (?,?,?,?)",
                    (i, d, json.dumps(m or {}), self._vec_to_blob(e)))
                new_rows.append((i, np.asarray(e, dtype=np.float32).reshape(-1)))
            self._db.commit()
            # Extend the cache in place (append-only fast path); rebuild if any id
            # already existed (a replace would desync positions).
            append_vecs, replaced = [], False
            for i, v in new_rows:
                if i in self._idx:
                    replaced = True
                    break
                self._idx[i] = len(self._ids)
                self._ids.append(i)
                append_vecs.append(v)
            if replaced:
                self._load_cache()
            elif append_vecs:
                block = np.stack(append_vecs).astype(np.float32, copy=False)
                self._mat = np.vstack([self._mat, block]) if len(self._mat) else block
                self._norms = np.concatenate(
                    [self._norms, np.linalg.norm(block, axis=1)])

    def update(self, ids, documents=None, embeddings=None, metadatas=None) -> None:
        with self._lock:
            emb_changed = False
            for n, i in enumerate(ids):
                sets, vals = [], []
                if documents is not None:
                    sets.append("document=?"); vals.append(documents[n])
                if metadatas is not None:
                    sets.append("metadata=?"); vals.append(json.dumps(metadatas[n] or {}))
                if embeddings is not None:
                    sets.append("embedding=?"); vals.append(self._vec_to_blob(embeddings[n]))
                    emb_changed = True
                if not sets:
                    continue
                vals.append(i)
                self._db.execute(
                    f"UPDATE memories SET {', '.join(sets)} WHERE id=?", vals)
            self._db.commit()
            if emb_changed:
                # A changed vector shifts the KNN cache; refresh the affected rows.
                for n, i in enumerate(ids):
                    if embeddings is not None and i in self._idx:
                        v = np.asarray(embeddings[n], dtype=np.float32).reshape(-1)
                        self._mat[self._idx[i]] = v
                        self._norms[self._idx[i]] = np.linalg.norm(v)

    def delete(self, ids) -> None:
        with self._lock:
            for i in ids:
                self._db.execute("DELETE FROM memories WHERE id=?", (i,))
            self._db.commit()
            self._ids = [i for i in self._ids if i not in set(ids)]
            self._idx = {i: n for n, i in enumerate(self._ids)}
            self._rebuild_matrix()

    # ------------------------------------------------------------------ health
    def integrity_check(self) -> str:
        """SQLite's own corruption check. Returns 'ok' on a healthy file, else the
        first problem SQLite reports."""
        with self._lock:
            try:
                rows = self._db.execute("PRAGMA quick_check").fetchall()
                return rows[0][0] if rows else "unknown"
            except Exception as e:
                return f"error: {e}"

    def self_check(self) -> dict:
        """Structural health of this store: on-disk row count vs the in-memory KNN
        cache/matrix (a mismatch means a desync bug), embedding-dimension sanity, and
        the SQLite integrity check."""
        with self._lock:
            try:
                table_count = self._db.execute(
                    "SELECT COUNT(*) FROM memories").fetchone()[0]
            except Exception as e:
                return {"backend": "sqlite", "error": str(e), "integrity": "error"}
            bad_dim = 0
            for (blob,) in self._db.execute("SELECT embedding FROM memories"):
                if blob is None or (len(blob) // 4) != self.dim:
                    bad_dim += 1
            cache_rows = len(self._ids)
            matrix_rows = int(self._mat.shape[0]) if self._mat.ndim == 2 else 0
            return {
                "backend": "sqlite",
                "count": int(table_count),
                "cache_rows": cache_rows,
                "matrix_rows": matrix_rows,
                "cache_consistent": table_count == cache_rows == matrix_rows,
                "dim": self.dim,
                "dim_mismatches": bad_dim,
                "integrity": self.integrity_check(),
            }

    def peek(self, limit: int = 10) -> dict:
        return self.get(include=["documents", "metadatas"], limit=limit)

    def modify(self, *args, **kwargs) -> None:   # no-op: collection rename unused
        pass
