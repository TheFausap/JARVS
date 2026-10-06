"""Memory-layer tests: the SQLite vector store (no embedder needed) and the Memory
manager's recall round-trip + relevance floor + graceful degradation."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jarvs.vector_store import SQLiteVectorStore  # noqa: E402


class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.store = SQLiteVectorStore(os.path.join(self.dir, "t.sqlite3"), dim=3)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.dir)

    def test_add_count_query(self):
        self.store.add(ids=["a", "b"], documents=["apple", "banana"],
                       embeddings=[[1, 0, 0], [0, 1, 0]],
                       metadatas=[{"kind": "x"}, {"kind": "y"}])
        self.assertEqual(self.store.count(), 2)
        res = self.store.query([[1, 0, 0]], n_results=1, include=["documents"])
        self.assertEqual(res["ids"][0][0], "a")
        self.assertEqual(res["documents"][0][0], "apple")

    def test_where_filter(self):
        self.store.add(ids=["a", "b"], documents=["x", "y"],
                       embeddings=[[1, 0, 0], [1, 0, 0]],
                       metadatas=[{"kind": "keep"}, {"kind": "drop"}])
        res = self.store.query([[1, 0, 0]], n_results=5, where={"kind": "keep"})
        self.assertEqual(res["ids"][0], ["a"])

    def test_delete_and_persist(self):
        self.store.add(ids=["a"], documents=["x"], embeddings=[[1, 0, 0]],
                       metadatas=[{}])
        self.store.delete(["a"])
        self.assertEqual(self.store.count(), 0)
        # reopen the same file: empty, and not corrupt
        reopened = SQLiteVectorStore(self.store.path, dim=3)
        self.assertEqual(reopened.count(), 0)
        self.assertEqual(reopened.self_check()["integrity"], "ok")


class _StubEmbedder:
    """Deterministic 3-dim embeddings so Memory can be tested without a real model."""
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, normalize_embeddings=True):
        import numpy as np
        out = []
        for t in texts:
            v = np.array([t.lower().count("cat"), t.lower().count("dog"),
                          t.lower().count("sql")], dtype="float32")
            if not v.any():
                v[0] = 1e-6
            if normalize_embeddings:
                v = v / (np.linalg.norm(v) or 1.0)
            out.append(v)
        return out


class MemoryTests(unittest.TestCase):
    def _mem(self):
        from jarvs import memory
        m = memory.Memory.__new__(memory.Memory)  # bypass __init__ (no real model)
        m.data_dir = tempfile.mkdtemp()
        m.embed_model_name = "stub"
        m._generate = None
        m._embedder = _StubEmbedder()
        m._dim = 3
        m._init_error = ""
        m.episodic = SQLiteVectorStore(os.path.join(m.data_dir, "e.sqlite3"), 3)
        m.semantic = SQLiteVectorStore(os.path.join(m.data_dir, "s.sqlite3"), 3)
        from jarvs.assoc_memory import LinkGraph
        m.links = LinkGraph(os.path.join(m.data_dir, "l.json"))
        m.enabled = True
        return m

    def test_recall_round_trip(self):
        m = self._mem()
        m.remember_fact("I have a cat named Mona.")
        m.remember_fact("SQL databases store rows.")
        hits = m.recall("tell me about my cat")
        self.assertTrue(hits)
        self.assertIn("cat", hits[0]["text"].lower())

    def test_relevance_floor_excludes_unrelated(self):
        m = self._mem()
        m.remember_fact("I have a dog.")
        # Query about SQL has an orthogonal embedding -> cosine 0 -> below floor.
        self.assertEqual(m.recall_block("something about sql"), "")

    def test_disabled_is_noop(self):
        from jarvs import memory
        m = memory.Memory.__new__(memory.Memory)
        m.enabled = False
        m._init_error = "no model"
        self.assertEqual(m.recall_block("x"), "")
        self.assertIsNone(m.remember_fact("x"))
        self.assertEqual(m.recall("x"), [])


if __name__ == "__main__":
    unittest.main()
