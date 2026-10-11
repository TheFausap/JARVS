"""Self-model tests — the intent gate (is_self_query), heading-based chunking, cosine
recall with the relevance floor, and the [SELF] block render. A deterministic bag-of-
words embedder stands in for sentence-transformers so no model is loaded."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jarvs.self_model import SelfModel, is_self_query  # noqa: E402

# A tiny deterministic embedder: one-hot over a fixed vocab. Shared words → high cosine.
_VOCAB = ["memory", "voice", "dream", "background", "claude", "architecture",
          "model", "routine", "token", "gpu"]


def _embed(text):
    t = (text or "").lower()
    return [1.0 if w in t else 0.0 for w in _VOCAB]


_DOSSIER = """# What JARVS is

JARVS is a local assistant built on a fixed base model with external components.

# Memory

JARVS keeps episodic and semantic memory, recalled by similarity. The memory layer
is degradable.

# Voice

JARVS speaks replies with a voice backend.

# The Claude bridge

JARVS can delegate coding to claude with a token budget.
"""


class IntentTests(unittest.TestCase):
    def test_positive(self):
        for t in ["how are you built?",
                  "what's your architecture?",
                  "describe yourself",
                  "tell me about your memory",
                  "how do you work?",
                  "how does your memory work?",
                  "what model are you running on?",
                  "what are your capabilities?",
                  "who are you?",
                  "are you an llm?"]:
            self.assertTrue(is_self_query(t), t)

    def test_negative(self):
        for t in ["how are you?",
                  "how are you doing today?",
                  "what are you doing?",
                  "how do I build a web scraper?",
                  "tell me about routines in general",
                  "what's the cpu load?",
                  "what model should I download?"]:
            self.assertFalse(is_self_query(t), t)


class SelfModelTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        with open(os.path.join(self.dir, "architecture.md"), "w", encoding="utf-8") as f:
            f.write(_DOSSIER)
        self.sm = SelfModel(self.dir, _embed, min_sim=0.2)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_chunks_by_heading(self):
        headings = {c["heading"] for c in self.sm.chunks}
        self.assertEqual(headings,
                         {"What JARVS is", "Memory", "Voice", "The Claude bridge"})

    def test_recall_returns_relevant_chunk(self):
        hits = self.sm.recall("how does your memory work")
        self.assertTrue(hits)
        self.assertEqual(hits[0]["heading"], "Memory")

    def test_recall_relevance_floor_filters_unrelated(self):
        # A query sharing no vocab with any chunk embeds to the zero vector → nothing.
        self.assertEqual(self.sm.recall("xyzzy frobnicate"), [])

    def test_self_block_render_and_empty(self):
        block = self.sm.self_block("tell me about your voice")
        self.assertTrue(block.startswith("[SELF]"))
        self.assertIn("Voice", block)
        self.assertEqual(self.sm.self_block("xyzzy frobnicate"), "")

    def test_stats_and_reload(self):
        self.assertIn("chunk(s)", self.sm.stats())
        self.assertIn("architecture.md", self.sm.stats())
        n = len(self.sm.chunks)
        self.sm.load()
        self.assertEqual(len(self.sm.chunks), n)

    def test_missing_dir_is_empty(self):
        sm = SelfModel(os.path.join(self.dir, "nope"), _embed)
        self.assertEqual(sm.chunks, [])
        self.assertEqual(sm.recall("memory"), [])
        self.assertIn("no dossier", sm.stats())


if __name__ == "__main__":
    unittest.main()
