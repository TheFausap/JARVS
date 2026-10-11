"""Standing-rules tests — intent detection, web-verify inference, and the persisted
RulesStore (add/dedupe/remove/flag/block/persist)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jarvs.rules import (  # noqa: E402
    RulesStore, is_rule_request, implies_web_verify,
)


class IntentTests(unittest.TestCase):
    def test_positive(self):
        for t in ["put it as a rule to check always for updated information",
                  "make it a rule to always verify online",
                  "from now on, always check the web before answering",
                  "going forward, use metric units",
                  "please set a standing rule: cite your sources",
                  "add a rule to keep answers short"]:
            self.assertTrue(is_rule_request(t), t)

    def test_negative(self):
        for t in ["what are the rules?",
                  "the rule of thumb is to keep it simple",
                  "tell me about routines",
                  "what's the cpu load?"]:
            self.assertFalse(is_rule_request(t), t)


class WebVerifyInferenceTests(unittest.TestCase):
    def test_positive(self):
        for t in ["check always for updated information",
                  "verify current facts online",
                  "always search the web first",
                  "keep yourself up-to-date before you answer"]:
            self.assertTrue(implies_web_verify(t), t)

    def test_negative(self):
        for t in ["use metric units", "keep answers short", "be more formal"]:
            self.assertFalse(implies_web_verify(t), t)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "rules.json")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_add_dedupe_remove(self):
        s = RulesStore(self.path)
        r = s.add("Always cite sources.")
        self.assertIsNotNone(r)
        self.assertIsNotNone(s.add("ALWAYS CITE SOURCES."))   # dedupe (case-insensitive)
        self.assertEqual(len(s.list()), 1)
        self.assertTrue(s.remove(r["id"]))
        self.assertEqual(s.list(), [])
        self.assertFalse(s.remove(r["id"]))

    def test_too_short_rejected(self):
        self.assertIsNone(RulesStore(self.path).add("x"))

    def test_web_verify_flag_and_block(self):
        s = RulesStore(self.path)
        self.assertFalse(s.web_verify)
        s.set_web_verify(True)
        self.assertTrue(s.web_verify)
        self.assertEqual(s.block(), "")                        # no directives yet
        s.add("Verify current tech facts on the web before answering.")
        block = s.block()
        self.assertTrue(block.startswith("[STANDING RULES]"))
        self.assertIn("Verify current tech facts", block)

    def test_persistence(self):
        s = RulesStore(self.path)
        s.add("Keep replies under 50 words.")
        s.set_web_verify(True)
        s2 = RulesStore(self.path)
        self.assertEqual(len(s2.list()), 1)
        self.assertTrue(s2.web_verify)
        self.assertIn("50 words", s2.list()[0]["text"])

    def test_clear(self):
        s = RulesStore(self.path)
        s.add("Some rule here."); s.set_web_verify(True)
        s.clear()
        self.assertEqual(s.list(), [])
        self.assertFalse(s.web_verify)


if __name__ == "__main__":
    unittest.main()
