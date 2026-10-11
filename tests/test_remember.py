"""Tests for the plain-language 'remember this' path — intent detection, inline-fact
extraction, and the resolve→distill→store orchestration (generate + store stubbed)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jarvs.memory import (  # noqa: E402
    is_remember_request, extract_inline_fact, distill_fact, remember_from_request,
)


class IntentTests(unittest.TestCase):
    def test_positive(self):
        for t in ["store this into memory so we can chat about it later",
                  "remember this",
                  "remember that the deploy runs at 02:00",
                  "please save this",
                  "note this down",
                  "keep this in mind",
                  "jarvs, remember to rotate the key monthly",
                  "ok, memorize that the GPU is a GTX 1660"]:
            self.assertTrue(is_remember_request(t), t)

    def test_negative(self):
        for t in ["I remember that film we watched",
                  "what do you remember about me?",
                  "do you have a memory system?",
                  "tell me about routines in general",
                  "what's the cpu load?"]:
            self.assertFalse(is_remember_request(t), t)


class InlineFactTests(unittest.TestCase):
    def test_inline(self):
        self.assertEqual(extract_inline_fact("remember that the deploy runs at 02:00"),
                         "the deploy runs at 02:00")
        self.assertEqual(extract_inline_fact("remember to rotate the key monthly"),
                         "rotate the key monthly")
        self.assertEqual(extract_inline_fact("note that the GPU is a GTX 1660"),
                         "the GPU is a GTX 1660")

    def test_referential_returns_empty(self):
        for t in ["remember this", "store this into memory so we can chat later",
                  "save it", "remember what you just said", "note this down"]:
            self.assertEqual(extract_inline_fact(t), "", t)


class DistillTests(unittest.TestCase):
    def test_distills_via_model(self):
        got = distill_fact("the deploy runs at 2am", "",
                           lambda s, u, r: "The user's deploy runs at 02:00 daily.")
        self.assertEqual(got, "The user's deploy runs at 02:00 daily.")

    def test_falls_back_to_source_when_model_empty(self):
        self.assertEqual(distill_fact("raw note here", "", lambda s, u, r: ""),
                         "raw note here")

    def test_falls_back_on_model_error(self):
        def boom(s, u, r):
            raise RuntimeError("backend down")
        self.assertEqual(distill_fact("keep this", "", boom), "keep this")


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.stored = []

    def _remember(self, text):
        self.stored.append(text)
        return "mem-id-1"

    def test_inline_request_stores_distilled_fact(self):
        fact = remember_from_request(
            "remember that the API key rotates monthly", [],
            generate_fn=lambda s, u, r: "The user's API key rotates monthly.",
            remember_fn=self._remember)
        self.assertEqual(fact, "The user's API key rotates monthly.")
        self.assertEqual(self.stored, ["The user's API key rotates monthly."])

    def test_referential_request_uses_previous_reply(self):
        history = [
            {"role": "user", "content": "how does blue-green deploy work?"},
            {"role": "assistant", "content": "Blue-green keeps two envs; you flip "
                                             "traffic after health checks pass."},
            {"role": "user", "content": "store this into memory"},
        ]
        captured = {}

        def gen(s, u, r):
            captured["user"] = u
            return "Blue-green deployment flips traffic between two envs after health "\
                   "checks pass."

        fact = remember_from_request("store this into memory", history,
                                     generate_fn=gen, remember_fn=self._remember)
        self.assertIn("Blue-green", fact)
        self.assertEqual(len(self.stored), 1)
        # The prior assistant reply was the source, with its prompting question as context.
        self.assertIn("two envs", captured["user"])
        self.assertIn("blue-green deploy work", captured["user"])

    def test_referential_with_no_history_stores_nothing(self):
        fact = remember_from_request("remember this", [],
                                     generate_fn=lambda s, u, r: "x",
                                     remember_fn=self._remember)
        self.assertIsNone(fact)
        self.assertEqual(self.stored, [])

    def test_store_failure_returns_none(self):
        fact = remember_from_request("remember that X is true", [],
                                     generate_fn=lambda s, u, r: "X is true.",
                                     remember_fn=lambda t: None)
        self.assertIsNone(fact)


if __name__ == "__main__":
    unittest.main()
