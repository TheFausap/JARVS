"""Claude-bridge tests — code-request detection, verdict parsing, the rolling token
ledger / budget governance, and the three-gate negotiation (Gemma + Claude both
stubbed, so no Agent SDK is required)."""

import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jarvs import config  # noqa: E402
from jarvs.claude_bridge import (  # noqa: E402
    ClaudeBridge, is_code_request, parse_verdict, sum_usage,
)


class IntentTests(unittest.TestCase):
    def test_positive(self):
        for t in ["ask Claude to write a function to parse logs",
                  "have Claude refactor the server module",
                  "claude code: add a retry to the client",
                  "write a unit test for the parser",
                  "fix the bug in background.py"]:
            self.assertTrue(is_code_request(t), t)

    def test_negative(self):
        for t in ["what's the cpu load?",
                  "research LLM architectures in the background",
                  "what do you think about Claude?",
                  "tell me about routines in general",
                  "schedule a daily news digest"]:
            self.assertFalse(is_code_request(t), t)


class VerdictTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_verdict("VERDICT: proceed\nlooks fine"), "proceed")
        self.assertEqual(parse_verdict("verdict : reject — unsafe"), "reject")
        self.assertEqual(parse_verdict("VERDICT: needs_clarification\nwhich file?"),
                         "needs_clarification")
        self.assertEqual(parse_verdict("Verdict needs clarification please"),
                         "needs_clarification")
        self.assertEqual(parse_verdict("no verdict here"), "")


class _FakeResult:
    """Mimics an Agent SDK ResultMessage for sum_usage()."""
    def __init__(self, model_usage=None, usage=None, num_turns=0, session_id=""):
        self.model_usage = model_usage
        self.usage = usage
        self.num_turns = num_turns
        self.session_id = session_id


class UsageTests(unittest.TestCase):
    def test_model_usage_preferred(self):
        msg = _FakeResult(
            model_usage={"claude": {"inputTokens": 100, "outputTokens": 40,
                                    "cacheReadInputTokens": 10,
                                    "cacheCreationInputTokens": 5}},
            num_turns=2, session_id="abc")
        u = sum_usage(msg)
        self.assertEqual((u["in"], u["out"], u["cache"]), (100, 40, 15))
        self.assertEqual(u["turns"], 2)
        self.assertEqual(u["session"], "abc")

    def test_usage_fallback(self):
        msg = _FakeResult(usage={"input_tokens": 7, "output_tokens": 3,
                                 "cache_read_input_tokens": 1,
                                 "cache_creation_input_tokens": 0})
        u = sum_usage(msg)
        self.assertEqual((u["in"], u["out"], u["cache"]), (7, 3, 1))


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.ledger = os.path.join(self.dir, "claude_ledger.json")
        # Pin a known budget regardless of the environment.
        self._cap, self._warn = config.CLAUDE_TOKEN_DAILY_CAP, config.CLAUDE_TOKEN_WARN_AT
        config.CLAUDE_TOKEN_DAILY_CAP = 1000
        config.CLAUDE_TOKEN_WARN_AT = 0.8
        self.claude_calls = []    # captured (prompt, allow_write, resume)
        self.learned = []         # captured memory_fn lessons

    def tearDown(self):
        import shutil
        config.CLAUDE_TOKEN_DAILY_CAP, config.CLAUDE_TOKEN_WARN_AT = self._cap, self._warn
        shutil.rmtree(self.dir, ignore_errors=True)

    def _bridge(self, *, generate, claude):
        def _claude(prompt, *, allow_write, resume):
            self.claude_calls.append((prompt, allow_write, resume))
            return claude(prompt, allow_write=allow_write, resume=resume)
        return ClaudeBridge(generate_fn=generate, ledger_path=self.ledger,
                            cwd=self.dir, claude_fn=_claude,
                            memory_fn=self.learned.append)

    def test_local_rejection_spends_no_claude(self):
        b = self._bridge(
            generate=lambda s, u, r: "REVISE: too vague",
            claude=lambda p, **k: ("should not be called", {}))
        res = b.request_code("do stuff")
        self.assertEqual(res["status"], "rejected_local")
        self.assertEqual(self.claude_calls, [])

    def test_happy_path_two_gates_and_ledger(self):
        gen = iter(["PROCEED", "AUTHORIZE"])

        def claude(prompt, *, allow_write, resume):
            if not allow_write:
                return ("VERDICT: proceed\nlooks sound", {
                    "in": 100, "out": 50, "turns": 1, "session": "sess1"})
            return ("Edited parser.py, added a guard.", {
                "in": 200, "out": 80, "turns": 2, "session": "sess1"})

        b = self._bridge(generate=lambda s, u, r: next(gen), claude=claude)
        res = b.request_code("fix the bug in parser.py")
        self.assertEqual(res["status"], "done")
        self.assertEqual(len(res["turns"]), 2)
        # Both Claude calls recorded in the ledger (430 tokens total).
        self.assertEqual(b._tokens_since(86400), 100 + 50 + 200 + 80)
        # Gate 3 resumed the gate-2 session, with write access.
        self.assertEqual(self.claude_calls[0][1:], (False, None))
        self.assertEqual(self.claude_calls[1][1:], (True, "sess1"))

    def test_needs_clarification_stops_before_write(self):
        b = self._bridge(
            generate=lambda s, u, r: "PROCEED",
            claude=lambda p, **k: ("VERDICT: needs_clarification\nwhich module?",
                                   {"in": 10, "out": 5}))
        res = b.request_code("refactor the code")
        self.assertEqual(res["status"], "needs_clarification")
        self.assertEqual(len(self.claude_calls), 1)   # no execution call

    def test_hold_stops_before_write(self):
        gen = iter(["PROCEED", "HOLD: not now"])
        b = self._bridge(
            generate=lambda s, u, r: next(gen),
            claude=lambda p, **k: ("VERDICT: proceed\nready", {"in": 10, "out": 5}))
        res = b.request_code("write a test for the client")
        self.assertEqual(res["status"], "held")
        self.assertEqual(len(self.claude_calls), 1)

    def test_over_budget_refuses(self):
        b = self._bridge(
            generate=lambda s, u, r: "PROCEED",
            claude=lambda p, **k: ("should not run", {}))
        # Pre-seed the ledger over the 1000-token cap.
        b._record({"in": 800, "out": 400})
        res = b.request_code("fix the bug in parser.py")
        self.assertEqual(res["status"], "over_budget")
        self.assertEqual(self.claude_calls, [])
        self.assertIn("[BUDGET]", res["detail"])

    def test_warning_injected_into_prompt(self):
        gen = iter(["PROCEED", "HOLD: stop here"])
        b = self._bridge(
            generate=lambda s, u, r: next(gen),
            claude=lambda p, **k: ("VERDICT: proceed", {"in": 1, "out": 1}))
        b._record({"in": 850, "out": 0})          # 85% of the 1000-token cap
        b.request_code("refactor the parser module")
        self.assertIn("[BUDGET]", self.claude_calls[0][0])

    def test_learns_from_local_rejection(self):
        b = self._bridge(generate=lambda s, u, r: "REVISE: too vague",
                         claude=lambda p, **k: ("", {}))
        b.request_code("do something")
        self.assertEqual(len(self.learned), 1)
        self.assertIn("held its own coding request back", self.learned[0])

    def test_learns_from_claude_observation_on_done(self):
        gen = iter(["PROCEED", "AUTHORIZE"])

        def claude(prompt, *, allow_write, resume):
            if not allow_write:
                return ("VERDICT: proceed", {"in": 1, "out": 1, "session": "s"})
            return ("Added a null guard in parser.py line 42.",
                    {"in": 2, "out": 2, "session": "s"})

        b = self._bridge(generate=lambda s, u, r: next(gen), claude=claude)
        b.request_code("fix the bug in parser.py")
        self.assertEqual(len(self.learned), 1)
        self.assertIn("Claude completed", self.learned[0])
        self.assertIn("null guard", self.learned[0])

    def test_exchanges_persist_and_payload(self):
        b = self._bridge(generate=lambda s, u, r: "REVISE: no",
                         claude=lambda p, **k: ("", {}))
        b.request_code("write a function somewhere")
        pay = b.payload()
        self.assertTrue(pay["enabled"])
        self.assertEqual(len(pay["exchanges"]), 1)
        self.assertEqual(pay["exchanges"][0]["status"], "rejected_local")
        self.assertIn("day", pay["budget"])
        # A fresh bridge over the same files recovers the exchange history.
        b2 = ClaudeBridge(generate_fn=lambda s, u, r: "x", ledger_path=self.ledger,
                          cwd=self.dir, claude_fn=lambda p, **k: ("", {}))
        self.assertEqual(len(b2.recent()), 1)

    def test_ledger_persists_across_reload(self):
        b = self._bridge(generate=lambda s, u, r: "REVISE: nope",
                         claude=lambda p, **k: ("", {}))
        b._record({"in": 120, "out": 30})
        b2 = ClaudeBridge(generate_fn=lambda s, u, r: "x",
                          ledger_path=self.ledger, cwd=self.dir,
                          claude_fn=lambda p, **k: ("", {}))
        self.assertEqual(b2._tokens_since(86400), 150)


if __name__ == "__main__":
    unittest.main()
