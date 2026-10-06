"""Spine tests: the reply loop, history budgeting, and reasoning-strip — all with a
mocked backend so no live model is needed."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jarvs import client  # noqa: E402
from jarvs.chat import JarvsSession  # noqa: E402


class StripReasoningTests(unittest.TestCase):
    def test_strips_think_block(self):
        self.assertEqual(client.strip_reasoning("<think>plan</think>Hello."), "Hello.")

    def test_drops_all_reasoning_completion(self):
        self.assertEqual(client.strip_reasoning("<think>only thinking, no answer"), "")

    def test_dangling_close_keeps_tail(self):
        self.assertEqual(client.strip_reasoning("leaked reasoning</think>Answer."), "Answer.")

    def test_plain_text_untouched(self):
        self.assertEqual(client.strip_reasoning("Just a reply."), "Just a reply.")


class ReplyLoopTests(unittest.TestCase):
    def setUp(self):
        # Mute voice and stub the backend so nothing hits the network.
        self.session = JarvsSession(speak=False, voice=_SilentVoice())
        self._orig = client.generate_with_history
        client.generate_with_history = lambda system, messages, **kw: "Acknowledged."
        client.LAST_GENERATION_ERROR = ""

    def tearDown(self):
        client.generate_with_history = self._orig

    def test_records_user_and_assistant(self):
        self.session.reply("Hello JARVS")
        self.assertEqual(self.session.history,
                         [{"role": "user", "content": "Hello JARVS"},
                          {"role": "assistant", "content": "Acknowledged."}])

    def test_empty_input_noop(self):
        self.assertEqual(self.session.reply("   "), "")
        self.assertEqual(self.session.history, [])

    def test_backend_error_not_recorded_as_turn(self):
        client.generate_with_history = lambda system, messages, **kw: ""
        client.LAST_GENERATION_ERROR = "backend timeout"
        out = self.session.reply("ping")
        self.assertTrue(out.startswith("[backend error]"))
        # The user line is recorded; the failed reply is NOT appended as assistant.
        self.assertEqual([m["role"] for m in self.session.history], ["user"])

    def test_history_fit_drops_oldest(self):
        # Force a tiny budget and confirm the oldest turns are dropped, newest kept.
        from jarvs import config
        orig = (config.SLOT_CTX, config.GEN_RESERVE, config.SAFETY_MARGIN)
        config.SLOT_CTX, config.GEN_RESERVE, config.SAFETY_MARGIN = 2048, 2000, 0
        try:
            self.session.history = [
                {"role": "user", "content": "oldest " * 50},
                {"role": "assistant", "content": "mid " * 50},
                {"role": "user", "content": "newest"},
            ]
            kept = self.session._fit_history()
            self.assertIn({"role": "user", "content": "newest"}, kept)
            self.assertLess(len(kept), 3)
        finally:
            config.SLOT_CTX, config.GEN_RESERVE, config.SAFETY_MARGIN = orig


class _SilentVoice:
    """A no-op stand-in for JarvsVoice so tests never spawn piper/say."""
    enabled = False

    def speak(self, text): pass
    def speak_synced(self, text, reveal): pass
    def set_enabled(self, on): pass
    def stop(self): pass
    def status(self): return "voice: unavailable"


if __name__ == "__main__":
    unittest.main()
