"""Spine tests: the reply loop, history budgeting, and reasoning-strip — all with a
mocked backend so no live model is needed."""

import os
import sys
import time
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


class FoundryDispatchTests(unittest.TestCase):
    """A build request must reach the foundry — and ordinary chat must not."""

    def setUp(self):
        import shutil
        import tempfile
        from pathlib import Path

        from jarvs import config
        from jarvs.coder import CoderBackend

        self.dir = tempfile.mkdtemp()
        self._cleanup = lambda: shutil.rmtree(self.dir, ignore_errors=True)
        self._data, self._complete = config.DATA_DIR, CoderBackend.complete
        config.DATA_DIR = Path(self.dir)
        # Any build the worker picks up fails immediately and silently.
        CoderBackend.complete = lambda self, system, user, max_tokens=None: ""
        self.session = JarvsSession(speak=False, voice=_SilentVoice())

    def tearDown(self):
        from jarvs import config
        from jarvs.coder import CoderBackend
        config.DATA_DIR = self._data
        CoderBackend.complete = self._complete
        self._cleanup()

    def test_build_request_queues_a_project(self):
        out = self.session._dispatch_background(
            "build me a CLI habit tracker", "On it, sir.")
        self.assertEqual(out, "On it, sir.")
        projects = self.session.coder.list()
        self.assertEqual(len(projects), 1)
        self.assertEqual(projects[0]["topic"], "build me a CLI habit tracker")

    def test_ordinary_chat_does_not_build(self):
        self.session._dispatch_background("what is the cpu load?", "Forty percent.")
        self.assertEqual(self.session.coder.list(), [])

    def test_code_marker_in_reply_queues(self):
        out = self.session._dispatch_background("hello", "on it [[CODE: a snake game]]")
        self.assertEqual(out, "on it")
        self.assertEqual(self.session.coder.list()[0]["topic"], "a snake game")


class CodingRouteTests(unittest.TestCase):
    """Two coding paths, one router: Claude Code (opt-in) vs the local foundry."""

    def setUp(self):
        import shutil
        import tempfile
        from pathlib import Path

        from jarvs import config
        from jarvs.coder import CoderBackend

        self.dir = tempfile.mkdtemp()
        self._cleanup = lambda: shutil.rmtree(self.dir, ignore_errors=True)
        self._data, self._complete = config.DATA_DIR, CoderBackend.complete
        self._bridge = config.CLAUDE_BRIDGE_ENABLED
        config.DATA_DIR = Path(self.dir)
        CoderBackend.complete = lambda self, system, user, max_tokens=None: ""
        self.session = JarvsSession(speak=False, voice=_SilentVoice())

    def tearDown(self):
        from jarvs import config
        from jarvs.coder import CoderBackend
        config.DATA_DIR = self._data
        config.CLAUDE_BRIDGE_ENABLED = self._bridge
        CoderBackend.complete = self._complete
        self._cleanup()

    def _enable_bridge(self):
        """Rebuild the session with the bridge on and its executor stubbed."""
        from jarvs import config
        config.CLAUDE_BRIDGE_ENABLED = True
        self.session = JarvsSession(speak=False, voice=_SilentVoice())
        self.bridge_calls: list[str] = []
        # Runs on a daemon thread, so record the call and join briefly below.
        self.session.claude_bridge.request_code = (
            lambda task, context="": (self.bridge_calls.append(task),
                                      {"status": "done"})[1])

    def test_build_request_goes_to_the_foundry(self):
        self.session._dispatch_background("build me a CLI habit tracker", "On it.")
        self.assertEqual(len(self.session.coder.list()), 1)

    def test_claude_named_but_bridge_off_is_not_silently_rerouted(self):
        # The user named a specific tool that isn't enabled: the foundry must not
        # quietly answer it with a different model.
        self.session._dispatch_background("use claude code to fix parser.py", "Ok.")
        self.assertEqual(self.session.coder.list(), [])

    def test_bridge_wins_when_enabled(self):
        self._enable_bridge()
        self.session._route_coding("use claude code to fix parser.py")
        self.assertEqual(self.session.coder.list(), [])
        time.sleep(0.2)
        self.assertEqual(len(self.bridge_calls), 1)

    def test_plain_build_still_uses_the_foundry_with_bridge_on(self):
        self._enable_bridge()
        # No mention of Claude, and 'habit tracker' is not a Claude-bridge code noun.
        self.session._route_coding("build me a CLI habit tracker")
        self.assertEqual(len(self.session.coder.list()), 1)
        time.sleep(0.2)
        self.assertEqual(self.bridge_calls, [])


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
