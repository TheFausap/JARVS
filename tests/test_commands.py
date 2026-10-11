"""Tests for the shared slash-command dispatcher used by both the REPL and the web UI.
dispatch_command must RETURN its output (so the web path can stream it back) rather than
print it, and it must not raise on quit/help/unknown."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jarvs.chat import dispatch_command  # noqa: E402


class _FakeSelf:
    def __init__(self):
        self.reloaded = False

    def load(self):
        self.reloaded = True

    def stats(self):
        return "self-model: 15 chunk(s) from 1 file(s) [architecture.md]"


class _FakeSession:
    """Only the attributes the exercised command branches touch."""
    def __init__(self, self_model=None):
        self.self_model = self_model
        self.reset_called = False

    def reset(self):
        self.reset_called = True


class CommandTests(unittest.TestCase):
    def test_returns_string_not_none(self):
        self.assertIsInstance(dispatch_command(_FakeSession(), "/help"), str)

    def test_help_lists_self(self):
        out = dispatch_command(_FakeSession(), "/help")
        self.assertIn("/self", out)
        self.assertIn("/claude", out)

    def test_self_reload_runs_and_reports(self):
        sm = _FakeSelf()
        out = dispatch_command(_FakeSession(sm), "/self reload")
        self.assertTrue(sm.reloaded)
        self.assertIn("reloaded dossier", out)
        self.assertIn("15 chunk(s)", out)

    def test_self_stats_without_reload(self):
        sm = _FakeSelf()
        out = dispatch_command(_FakeSession(sm), "/self")
        self.assertFalse(sm.reloaded)
        self.assertIn("15 chunk(s)", out)

    def test_self_disabled(self):
        out = dispatch_command(_FakeSession(None), "/self reload")
        self.assertIn("disabled", out)

    def test_reset(self):
        sess = _FakeSession()
        out = dispatch_command(sess, "/reset")
        self.assertTrue(sess.reset_called)
        self.assertIn("cleared", out)

    def test_unknown_command(self):
        out = dispatch_command(_FakeSession(), "/bogus now")
        self.assertIn("unknown command", out)
        self.assertIn("/bogus", out)


if __name__ == "__main__":
    unittest.main()
