"""Background-jobs tests — trigger detection, marker extraction, and the full
queue→worker→complete lifecycle with a stubbed generate_fn (no backend, no web)."""

import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jarvs import config, background  # noqa: E402
from jarvs.background import (  # noqa: E402
    BackgroundManager, should_auto_queue, extract_think_markers, background_signals,
)


class TriggerTests(unittest.TestCase):
    def test_auto_queue_requires_clear_signal(self):
        self.assertTrue(should_auto_queue("Research the X protocol in the background."))
        self.assertTrue(should_auto_queue(
            "Look into this and let me know when you're done."))
        self.assertFalse(should_auto_queue("What time is it?"))
        self.assertFalse(should_auto_queue("Tell me about the background of jazz."))

    def test_marker_extraction_strips_and_collects(self):
        clean, tasks = extract_think_markers(
            "Sure, I'll handle it. [[THINK: compare A vs B]] Anything else?")
        self.assertEqual(tasks, ["compare A vs B"])
        self.assertNotIn("[[THINK", clean)
        self.assertIn("Sure, I'll handle it.", clean)

    def test_no_markers_returns_reply_unchanged(self):
        clean, tasks = extract_think_markers("Just a normal reply.")
        self.assertEqual(tasks, [])
        self.assertEqual(clean, "Just a normal reply.")


class _Env:
    """Temporarily force background config into a fast, web-free, offline state."""
    def __enter__(self):
        self._saved = {k: getattr(config, k) for k in
                       ("ALLOW_WEB", "BACKGROUND_WEB", "BACKGROUND_RETRIES",
                        "BACKGROUND_RETRY_BACKOFF", "BACKGROUND_MAX_PENDING")}
        config.ALLOW_WEB = False          # no network in tests
        config.BACKGROUND_WEB = False
        config.BACKGROUND_RETRIES = 1
        config.BACKGROUND_RETRY_BACKOFF = 0
        config.BACKGROUND_MAX_PENDING = 3
        return self

    def __exit__(self, *a):
        for k, v in self._saved.items():
            setattr(config, k, v)


def _wait(mgr, job_id, status, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        j = next((x for x in mgr.jobs if x["id"] == job_id), None)
        if j and j["status"] == status:
            return j
        time.sleep(0.02)
    return next((x for x in mgr.jobs if x["id"] == job_id), None)


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.store = os.path.join(self.dir, "jobs.json")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.dir)

    def _mgr(self, generate):
        return BackgroundManager(generate_fn=generate, store_path=self.store)

    def test_job_runs_to_completion(self):
        with _Env():
            mgr = self._mgr(lambda s, u, r: "The analysis result.")
            job = mgr.queue("Analyze the tradeoffs of approach A.")
            self.assertIsNotNone(job)
            done = _wait(mgr, job["id"], "completed")
            self.assertEqual(done["status"], "completed")
            self.assertEqual(done["result"], "The analysis result.")

    def test_dedup_and_cap(self):
        with _Env():
            # a generate that blocks so jobs stay pending while we test the cap
            import threading
            gate = threading.Event()
            mgr = self._mgr(lambda s, u, r: (gate.wait(2), "done")[1])
            a = mgr.queue("Task number one about something.")
            dup = mgr.queue("Task number one about something.")  # identical -> dedup
            self.assertEqual(dup["id"], a["id"])
            mgr.queue("Task number two about something.")
            mgr.queue("Task number three about something.")
            over = mgr.queue("Task number four exceeds the cap.")  # cap=3
            self.assertIsNone(over)
            gate.set()

    def test_nonactionable_rejected(self):
        with _Env():
            mgr = self._mgr(lambda s, u, r: "x")
            self.assertIsNone(mgr.queue("ok"))
            self.assertIsNone(mgr.queue("thanks"))
            self.assertIsNone(mgr.queue("short"))  # < 8 chars

    def test_transient_failure_retries_then_succeeds(self):
        with _Env():
            calls = {"n": 0}
            def gen(s, u, r):
                calls["n"] += 1
                return ("Error connecting to the backend: read timed out"
                        if calls["n"] == 1 else "Recovered result.")
            mgr = self._mgr(gen)
            job = mgr.queue("Investigate the flaky thing in detail.")
            done = _wait(mgr, job["id"], "completed", timeout=8)
            self.assertEqual(done["status"], "completed")
            self.assertEqual(done["result"], "Recovered result.")
            self.assertEqual(done["attempts"], 1)

    def test_events_and_persistence(self):
        with _Env():
            mgr = self._mgr(lambda s, u, r: "result text")
            job = mgr.queue("Do a thing worth analyzing here.")
            _wait(mgr, job["id"], "completed")
            kinds = {e["kind"] for e in mgr.events}
            self.assertIn("queued", kinds)
            self.assertIn("completed", kinds)
            # a fresh manager loads the persisted completed job
            mgr2 = BackgroundManager(generate_fn=lambda s, u, r: "x",
                                     store_path=self.store)
            self.assertTrue(any(j["id"] == job["id"] and j["status"] == "completed"
                                for j in mgr2.jobs))


if __name__ == "__main__":
    unittest.main()
