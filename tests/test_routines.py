"""Routine-scheduler tests — interval parsing, schedule-intent detection, plan
extraction, and the due→fire→reschedule cycle (stubbed background + generate)."""

import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jarvs.routines import (  # noqa: E402
    RoutineScheduler, parse_interval, is_schedule_request, human_interval,
)


class ParseTests(unittest.TestCase):
    def test_intervals(self):
        self.assertEqual(parse_interval("daily"), 86400)
        self.assertEqual(parse_interval("every day"), 86400)
        self.assertEqual(parse_interval("hourly"), 3600)
        self.assertEqual(parse_interval("weekly"), 604800)
        self.assertEqual(parse_interval("every 6 hours"), 21600)
        self.assertEqual(parse_interval("every 2 days"), 172800)
        self.assertEqual(parse_interval("twice a day"), 43200)
        self.assertEqual(parse_interval("no period here", default=99), 99)

    def test_human_interval(self):
        self.assertEqual(human_interval(86400), "daily")
        self.assertEqual(human_interval(3600), "hourly")
        self.assertEqual(human_interval(21600), "every 6 hours")

    def test_schedule_intent(self):
        self.assertTrue(is_schedule_request(
            "add repeating daily searches about LLM architectures"))
        self.assertTrue(is_schedule_request("schedule this every 6 hours"))
        self.assertTrue(is_schedule_request("check the news weekly"))
        self.assertFalse(is_schedule_request("what's the cpu load?"))
        self.assertFalse(is_schedule_request("research this once in the background"))


class _FakeBG:
    def __init__(self):
        self.queued = []

    def queue(self, task, context=""):
        job = {"id": f"j{len(self.queued)}", "task": task}
        self.queued.append(job)
        return job


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.store = os.path.join(self.dir, "routines.json")
        self.bg = _FakeBG()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.dir)

    def _sched(self, generate=lambda s, u, r: "[]"):
        return RoutineScheduler(background=self.bg, store_path=self.store,
                                generate_fn=generate)

    def test_add_and_due_and_fire(self):
        s = self._sched()
        r = s.add("LLM news", "Research latest LLM architecture news.", 3600,
                  first_delay=0)       # due immediately
        self.assertIn(r, [dict(x) for x in [r]])  # sanity
        due = s.due()
        self.assertEqual(len(due), 1)
        self.assertTrue(s._fire(due[0]))
        self.assertEqual(len(self.bg.queued), 1)
        # after firing, next_run moved into the future and runs incremented
        after = s.list()[0]
        self.assertEqual(after["runs"], 1)
        self.assertGreater(after["next_run"], time.time())
        self.assertEqual(s.due(), [])

    def test_disabled_not_due(self):
        s = self._sched()
        r = s.add("x", "Do a recurring thing please.", 60, first_delay=0)
        s.set_enabled(r["id"], False)
        self.assertEqual(s.due(), [])

    def test_plan_and_create_from_request(self):
        plan = ('[{"description":"LLM architectures","task":"Research new LLM '
                'architecture breakthroughs and summarize.","interval":"daily"},'
                '{"description":"Russia-Ukraine","task":"Summarize the latest news on '
                'the Russia-Ukraine conflict.","interval":"daily"}]')
        s = self._sched(generate=lambda sy, u, r: plan)
        created = s.plan_and_create(
            "add repeating daily searches about LLM architectures and the war")
        self.assertEqual(len(created), 2)
        self.assertEqual({c["interval_sec"] for c in created}, {86400})
        self.assertEqual(len(s.list()), 2)

    def test_plan_ignores_non_schedule(self):
        s = self._sched(generate=lambda sy, u, r: "[]")
        self.assertEqual(s.plan_and_create("just chatting"), [])

    def test_persistence(self):
        s = self._sched()
        s.add("keep", "Do this recurring task regularly.", 3600)
        s2 = RoutineScheduler(background=self.bg, store_path=self.store,
                              generate_fn=lambda a, b, c: "[]")
        self.assertEqual(len(s2.list()), 1)
        self.assertEqual(s2.list()[0]["description"], "keep")


if __name__ == "__main__":
    unittest.main()
