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
    parse_time_of_day, is_routine_command, schedule_label,
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


class TimeParseTests(unittest.TestCase):
    def test_times(self):
        self.assertEqual(parse_time_of_day("7am"), (7, 0))
        self.assertEqual(parse_time_of_day("8 pm"), (20, 0))
        self.assertEqual(parse_time_of_day("08:30"), (8, 30))
        self.assertEqual(parse_time_of_day("at 19:00"), (19, 0))
        self.assertEqual(parse_time_of_day("every morning"), (8, 0))
        self.assertEqual(parse_time_of_day("12am"), (0, 0))
        self.assertEqual(parse_time_of_day("12pm"), (12, 0))

    def test_interval_not_mistaken_for_time(self):
        self.assertIsNone(parse_time_of_day("every 6 hours"))
        self.assertIsNone(parse_time_of_day("daily"))


class ManageIntentTests(unittest.TestCase):
    def test_positive(self):
        for t in ["move the LLM digest to 7am", "pause the war news routine",
                  "cancel the daily search", "change the schedule to every 6 hours",
                  "reschedule it to the morning"]:
            self.assertTrue(is_routine_command(t), t)

    def test_negative(self):
        for t in ["what's the weather?", "research this in the background",
                  "tell me about routines in general"]:
            self.assertFalse(is_routine_command(t), t)


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
        shutil.rmtree(self.dir, ignore_errors=True)

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

    def test_reschedule_interval_and_time(self):
        s = self._sched()
        r = s.add("x", "A recurring daily task to run.", 86400)
        self.assertTrue(s.reschedule(r["id"], interval_sec=21600, at=(7, 0)))
        got = s.list()[0]
        self.assertEqual(got["interval_sec"], 21600)
        self.assertEqual(got["at_hour"], 7)
        self.assertIn("at 07:00", schedule_label(got))

    def test_handle_request_reschedule_not_duplicate(self):
        s = self._sched()
        r = s.add("LLM news", "Research LLM architecture news daily.", 86400)
        # the router returns a reschedule action referencing the real id
        action = f'[{{"action":"reschedule","id":"{r["id"]}","interval":null,"time":"07:00"}}]'
        s._generate = lambda sy, u, rr: action
        applied = s.handle_request("move the LLM digest to 7am")
        self.assertEqual(applied, [("rescheduled", r["id"])])
        self.assertEqual(len(s.list()), 1)            # changed, not duplicated
        self.assertEqual(s.list()[0]["at_hour"], 7)

    def test_handle_request_pause_and_remove(self):
        s = self._sched()
        r = s.add("war news", "Summarize Russia-Ukraine news daily.", 86400)
        s._generate = lambda sy, u, rr: f'[{{"action":"pause","id":"{r["id"]}"}}]'
        self.assertEqual(s.handle_request("pause the war news routine"),
                         [("paused", r["id"])])
        self.assertFalse(s.list()[0]["enabled"])
        s._generate = lambda sy, u, rr: f'[{{"action":"remove","id":"{r["id"]}"}}]'
        self.assertEqual(s.handle_request("cancel it"), [("removed", r["id"])])
        self.assertEqual(s.list(), [])

    def test_handle_request_create(self):
        s = self._sched(generate=lambda sy, u, rr:
                        '[{"action":"create","description":"Crypto","task":"Research '
                        'crypto market news.","interval":"hourly","time":null}]')
        applied = s.handle_request("every hour, check crypto news")
        self.assertEqual(applied[0][0], "created")
        self.assertEqual(s.list()[0]["interval_sec"], 3600)

    def test_handle_request_ignores_unknown_id(self):
        s = self._sched(generate=lambda sy, u, rr:
                        '[{"action":"reschedule","id":"deadbeef","time":"07:00"}]')
        self.assertEqual(s.handle_request("change the thing"), [])

    def test_persistence(self):
        s = self._sched()
        s.add("keep", "Do this recurring task regularly.", 3600)
        s2 = RoutineScheduler(background=self.bg, store_path=self.store,
                              generate_fn=lambda a, b, c: "[]")
        self.assertEqual(len(s2.list()), 1)
        self.assertEqual(s2.list()[0]["description"], "keep")


if __name__ == "__main__":
    unittest.main()
