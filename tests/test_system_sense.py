"""system_sense tests — live /proc readings, trigger gating, and geolocation caching
(no network: the cache path is exercised directly)."""

import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jarvs import config, system_sense as ss  # noqa: E402


class ReadingsTests(unittest.TestCase):
    def test_cpu(self):
        c = ss.cpu(sample=0.02)
        self.assertGreaterEqual(c["cores"], 1)
        if c["util_pct"] is not None:
            self.assertTrue(0.0 <= c["util_pct"] <= 100.0)

    def test_memory(self):
        m = ss.memory()
        self.assertIsNotNone(m)
        self.assertGreater(m["total_mb"], 0)
        self.assertTrue(0.0 <= m["used_pct"] <= 100.0)

    def test_gpu_is_list(self):
        self.assertIsInstance(ss.gpu(), list)   # empty if no NVIDIA GPU

    def test_host_and_time(self):
        self.assertTrue(ss.host()["hostname"])
        self.assertIn("iso", ss.time_now())

    def test_summary_block(self):
        block = ss.summary_block(include_geo=False)
        self.assertIn("[SYSTEM STATUS]", block)
        self.assertIn("CPU:", block)


class TriggerTests(unittest.TestCase):
    def test_system_triggers(self):
        for t in ["what's the cpu load?", "how hot is the gpu",
                  "check memory usage", "what time is it?", "uptime?"]:
            self.assertTrue(ss.needs_system_info(t), t)

    def test_location_triggers(self):
        self.assertTrue(ss.needs_location("where am i?"))
        self.assertTrue(ss.needs_location("what city are we in"))
        self.assertTrue(ss.needs_system_info("where am i"))  # location ⊂ system

    def test_non_triggers(self):
        for t in ["tell me a joke", "summarize this text", "who wrote Dubliners"]:
            self.assertFalse(ss.needs_system_info(t), t)


class GeoCacheTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self._saved = (config.DATA_DIR, config.GEOLOCATION, config.ALLOW_WEB)
        config.DATA_DIR = self.dir
        config.GEOLOCATION = True
        config.ALLOW_WEB = True

    def tearDown(self):
        config.DATA_DIR, config.GEOLOCATION, config.ALLOW_WEB = self._saved
        import shutil
        shutil.rmtree(self.dir)

    def test_fresh_cache_used_without_network(self):
        geo = {"city": "Warsaw", "country": "Poland", "lat": 52.2, "lon": 21.0,
               "ts": time.time(), "cached": False}
        with open(os.path.join(self.dir, "geo_cache.json"), "w") as fh:
            json.dump(geo, fh)
        got = ss.geolocation()           # must read cache, not hit the network
        self.assertEqual(got["city"], "Warsaw")
        self.assertTrue(got["cached"])

    def test_disabled_returns_none(self):
        config.GEOLOCATION = False
        self.assertIsNone(ss.geolocation())

    def test_stale_cache_expired(self):
        geo = {"city": "Old", "ts": time.time() - (config.GEO_TTL + 10)}
        with open(os.path.join(self.dir, "geo_cache.json"), "w") as fh:
            json.dump(geo, fh)
        from jarvs.system_sense import _load_geo_cache
        self.assertIsNone(_load_geo_cache())   # too old


if __name__ == "__main__":
    unittest.main()
