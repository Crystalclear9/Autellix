import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

from integrations.sglang.reserve import release_reserves


class ReserveReleaseTests(unittest.TestCase):
    def scheduler(self):
        events = []
        scheduler = NS(_autellix_reserve={"r": "node"},
                       waiting_queue=[NS(rid="r", _autellix_reserved_tokens=[10, 11])],
                       _autellix_host=NS(save=lambda *args: events.append("save")),
                       autellix=NS(emit=Mock()),
                       tree_cache=NS(match_prefix=lambda _: NS(device_indices=[7, 8]),
                                     dec_lock_ref=lambda _: events.append("unlock")))
        return scheduler, events

    def test_pressure_preserves_reserve_before_unpinning(self):
        scheduler, events = self.scheduler()
        release_reserves(scheduler)
        self.assertEqual(events, ["save", "unlock"])
        self.assertEqual(scheduler._autellix_reserve, {})

    def test_failed_host_copy_keeps_pin_and_reserve(self):
        scheduler, events = self.scheduler()
        scheduler._autellix_host.save = Mock(side_effect=RuntimeError("host budget exhausted"))
        with self.assertRaises(RuntimeError):
            release_reserves(scheduler)
        self.assertEqual(events, [])
        self.assertEqual(scheduler._autellix_reserve, {"r": "node"})
