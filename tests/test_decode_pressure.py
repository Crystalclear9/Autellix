import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

from integrations.sglang.pressure import make_decode_room


class DecodePressureTests(unittest.TestCase):
    def test_pressure_retracts_lowest_priority_not_longest_request(self):
        high = NS(rid="high", req_pool_idx=0, origin_input_ids=list(range(8)), output_ids=[8],
                  prefix_indices=[], last_node="high-cache", reset_for_retract=Mock())
        low = NS(rid="low", req_pool_idx=1, origin_input_ids=[1], output_ids=[2],
                 prefix_indices=[], last_node="low-cache", reset_for_retract=Mock())
        class Slots:
            def __getitem__(self, key):
                return list(range(9))[key[1]]
        batch = NS(reqs=[high, low], seq_lens=NS(cpu=lambda: NS(tolist=lambda: [9, 2])),
                   req_to_token_pool=NS(req_to_token=Slots(), free=Mock()),
                   token_to_kv_pool_allocator=NS(free=Mock()), tree_cache=NS(dec_lock_ref=Mock()))
        batch.check_decode_mem = lambda _: len(batch.reqs) == 1
        batch.filter_batch = lambda keep_indices: setattr(batch, "reqs", [batch.reqs[i] for i in keep_indices])
        window = NS(cohort=["high", "low"], drop_reserve=Mock())
        window.retain_prefix = lambda count: setattr(window, "cohort", window.cohort[:count])
        scheduler = NS(autellix=NS(emit=Mock()), _autellix_window=window,
                       _autellix_reserve={"reserve": "reserve-cache"},
                       tree_cache=NS(dec_lock_ref=Mock()), _autellix_host=NS(save=Mock()),
                       decode_mem_cache_buf_multiplier=1, _extend_requests_to_queue=Mock())
        make_decode_room(scheduler, batch)
        self.assertEqual(batch.reqs, [high])
        self.assertEqual(window.cohort, ["high"])
        self.assertEqual(scheduler._autellix_host.save.call_args.args[0], "low")
        scheduler._extend_requests_to_queue.assert_called_once_with([low], is_retracted=True)
        low.reset_for_retract.assert_called_once()
        high.reset_for_retract.assert_not_called()
        self.assertEqual(scheduler._autellix_reserve, {})

    def test_no_pressure_leaves_reserves_and_plan_untouched(self):
        scheduler = NS(_autellix_reserve={"reserve": "cache"})
        scheduler.decode_mem_cache_buf_multiplier = 1
        make_decode_room(scheduler, NS(check_decode_mem=lambda _: True))
        self.assertEqual(scheduler._autellix_reserve, {"reserve": "cache"})
