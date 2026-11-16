import unittest
from types import SimpleNamespace as NS
from integrations.sglang.admission import can_keep_running


class KeepRunningTests(unittest.TestCase):
    def test_preserve_only_when_complete_cohort_fits(self):
        req = lambda prompt, output, maximum: NS(rid="running", origin_input_ids=[1]*prompt, output_ids=[2]*output,
                                                sampling_params=NS(max_new_tokens=maximum))
        running, pending = req(50, 10, 100), req(80, 20, 100)
        scheduler = NS(running_batch=NS(reqs=[running]), server_args=NS(max_running_requests=2),
                       max_prefill_tokens=1024, token_to_kv_pool_allocator=NS(available_size=lambda: 200),
                       tree_cache=NS(evictable_size=lambda: 100))
        self.assertTrue(can_keep_running(scheduler, [pending], {"running"}))  # 100 + 100 + 90 + 1 < 300
        scheduler.tree_cache.evictable_size = lambda: 90
        self.assertFalse(can_keep_running(scheduler, [pending], {"running"}))
        scheduler.tree_cache.evictable_size = lambda: 100
        scheduler.max_prefill_tokens = 101
        self.assertFalse(can_keep_running(scheduler, [pending], {"running"}))
        scheduler.max_prefill_tokens = 1024
        scheduler.server_args.max_running_requests = 1
        self.assertFalse(can_keep_running(scheduler, [pending], {"running"}))
        scheduler.server_args.max_running_requests = 2
        scheduler.running_batch.reqs.append(NS(rid="victim"))
        self.assertTrue(can_keep_running(scheduler, [pending], {"running"}))
