import unittest
from importlib.util import find_spec
from types import SimpleNamespace as NS
from integrations.sglang.admission import can_keep_running


class KeepRunningTests(unittest.TestCase):
    def test_preserve_when_current_decode_window_fits(self):
        req = lambda prompt, output, maximum: NS(rid="running", origin_input_ids=[1]*prompt, output_ids=[2]*output,
                                                sampling_params=NS(max_new_tokens=maximum),
                                                extend_input_len=prompt + output, init_next_round_input=lambda _: None)
        running, pending = req(50, 10, 100), req(80, 20, 100)
        scheduler = NS(running_batch=NS(reqs=[running]), server_args=NS(max_running_requests=2),
                       max_prefill_tokens=1024, token_to_kv_pool_allocator=NS(available_size=lambda: 100),
                       tree_cache=NS(evictable_size=lambda: 20),
                       autellix=NS(config=NS(implementation="paper", schedule_interval=8)))
        self.assertTrue(can_keep_running(scheduler, [pending], {"running"}))  # 100 + 8 + 8 + 1 < 120
        self.assertEqual(pending.sampling_params.max_new_tokens, 100)
        scheduler.tree_cache.evictable_size = lambda: 17
        self.assertFalse(can_keep_running(scheduler, [pending], {"running"}))
        scheduler.tree_cache.evictable_size = lambda: 20
        scheduler.max_prefill_tokens = 101
        self.assertFalse(can_keep_running(scheduler, [pending], {"running"}))
        scheduler.max_prefill_tokens = 1024
        scheduler.server_args.max_running_requests = 1
        self.assertFalse(can_keep_running(scheduler, [pending], {"running"}))
        scheduler.server_args.max_running_requests = 2
        scheduler.running_batch.reqs.append(NS(rid="victim"))
        self.assertTrue(can_keep_running(scheduler, [pending], {"running"}))


@unittest.skipUnless(find_spec("sglang"), "requires the pinned SGLang package")
class NativeAdmissionBudgetTests(unittest.TestCase):
    def test_window_budget_admits_without_shortening_generation(self):
        from sglang.srt.managers import schedule_policy as native
        original = native.CLIP_MAX_NEW_TOKENS_ESTIMATION
        try:
            for horizon, expected_count, ignore_eos in [(4096, 0, False), (8, 1, False),
                                                       (4096, 0, True), (8, 1, True)]:
                native.CLIP_MAX_NEW_TOKENS_ESTIMATION = horizon
                req = NS(extend_input_len=100, host_hit_length=0, prefix_indices=[], last_node="node",
                         sampling_params=NS(ignore_eos=ignore_eos, max_new_tokens=1000))
                tree = NS(disable=False, evictable_size=lambda: 0,
                          inc_lock_ref=lambda _: None, dec_lock_ref=lambda _: None)
                allocator = NS(available_size=lambda: 120)
                adder = native.PrefillAdder(1, tree, allocator, None, 1., 1024, None)
                adder.add_one_req(req, False)
                self.assertEqual(len(adder.can_run_list), expected_count)
                self.assertEqual(req.sampling_params.max_new_tokens, 1000)
        finally:
            native.CLIP_MAX_NEW_TOKENS_ESTIMATION = original
