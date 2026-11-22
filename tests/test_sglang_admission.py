import unittest
from importlib.util import find_spec
from types import SimpleNamespace as NS
from integrations.sglang.admission import can_keep_running


class KeepRunningTests(unittest.TestCase):
    def test_preserve_when_current_decode_window_fits(self):
        req = lambda prompt, output, maximum: NS(rid="running", origin_input_ids=[1]*prompt, output_ids=[2]*output,
                                                sampling_params=NS(max_new_tokens=maximum),
                                                extend_input_len=prompt + output, init_next_round_input=lambda _: None,
                                                last_node=None)
        running, pending = req(50, 10, 100), req(80, 20, 100)
        scheduler = NS(running_batch=NS(reqs=[running]), server_args=NS(max_running_requests=2),
                       pp_size=1, is_mixed_chunk=True,
                       max_prefill_tokens=1024, token_to_kv_pool_allocator=NS(available_size=lambda: 100),
                       tree_cache=NS(evictable_size=lambda: 20, inc_lock_ref=lambda _: None, dec_lock_ref=lambda _: None),
                       autellix=NS(config=NS(implementation="paper", schedule_interval=8)))
        self.assertTrue(can_keep_running(scheduler, [pending], {"running"}))  # 100 + 8 + 8 + 1 < 120
        self.assertEqual(pending.sampling_params.max_new_tokens, 100)
        scheduler.tree_cache.evictable_size = lambda: 17
        self.assertFalse(can_keep_running(scheduler, [pending], {"running"}))
        scheduler.tree_cache.evictable_size = lambda: 20
        scheduler.max_prefill_tokens = 101
        self.assertTrue(can_keep_running(scheduler, [pending], {"running"}))
        scheduler.chunked_prefill_size = 1
        self.assertFalse(can_keep_running(scheduler, [pending], {"running"}))
        scheduler.chunked_prefill_size = 2
        self.assertTrue(can_keep_running(scheduler, [pending], {"running"}))
        scheduler.max_prefill_tokens = 1024
        scheduler.server_args.max_running_requests = 1
        self.assertFalse(can_keep_running(scheduler, [pending], {"running"}))
        scheduler.server_args.max_running_requests = 2
        scheduler.running_batch.reqs.append(NS(rid="victim"))
        self.assertTrue(can_keep_running(scheduler, [pending], {"running"}))

    def test_two_prefills_charge_one_shared_radix_prefix(self):
        locks = []
        tree = NS(evictable_size=lambda: 0 if locks else 100,
                  inc_lock_ref=locks.append, dec_lock_ref=locks.remove)
        requests = [NS(rid=str(i), last_node="shared", origin_input_ids=[0] * 112, output_ids=[],
                       extend_input_len=12, sampling_params=NS(max_new_tokens=50),
                       init_next_round_input=lambda _: None) for i in range(2)]
        scheduler = NS(running_batch=NS(reqs=[]), server_args=NS(max_running_requests=2),
                       pp_size=1, is_mixed_chunk=True,
                       max_prefill_tokens=128, token_to_kv_pool_allocator=NS(available_size=lambda: 50),
                       tree_cache=tree, autellix=NS(config=NS(implementation="paper", schedule_interval=8)))
        self.assertTrue(can_keep_running(scheduler, requests, {r.rid for r in requests}))
        self.assertEqual(locks, [])


@unittest.skipUnless(find_spec("sglang"), "requires the pinned SGLang package")
class NativeAdmissionBudgetTests(unittest.TestCase):
    def test_full_forward_budget_is_distinct_from_full_kv_pool(self):
        from sglang.srt.managers import schedule_policy as native
        tree = NS(disable=False, evictable_size=lambda: 0,
                  inc_lock_ref=lambda _: None, dec_lock_ref=lambda _: None)
        sampling = NS(ignore_eos=False, max_new_tokens=8)
        running = NS(reqs=[NS(sampling_params=sampling, output_ids=[])])
        for available, input_budget, chunk_budget, expected, admitted in (
            (2048, 16, 4, native.AddReqResult.OTHER, 1),
            (2048, 4, None, native.AddReqResult.OTHER, 1),
            (12, 16, 4, native.AddReqResult.NO_TOKEN, 0),
        ):
            req = NS(extend_input_len=3, host_hit_length=0, prefix_indices=[],
                     last_node="node", sampling_params=sampling)
            allocator = NS(available_size=lambda: available)
            adder = native.PrefillAdder(1, tree, allocator, running, 1., input_budget, chunk_budget, 1)
            self.assertEqual(adder.add_one_req(req, False), expected)
            self.assertEqual(len(adder.can_run_list), admitted)
            self.assertIsNone(adder.new_chunked_req)

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
