"""Verify the boundary and accounting contract with SGLang's native overlap loop."""

import unittest
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock, patch

from integrations.sglang.overlap import OverlapState, install, resolve_outputs


class StableDecodeOverlapTests(unittest.TestCase):
    def scheduler(self, *, lazy_resolve=False):
        class Native:
            def __init__(self):
                req = NS(rid="resident", finished=Mock(return_value=False), to_abort=False)
                self.last_batch = self.running_batch = NS(
                    reqs=[req], has_grammar=False, is_hybrid=False,
                    forward_mode=NS(is_decode=Mock(return_value=True)), prepare_for_decode=Mock(),
                )
                self.cur_batch = None
                self._autellix_window = NS(remaining=4, active={"resident"}, reserve=set(),
                                           can_continue_decode=Mock(return_value=False))
                self._autellix_overlap = NS(deferred_inputs=[], drained=None)
                self._autellix_reserve = {}
                self.waiting_queue, self.grammar_queue = [], []
                self.chunked_req = None
                self.token_to_kv_pool_allocator = NS(available_size=Mock(return_value=1))
                self.decode_mem_cache_buf_multiplier = 1
                self.autellix = NS(trace=True, emit=Mock())
                self.native_input = Mock()
                self.native_next = Mock(return_value=NS(forward_mode=NS(is_extend=lambda: False)))

            def process_input_requests(self, requests):
                self.native_input(requests, self.cur_batch)

            def get_next_batch_to_run(self):
                return self.native_next(self.last_batch)

        install(Native, lazy_resolve=lazy_resolve)
        return Native()

    def test_stable_continuation_then_deferred_control_at_empty_boundary(self):
        scheduler = self.scheduler()
        batch = scheduler.last_batch
        self.assertIs(scheduler.get_next_batch_to_run(), batch)
        batch.prepare_for_decode.assert_called_once_with()
        scheduler.native_next.assert_not_called()

        scheduler.process_input_requests(["cancel", "arrival"])
        self.assertIsNone(scheduler.get_next_batch_to_run())
        scheduler.native_input.assert_not_called()
        self.assertIs(scheduler._autellix_overlap.drained, batch)
        scheduler.autellix.emit.assert_any_call("overlap_drain", reason="input")
        # The native loop consumes the old result, then clears last_batch.
        scheduler.last_batch = None
        scheduler.process_input_requests(["new-arrival"])
        scheduler.native_input.assert_called_once_with(["cancel", "arrival", "new-arrival"], batch)
        self.assertIs(scheduler.get_next_batch_to_run(), scheduler.native_next.return_value)
        scheduler.native_next.assert_called_once_with(batch)
        self.assertIsNone(scheduler.last_batch)
        self.assertIsNone(scheduler._autellix_overlap.drained)

    def test_waiting_request_respects_the_native_window_refill_gate(self):
        scheduler = self.scheduler()
        scheduler.waiting_queue.append("waiting")
        self.assertIsNone(scheduler.get_next_batch_to_run())
        window = scheduler._autellix_window
        window.can_continue_decode.assert_called_once_with({"resident"}, {})
        scheduler.autellix.emit.assert_called_once_with("overlap_drain", reason="refill")

        scheduler._autellix_overlap.drained = None
        window.can_continue_decode.return_value = True
        self.assertIs(scheduler.get_next_batch_to_run(), scheduler.last_batch)

    def test_pending_decode_drains_before_new_window_retirement_or_pressure(self):
        for reason in ("window", "retire", "pressure", "prefill"):
            with self.subTest(reason=reason):
                scheduler = self.scheduler()
                batch = scheduler.last_batch
                if reason == "window":
                    scheduler._autellix_window.remaining = 0
                elif reason == "retire":
                    batch.reqs[0].finished.return_value = True
                elif reason == "pressure":
                    scheduler.token_to_kv_pool_allocator.available_size.return_value = 0
                else:
                    batch.forward_mode.is_decode.return_value = False
                self.assertIsNone(scheduler.get_next_batch_to_run())
                self.assertIs(scheduler._autellix_overlap.drained, batch)
                batch.prepare_for_decode.assert_not_called()
                scheduler.native_next.assert_not_called()
                scheduler.autellix.emit.assert_called_once_with("overlap_drain", reason=reason)

    def test_lazy_resolution_uses_post_plan_residents_only_when_split_prefill_runs(self):
        for mode in ("prefill", "mixed", "decode", "idle"):
            with self.subTest(mode=mode):
                scheduler = self.scheduler(lazy_resolve=True)
                drained = scheduler.last_batch
                scheduler._autellix_overlap.drained = drained
                scheduler.last_batch = None
                resident = NS(reqs=[NS(rid="kept")], output_ids=object())
                batch = None if mode == "idle" else NS(
                    forward_mode=NS(is_extend=lambda: mode in ("prefill", "mixed")),
                    decoding_reqs=resident.reqs if mode == "mixed" else None,
                    prefix_lens=[8, 17],
                )

                def select(previous):
                    self.assertIs(previous, drained)
                    scheduler.running_batch = resident  # Native merge/retirement replaces the old view.
                    return batch

                scheduler.native_next.side_effect = select
                with patch("integrations.sglang.overlap.resolve_outputs") as resolve:
                    self.assertIs(scheduler.get_next_batch_to_run(), batch)
                    if mode == "prefill":
                        resolve.assert_called_once_with(scheduler, resident, policy="paused_prefill")
                    else:
                        resolve.assert_not_called()
                self.assertIsNone(scheduler.last_batch)
                self.assertIsNone(scheduler._autellix_overlap.drained)

    def test_drained_mixed_prefix_corrects_only_the_decode_tail(self):
        for lazy_resolve in (False, True):
            with self.subTest(lazy_resolve=lazy_resolve):
                scheduler = self.scheduler(lazy_resolve=lazy_resolve)
                scheduler._autellix_overlap.drained = scheduler.last_batch
                scheduler.last_batch = None
                batch = NS(forward_mode=NS(is_extend=lambda: True),
                           decoding_reqs=[object(), object()], prefix_lens=[4, 0, 12, 23])
                scheduler.native_next.return_value = batch
                self.assertIs(scheduler.get_next_batch_to_run(), batch)
                self.assertEqual(batch.prefix_lens, [4, 0, 11, 22])

    def test_resolver_preserves_the_resident_output_tensor_identity(self):
        worker = ModuleType("sglang.srt.managers.tp_worker_overlap_thread")
        worker.resolve_future_token_ids = Mock()
        scheduler = self.scheduler()
        scheduler.tp_worker = NS(future_token_ids_map=object())
        resident = NS(reqs=[NS(rid="resident")], output_ids=object())
        ids = resident.output_ids
        with patch.dict("sys.modules", {worker.__name__: worker}):
            resolve_outputs(scheduler, resident, policy="paused_prefill")
            resolve_outputs(scheduler, NS(reqs=[], output_ids=object()), policy="paused_prefill")
        worker.resolve_future_token_ids.assert_called_once_with(ids, scheduler.tp_worker.future_token_ids_map)
        self.assertIs(resident.output_ids, ids)
        scheduler.autellix.emit.assert_called_once_with("overlap_resolve", policy="paused_prefill", requests=["resident"])

    def test_result_timing_is_keyed_by_bid_and_ignores_a_retired_delayed_token(self):
        worker = NS(forward_batch_generation=Mock(return_value="native"))
        starts = [Mock(), Mock()]
        starts[0].elapsed_time.return_value = 7
        starts[1].elapsed_time.return_value = 13
        torch = NS(cuda=NS(Event=Mock(side_effect=[starts[0], Mock(), starts[1], Mock()])))
        state = OverlapState(worker, torch)
        for bid in (41, 42):
            self.assertEqual(worker.forward_batch_generation(NS(bid=bid)), "native")
            state.pending[bid] = (["resident", "retires"], 0, None)

        calls = {rid: NS(scheduled=False) for rid in ("resident", "retires")}
        controller = NS(calls=calls, executed=Mock(), emit=Mock(), trace=True,
                        publish_activity=False)
        state.finish(41, controller)
        self.assertTrue(calls["resident"].scheduled)
        controller.executed.assert_called_once_with(["resident", "retires"], .007, prefill_count=0)
        calls.pop("retires")
        state.finish(42, controller)
        controller.executed.assert_called_with(["resident"], .013, prefill_count=0)
        self.assertEqual(state.pending, {})
        self.assertEqual(state.worker_events, {})
        self.assertEqual(len(state.free_events), 2)
        self.assertEqual(torch.cuda.Event.call_count, 4)
