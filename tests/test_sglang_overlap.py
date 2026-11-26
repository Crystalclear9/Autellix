"""Verify the boundary and accounting contract with SGLang's native overlap loop."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

from integrations.sglang.overlap import OverlapState, install


class StableDecodeOverlapTests(unittest.TestCase):
    def scheduler(self):
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
                self.native_next = Mock(return_value="regular")

            def process_input_requests(self, requests):
                self.native_input(requests, self.cur_batch)

            def get_next_batch_to_run(self):
                return self.native_next(self.last_batch)

        install(Native)
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
        # The native loop consumes the old result, then clears last_batch.
        scheduler.last_batch = None
        scheduler.process_input_requests(["new-arrival"])
        scheduler.native_input.assert_called_once_with(["cancel", "arrival", "new-arrival"], batch)
        self.assertEqual(scheduler.get_next_batch_to_run(), "regular")
        scheduler.native_next.assert_called_once_with(batch)
        self.assertIsNone(scheduler.last_batch)
        self.assertIsNone(scheduler._autellix_overlap.drained)

    def test_waiting_request_respects_the_native_window_refill_gate(self):
        scheduler = self.scheduler()
        scheduler.waiting_queue.append("waiting")
        self.assertIsNone(scheduler.get_next_batch_to_run())
        window = scheduler._autellix_window
        window.can_continue_decode.assert_called_once_with({"resident"}, {})

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
