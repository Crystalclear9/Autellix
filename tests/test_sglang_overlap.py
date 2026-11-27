"""Verify the boundary and accounting contract with SGLang's native overlap loop."""

import unittest
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock, patch

from autellix.runtime.window import SchedulingWindow
from integrations.sglang.overlap import OverlapState, install, resolve_outputs
from integrations.sglang.scheduler import encode_request


class LengthFinish:
    pass


class TokenFinish:
    pass


class GenerateInput:
    def __init__(self, program):
        self.rid = encode_request(program, "arrival")
        self.session_params = None


class StableDecodeOverlapTests(unittest.TestCase):
    def scheduler(self, *, lazy_resolve=False, prefill_decode=False, retire_decode=False, window_decode=False,
                  prefill_refill=False):
        class Native:
            def __init__(self):
                req = NS(rid="resident", finished=Mock(return_value=False), to_abort=False)
                self.last_batch = self.running_batch = NS(
                    reqs=[req], has_grammar=False, is_hybrid=False,
                    forward_mode=NS(is_decode=Mock(return_value=True)), prepare_for_decode=Mock(),
                )
                self.cur_batch = None
                self._autellix_window = NS(remaining=4, active={"resident"}, reserve=set(),
                                           capacity=2, capacity_limited=False, refill_idle_slots=True,
                                           can_continue_decode=Mock(return_value=False))
                self._autellix_overlap = NS(deferred_inputs=[], drained=None, pending={31: None},
                                            window_step=None, parked_decode=None)
                self._autellix_reserve = {}
                self._autellix_host = None
                self.waiting_queue, self.grammar_queue = [], []
                self.chunked_req = None
                self.token_to_kv_pool_allocator = NS(available_size=Mock(return_value=1))
                self.decode_mem_cache_buf_multiplier = 1
                self.autellix = NS(trace=True, emit=Mock(), calls={"resident": NS(pid="active")},
                                  publish_activity=False, publish_activity_at_window=False)
                self.get_new_batch_prefill = Mock()
                self.native_input = Mock()
                self.native_next = Mock(return_value=NS(forward_mode=NS(is_extend=lambda: False)))

            def process_input_requests(self, requests):
                self.native_input(requests, self.cur_batch)

            def get_next_batch_to_run(self):
                return self.native_next(self.last_batch)

        module = ModuleType("sglang.srt.managers.schedule_batch")
        module.FINISH_LENGTH, module.FINISH_MATCHED_TOKEN = LengthFinish, TokenFinish
        io = ModuleType("sglang.srt.managers.io_struct")
        io.TokenizedGenerateReqInput = GenerateInput
        with patch.dict("sys.modules", {module.__name__: module, io.__name__: io}):
            install(Native, lazy_resolve=lazy_resolve, prefill_decode=prefill_decode,
                    retire_decode=retire_decode, window_decode=window_decode, prefill_refill=prefill_refill)
        return Native()

    def test_boundary_handoff_drains_when_the_next_cohort_is_not_fixed(self):
        for reason in ("input", "waiting", "reserve", "pressure", "retire", "cohort", "absent", "grammar",
                       "publication", "chunk"):
            with self.subTest(reason=reason):
                scheduler = self.scheduler(window_decode=True)
                batch, window = scheduler.last_batch, scheduler._autellix_window
                window.remaining = 0
                if reason == "input":
                    scheduler.process_input_requests(["control"])
                elif reason == "waiting":
                    scheduler.waiting_queue.append("waiting")
                elif reason == "reserve":
                    window.reserve.add("other")
                elif reason == "pressure":
                    scheduler.token_to_kv_pool_allocator.available_size.return_value = 0
                elif reason == "retire":
                    batch.reqs[0].finished.return_value = True
                elif reason == "cohort":
                    window.active.add("other")
                elif reason == "absent":
                    scheduler.autellix.calls["other"] = None
                elif reason == "grammar":
                    scheduler.grammar_queue.append("grammar")
                elif reason == "publication":
                    scheduler.autellix.publish_activity = True
                else:
                    scheduler.chunked_req = object()
                self.assertIsNone(scheduler.get_next_batch_to_run())
                self.assertIs(scheduler._autellix_overlap.drained, batch)
                self.assertIsNone(scheduler._autellix_overlap.window_step)
                batch.prepare_for_decode.assert_not_called()

    def pending_retirement(self):
        scheduler = self.scheduler(retire_decode=True)
        batch = scheduler.last_batch
        batch.reqs = [NS(rid="short", finished=Mock(return_value=True), to_abort=False, finished_reason=LengthFinish()),
                      batch.reqs[0],
                      NS(rid="eos", finished=Mock(return_value=True), to_abort=False, finished_reason=TokenFinish())]
        batch.out_cache_loc = object()
        batch.batch_is_full = True
        window = SchedulingWindow(NS(key=Mock(), emit=Mock()), capacity=3, refill_idle_slots=True)
        window.remaining = 4
        window.cohort = [req.rid for req in batch.reqs]
        window.active = set(window.cohort)
        scheduler._autellix_window = window

        def filter_current(*, keep_indices):
            batch.reqs = [batch.reqs[index] for index in keep_indices]
            batch.out_cache_loc = None

        batch.filter_batch = Mock(side_effect=filter_current)
        return scheduler, batch

    def test_natural_retirement_preserves_the_queued_result_and_window(self):
        scheduler, batch = self.pending_retirement()
        queued = NS(reqs=batch.reqs, out_cache_loc=batch.out_cache_loc)
        self.assertIs(scheduler.get_next_batch_to_run(), batch)
        batch.filter_batch.assert_called_once_with(keep_indices=[1])
        batch.prepare_for_decode.assert_called_once_with()
        self.assertEqual([req.rid for req in batch.reqs], ["resident"])
        self.assertEqual([req.rid for req in queued.reqs], ["short", "resident", "eos"])
        self.assertIsNotNone(queued.out_cache_loc)
        self.assertFalse(batch.batch_is_full)
        self.assertEqual(scheduler._autellix_window.cohort, ["resident"])
        self.assertEqual(scheduler._autellix_window.remaining, 4)
        self.assertIsNone(scheduler._autellix_overlap.drained)
        scheduler.autellix.emit.assert_called_once_with("overlap_retire_decode", requests=["resident"],
                                                        retired=["short", "eos"])

    def test_retirement_drains_before_control_pressure_or_nonordinary_finish(self):
        for reason in ("input", "waiting", "reserve", "pressure", "abort", "unknown", "empty", "window", "cohort"):
            with self.subTest(reason=reason):
                scheduler, batch = self.pending_retirement()
                window = scheduler._autellix_window
                if reason == "input":
                    scheduler.process_input_requests(["control"])
                elif reason == "waiting":
                    scheduler.waiting_queue.append("waiting")
                elif reason == "reserve":
                    scheduler._autellix_reserve["other"] = "pin"
                elif reason == "pressure":
                    scheduler.token_to_kv_pool_allocator.available_size.return_value = 0
                elif reason == "abort":
                    batch.reqs[0].to_abort = True
                elif reason == "unknown":
                    batch.reqs[0].finished_reason = object()
                elif reason == "empty":
                    batch.reqs[1].finished.return_value = True
                    batch.reqs[1].finished_reason = LengthFinish()
                elif reason == "window":
                    window.remaining = 0
                else:
                    window.active.remove("resident")
                self.assertIsNone(scheduler.get_next_batch_to_run())
                self.assertIs(scheduler._autellix_overlap.drained, batch)
                batch.filter_batch.assert_not_called()
                batch.prepare_for_decode.assert_not_called()

    def pending_prefill(self, *, resident=True, prefill_refill=False):
        scheduler = self.scheduler(prefill_decode=True, prefill_refill=prefill_refill)
        running = scheduler.running_batch
        if not resident:
            running.reqs.clear()
        running.merge_batch = Mock(side_effect=lambda batch: running.reqs.extend(batch.reqs))
        req = NS(rid="prefill", finished=Mock(return_value=False), to_abort=False)
        prefill = NS(reqs=[req], has_grammar=False, is_hybrid=False, chunked_req=None,
                     decoding_reqs=None, forward_mode=NS(is_decode=lambda: False, is_extend=lambda: True),
                     prepare_for_decode=Mock())
        scheduler.last_batch = prefill
        scheduler._autellix_window.active = {req.rid for req in running.reqs + prefill.reqs}
        scheduler.token_to_kv_pool_allocator.available_size.return_value = 2
        return scheduler, running, prefill

    def test_prefill_to_first_decode_reuses_the_already_prepared_window(self):
        for resident in (False, True):
            with self.subTest(resident=resident):
                scheduler, running, prefill = self.pending_prefill(resident=resident)
                prefill_requests = prefill.reqs.copy()
                result = scheduler.get_next_batch_to_run()
                self.assertIs(result, running if resident else prefill)
                self.assertIs(result, scheduler.running_batch)
                self.assertEqual(prefill.reqs, prefill_requests)
                result.prepare_for_decode.assert_called_once_with()
                if resident:
                    running.merge_batch.assert_called_once_with(prefill)
                else:
                    running.merge_batch.assert_not_called()
                self.assertIsNone(scheduler._autellix_overlap.drained)
                scheduler.native_next.assert_not_called()
                requests = ["resident", "prefill"] if resident else ["prefill"]
                scheduler.autellix.emit.assert_called_once_with("overlap_prefill_decode", requests=requests,
                                                                prefill_requests=["prefill"])

    def test_prefill_to_decode_drains_if_a_fresh_plan_or_more_slots_are_needed(self):
        for reason in ("window", "input", "chunk", "mixed", "pressure", "retire", "refill", "reserve"):
            with self.subTest(reason=reason):
                scheduler, running, prefill = self.pending_prefill()
                if reason == "window":
                    scheduler._autellix_window.remaining = 0
                elif reason == "input":
                    scheduler.process_input_requests(["cancel"])
                elif reason == "chunk":
                    prefill.chunked_req = prefill.reqs[0]
                elif reason == "mixed":
                    prefill.decoding_reqs = [object()]
                elif reason == "pressure":
                    scheduler.token_to_kv_pool_allocator.available_size.return_value = 1
                elif reason == "retire":
                    running.reqs[0].finished.return_value = True
                elif reason == "refill":
                    scheduler.waiting_queue.append("waiting")
                else:
                    scheduler._autellix_window.reserve.add("unprepared")
                self.assertIsNone(scheduler.get_next_batch_to_run())
                self.assertIs(scheduler._autellix_overlap.drained, prefill)
                running.merge_batch.assert_not_called()
                prefill.prepare_for_decode.assert_not_called()
                running.prepare_for_decode.assert_not_called()

    def test_first_decode_handoff_precedes_refill_of_idle_slots(self):
        scheduler, running, prefill = self.pending_prefill(prefill_refill=True)
        scheduler._autellix_window.capacity = 3
        arrival = GenerateInput("other")
        scheduler.native_input.side_effect = lambda requests, _: scheduler.waiting_queue.extend(requests)
        scheduler.process_input_requests([arrival])
        self.assertEqual(scheduler.waiting_queue, [arrival])
        self.assertIs(scheduler.get_next_batch_to_run(), running)
        running.merge_batch.assert_called_once_with(prefill)
        running.prepare_for_decode.assert_called_once_with()

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

    def test_refill_can_enqueue_other_programs_and_launch_prefill_before_readback(self):
        scheduler = self.scheduler(prefill_refill=True)
        batch = scheduler.last_batch
        arrival = GenerateInput("other")
        scheduler.native_input.side_effect = lambda requests, _: scheduler.waiting_queue.extend(requests)
        scheduler.process_input_requests([arrival])
        scheduler.native_input.assert_called_once_with([arrival], None)
        prefill = NS(reqs=[arrival])
        scheduler.get_new_batch_prefill.return_value = prefill

        self.assertIs(scheduler.get_next_batch_to_run(), prefill)
        self.assertIs(scheduler._autellix_overlap.parked_decode, batch)
        self.assertIsNone(scheduler._autellix_overlap.drained)
        self.assertEqual(scheduler._autellix_window.remaining, 4)
        batch.prepare_for_decode.assert_not_called()

    def test_refill_keeps_controls_and_program_service_dependencies_after_readback(self):
        for requests in ([GenerateInput("active")], [GenerateInput("other"), "cancel"]):
            with self.subTest(requests=requests):
                scheduler = self.scheduler(prefill_refill=True)
                scheduler.process_input_requests(requests)
                self.assertIsNone(scheduler.get_next_batch_to_run())
                scheduler.native_input.assert_not_called()
                scheduler.get_new_batch_prefill.assert_not_called()
                self.assertEqual(scheduler._autellix_overlap.deferred_inputs, requests)

    def test_refill_does_not_replace_a_running_request_at_a_window_boundary(self):
        scheduler = self.scheduler(prefill_refill=True)
        scheduler.waiting_queue = [GenerateInput("other")]
        scheduler._autellix_window.remaining = 0
        self.assertIsNone(scheduler.get_next_batch_to_run())
        scheduler.get_new_batch_prefill.assert_not_called()
        self.assertIsNone(scheduler._autellix_overlap.parked_decode)

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
