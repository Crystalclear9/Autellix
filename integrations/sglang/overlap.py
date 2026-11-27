"""Reuse the native overlap worker inside an unchanged decode window."""

from collections import deque


class OverlapState:
    def __init__(self, worker, torch):
        self.pending = {}
        self.worker_events = {}
        self.deferred_inputs = []
        self.drained = None
        self.parked_decode = None
        self.window_step = None
        self.free_events = deque((torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                                 for _ in range(2))
        original_forward = worker.forward_batch_generation

        def timed_forward(batch, *args, **kwargs):
            start, end = self.free_events.popleft()
            start.record()
            value = original_forward(batch, *args, **kwargs)
            end.record()
            self.worker_events[batch.bid] = (start, end)
            return value

        worker.forward_batch_generation = timed_forward

    def active_requests(self):
        return {rid for rids, _, _ in self.pending.values() for rid in rids}

    def finish(self, bid, controller):
        original_rids, prefill_count, phase = self.pending.pop(bid)
        start, end = self.worker_events.pop(bid)
        # Native output readback has synchronized a later event on this stream.
        seconds = start.elapsed_time(end) / 1000.
        self.free_events.append((start, end))
        # An earlier result can finish a request while its one delayed decode
        # is still in flight. That discarded result does not extend service.
        rids = [rid for rid in original_rids if rid in controller.calls]
        if prefill_count:
            prefill_count = sum(rid in controller.calls for rid in original_rids[:prefill_count])
        controller.executed(rids, seconds, prefill_count=prefill_count, still_running=self.active_requests())
        if phase is not None:
            controller.emit("model_phase_time", seconds=seconds, requests=rids, **phase)
        if controller.trace:
            controller.emit("overlap_result", batch_id=bid, requests=rids)

    def finish_window_step(self, bid, window, controller):
        previous_bid, issued = self.window_step
        if bid != previous_bid:
            return
        self.window_step = None
        # The preceding result can retire requests. Native overlap discards
        # their delayed token, so only survivors join the newly ranked plan.
        rids = [rid for rid in issued if rid in controller.calls]
        window.select(rids, rids)
        if rids:
            controller.emit("plan_step", requests=rids, decode=True,
                            remaining=window.remaining, input_tokens=len(issued))
            window.executed()


def prepared_window(scheduler, running, *, refill_now=True):
    window = scheduler._autellix_window
    prepared = scheduler._autellix_reserve
    if refill_now and scheduler.waiting_queue:
        # A queued request alone need not break a full or capacity-limited plan.
        return window.can_continue_decode(running, prepared)
    return running == window.active and window.reserve.issubset(prepared)


def drain_reason(scheduler, pending, state):
    window = scheduler._autellix_window
    if state.deferred_inputs:
        return "input"
    if not pending.forward_mode.is_decode() or scheduler.chunked_req is not None:
        return "prefill"
    if not window.remaining:
        return "window"
    if pending is not scheduler.running_batch or not pending.reqs or pending.is_hybrid:
        return "cohort"
    if pending.has_grammar or scheduler.grammar_queue:
        return "grammar"
    if any(req.finished() or req.to_abort for req in pending.reqs):
        return "retire"
    if (scheduler.token_to_kv_pool_allocator.available_size()
            < len(pending.reqs) * scheduler.decode_mem_cache_buf_multiplier):
        return "pressure"
    if prepared_window(scheduler, {req.rid for req in pending.reqs}):
        return None
    return "refill" if scheduler.waiting_queue else "cohort"


def prefill_into_idle_slots(scheduler, running, state):
    window, host = scheduler._autellix_window, scheduler._autellix_host
    if (not window.refill_idle_slots or window.capacity_limited
            or window.reserve or scheduler._autellix_reserve
            or len(running.reqs) >= window.capacity
            or {req.rid for req in running.reqs} != window.active
            or (host is not None and any(req.rid in host.saved for req in scheduler.waiting_queue))):
        return None

    # At a refill point inside the window, new admissions are an append-only
    # suffix. The running requests keep their slots and radix locks.
    batch = scheduler.get_new_batch_prefill()
    if batch is not None:
        state.parked_decode = running
        if scheduler.autellix.trace:
            scheduler.autellix.emit("overlap_refill_prefill", requests=[req.rid for req in batch.reqs])
    return batch


def prepare_first_decode(scheduler, prefill, *, refill_now=True):
    """Use the native prefill handoff without making an admission decision."""
    window, running = scheduler._autellix_window, scheduler.running_batch
    if (not window.remaining or scheduler.chunked_req is not None or scheduler.grammar_queue
            or not prefill.forward_mode.is_extend() or prefill.decoding_reqs or prefill.chunked_req is not None
            or prefill.has_grammar or prefill.is_hybrid or not prefill.reqs or running is prefill
            or (running.reqs and (running.has_grammar or running.is_hybrid))):
        return None
    requests = running.reqs + prefill.reqs
    rids = {req.rid for req in requests}
    if (len(rids) != len(requests) or any(req.finished() or req.to_abort for req in requests)
            or not prepared_window(scheduler, rids, refill_now=refill_now)
            or scheduler.token_to_kv_pool_allocator.available_size()
            < len(requests) * scheduler.decode_mem_cache_buf_multiplier):
        return None

    # The native loop already queued a result copy of the prefill. Merge and
    # append decode positions exactly as native overlap would before readback.
    # Free space was checked without evicting anything. The worker serializes
    # KV writes on its forward stream, and the prior batch's lengths stay fixed.
    if running.reqs:
        running.merge_batch(prefill)
    else:
        running = scheduler.running_batch = prefill
    running.prepare_for_decode()
    if scheduler.autellix.trace:
        scheduler.autellix.emit("overlap_prefill_decode", requests=[req.rid for req in requests],
                                prefill_requests=[req.rid for req in prefill.reqs])
    return running


def retire_finished(scheduler, batch, natural_finish, *, refill):
    """Remove finished entries while preserving the queued result's KV frees."""
    window = scheduler._autellix_window
    if ((scheduler.waiting_queue and (not refill or window.capacity_limited))
            or window.reserve or scheduler._autellix_reserve
            or {req.rid for req in batch.reqs} != window.active):
        return None
    survivors, retired = [], []
    for index, req in enumerate(batch.reqs):
        if req.to_abort:
            return None
        if req.finished():
            if not isinstance(req.finished_reason, natural_finish):
                return None
            retired.append(req.rid)
        else:
            survivors.append(index)
    if (not survivors or not retired or scheduler.token_to_kv_pool_allocator.available_size()
            < len(survivors) * scheduler.decode_mem_cache_buf_multiplier):
        return None

    # Native filter replaces the live arrays. The queued result copy still
    # owns the original request list and out_cache_loc for delayed frees.
    batch.filter_batch(keep_indices=survivors)
    batch.batch_is_full = False
    rids = {req.rid for req in batch.reqs}
    window.select(rids, rids)
    return retired


def prepare_window_decode(scheduler, batch, state):
    """Stage one decode while readback closes a window with no contenders."""
    window = scheduler._autellix_window
    controller = scheduler.autellix
    if (batch is not scheduler.running_batch or not batch.reqs or batch.is_hybrid
            or batch.has_grammar or scheduler.grammar_queue or scheduler.waiting_queue
            or window.reserve or scheduler._autellix_reserve
            or (controller.publish_activity and not controller.publish_activity_at_window)
            or any(req.finished() or req.to_abort for req in batch.reqs)
            or scheduler.token_to_kv_pool_allocator.available_size()
            < len(batch.reqs) * scheduler.decode_mem_cache_buf_multiplier):
        return None
    rids = [req.rid for req in batch.reqs]
    resident = set(rids)
    if resident != window.active or resident != controller.calls.keys():
        return None

    # With every contender already resident, the next ranking can reorder
    # the plan for future admission but cannot change which survivors run.
    previous_bid = next(iter(state.pending))
    batch.prepare_for_decode()
    state.window_step = (previous_bid, rids)
    if controller.trace:
        controller.emit("overlap_window_decode", requests=rids, after_batch_id=previous_bid)
    return batch


def resolve_outputs(scheduler, resident, *, policy):
    """Replace negative future IDs after the worker's native readback fence."""
    if resident.reqs and resident.output_ids is not None:
        from sglang.srt.managers.tp_worker_overlap_thread import resolve_future_token_ids
        resolve_future_token_ids(resident.output_ids, scheduler.tp_worker.future_token_ids_map)
        if scheduler.autellix.trace:
            scheduler.autellix.emit("overlap_resolve", policy=policy, requests=[req.rid for req in resident.reqs])


def install(cls, *, lazy_resolve=False, prefill_decode=False, retire_decode=False, window_decode=False,
            prefill_refill=False, prepare_decode=None):
    native_input, native_next = cls.process_input_requests, cls.get_next_batch_to_run
    if prefill_refill:
        from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
        from .scheduler import decode_request

        def independent_arrivals(scheduler, pending, requests):
            calls = scheduler.autellix.calls
            pending_programs = {calls[req.rid].pid for req in pending.reqs if req.rid in calls}
            return all(isinstance(req, TokenizedGenerateReqInput) and req.session_params is None
                       and req.rid.startswith("autellix_")
                       and decode_request(req.rid)[0] not in pending_programs for req in requests)

    if retire_decode:
        from sglang.srt.managers.schedule_batch import FINISH_LENGTH, FINISH_MATCHED_TOKEN
        natural_finish = (FINISH_LENGTH, FINISH_MATCHED_TOKEN)

    def process_input(self, requests):
        state = self._autellix_overlap
        if self.last_batch is not None:
            # Queue-only arrivals from other programs cannot change the
            # pending result or inherit service that it has yet to publish.
            if (prefill_refill and requests and not state.deferred_inputs
                    and independent_arrivals(self, self.last_batch, requests)):
                return native_input(self, requests)
            state.deferred_inputs.extend(requests)
            return
        if state.drained is not None:
            # Native cancellation checks current as well as running requests.
            self.cur_batch = state.drained
        if state.deferred_inputs:
            state.deferred_inputs.extend(requests)
            requests, state.deferred_inputs = state.deferred_inputs, []
        return native_input(self, requests)

    def get_next(self):
        state = self._autellix_overlap
        pending = self.last_batch
        if pending is not None:
            reason = drain_reason(self, pending, state)
            if reason is None:
                if prepare_decode is not None and prepare_decode(pending):
                    if self.autellix.trace:
                        self.autellix.emit("fused_decode_metadata", requests=[req.rid for req in pending.reqs])
                else:
                    pending.prepare_for_decode()
                if self.autellix.trace:
                    self.autellix.emit("overlap_continue_decode", requests=[req.rid for req in pending.reqs])
                return pending
            if prefill_decode and reason == "prefill":
                # Launch the admitted cohort's first decode. Its readback
                # can then overlap any prefill that fills the idle slots.
                batch = prepare_first_decode(self, pending, refill_now=not prefill_refill)
                if batch is not None:
                    return batch
            if retire_decode and reason == "retire":
                retired = retire_finished(self, pending, natural_finish, refill=prefill_refill)
                if retired is not None:
                    if self.waiting_queue and self._autellix_window.refill_idle_slots:
                        batch = prefill_into_idle_slots(self, pending, state)
                        if batch is not None:
                            return batch
                    else:
                        pending.prepare_for_decode()
                        if self.autellix.trace:
                            self.autellix.emit("overlap_retire_decode", requests=[req.rid for req in pending.reqs],
                                               retired=retired)
                        return pending
            if window_decode and reason == "window":
                batch = prepare_window_decode(self, pending, state)
                if batch is not None:
                    return batch
            if prefill_refill and reason == "refill":
                batch = prefill_into_idle_slots(self, pending, state)
                if batch is not None:
                    return batch
            state.drained = pending
            if self.autellix.trace:
                self.autellix.emit("overlap_drain", reason=reason)
            # The native loop processes the last copy without launching a new
            # batch. Its next iteration starts with an empty result queue.
            return None
        self.last_batch, state.drained = state.drained, None
        try:
            # Merge an already-processed prefill into the running batch, using
            # the same native path that handles chunk ownership and retirement.
            batch = native_next(self)
            if batch is not None and batch.forward_mode.is_extend():
                if batch.decoding_reqs:
                    # Native overlap uses a zero delta for the decode tail's
                    # CPU token list because the previous result is pending.
                    # We only select mixed batches after processing that result.
                    first_decode = len(batch.prefix_lens) - len(batch.decoding_reqs)
                    for index in range(first_decode, len(batch.prefix_lens)):
                        batch.prefix_lens[index] -= 1
                elif lazy_resolve:
                    # An immediate decode or mixed batch consumes its future
                    # IDs in the worker. Separate prefill can park a decode
                    # batch until the future ring has wrapped.
                    resolve_outputs(self, self.running_batch, policy="paused_prefill")
            return batch
        finally:
            # The native event loop must bootstrap sampling for the next launch.
            self.last_batch = None

    cls.process_input_requests, cls.get_next_batch_to_run = process_input, get_next
