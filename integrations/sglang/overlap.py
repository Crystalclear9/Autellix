"""Reuse the native overlap worker inside an unchanged decode window."""

from collections import deque


class OverlapState:
    def __init__(self, worker, torch):
        self.pending = {}
        self.worker_events = {}
        self.deferred_inputs = []
        self.drained = None
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
        controller.executed(rids, seconds, prefill_count=prefill_count)
        ongoing = [controller.calls[rid] for rid in self.active_requests() if rid in controller.calls]
        for call in ongoing:
            call.scheduled = True
        if ongoing and controller.publish_activity and not controller.publish_activity_at_window:
            controller._publish(ongoing)
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


def prepared_window(scheduler, running):
    window = scheduler._autellix_window
    prepared = scheduler._autellix_reserve
    if scheduler.waiting_queue:
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


def prepare_first_decode(scheduler, prefill):
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
            or not prepared_window(scheduler, rids)
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


def prepare_survivor_decode(scheduler, batch, natural_finish):
    """Shrink a decode window after normal terminal results arrive."""
    window = scheduler._autellix_window
    if (scheduler.waiting_queue or window.reserve or scheduler._autellix_reserve
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
    batch.prepare_for_decode()
    if scheduler.autellix.trace:
        scheduler.autellix.emit("overlap_retire_decode", requests=[req.rid for req in batch.reqs], retired=retired)
    return batch


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


def install(cls, *, lazy_resolve=False, prefill_decode=False, retire_decode=False, window_decode=False):
    native_input, native_next = cls.process_input_requests, cls.get_next_batch_to_run
    if retire_decode:
        from sglang.srt.managers.schedule_batch import FINISH_LENGTH, FINISH_MATCHED_TOKEN
        natural_finish = (FINISH_LENGTH, FINISH_MATCHED_TOKEN)

    def process_input(self, requests):
        state = self._autellix_overlap
        if self.last_batch is not None:
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
                pending.prepare_for_decode()
                if self.autellix.trace:
                    self.autellix.emit("overlap_continue_decode", requests=[req.rid for req in pending.reqs])
                return pending
            if prefill_decode and reason == "prefill":
                batch = prepare_first_decode(self, pending)
                if batch is not None:
                    return batch
            if retire_decode and reason == "retire":
                batch = prepare_survivor_decode(self, pending, natural_finish)
                if batch is not None:
                    return batch
            if window_decode and reason == "window":
                batch = prepare_window_decode(self, pending, state)
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
