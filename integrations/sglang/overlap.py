"""Reuse the native overlap worker inside an unchanged decode window."""

from collections import deque


class OverlapState:
    def __init__(self, worker, torch):
        self.pending = {}
        self.worker_events = {}
        self.deferred_inputs = []
        self.drained = None
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


def can_continue(scheduler, pending, state):
    window = scheduler._autellix_window
    if (not window.remaining or pending is not scheduler.running_batch
            or not pending.forward_mode.is_decode() or not pending.reqs or pending.has_grammar
            or state.deferred_inputs or scheduler.grammar_queue or scheduler.chunked_req is not None
            or pending.is_hybrid or any(req.finished() or req.to_abort for req in pending.reqs)
            or scheduler.token_to_kv_pool_allocator.available_size()
            < len(pending.reqs) * scheduler.decode_mem_cache_buf_multiplier):
        return False
    running = {req.rid for req in pending.reqs}
    prepared = scheduler._autellix_reserve
    if scheduler.waiting_queue:
        # Use the same prepared-window fast path as synchronous decode. A
        # queued request alone need not break a full or capacity-limited plan.
        return window.can_continue_decode(running, prepared)
    return running == window.active and window.reserve.issubset(prepared)


def install(cls):
    native_input, native_next = cls.process_input_requests, cls.get_next_batch_to_run

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
            if can_continue(self, pending, state):
                pending.prepare_for_decode()
                if self.autellix.trace:
                    self.autellix.emit("overlap_continue_decode", requests=[req.rid for req in pending.reqs])
                return pending
            state.drained = pending
            if self.autellix.trace:
                self.autellix.emit("overlap_drain")
            # The native loop processes the last copy without launching a new
            # batch. Its next iteration starts with an empty result queue.
            return None
        self.last_batch, state.drained = state.drained, None
        try:
            # Merge an already-processed prefill into the running batch, using
            # the same native path that handles chunk ownership and retirement.
            return native_next(self)
        finally:
            # The native event loop must bootstrap sampling for the next launch.
            self.last_batch = None

    cls.process_input_requests, cls.get_next_batch_to_run = process_input, get_next
