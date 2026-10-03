from __future__ import annotations

import base64
import json


def encode_request(pid, rid):
    return "autellix_" + base64.urlsafe_b64encode(json.dumps([pid, rid]).encode()).decode()


def decode_request(value):
    if not value.startswith("autellix_"):
        raise ValueError("request lacks Autellix program metadata")
    pid, rid = json.loads(base64.urlsafe_b64decode(value[9:]))
    return pid, rid


def run_scheduler_process(*args, autellix_options, **kwargs):
    """Spawn-safe process target: install hooks inside the actual GPU process."""
    import sglang.srt.managers.scheduler as module
    install(module.Scheduler, autellix_options)
    return module.run_scheduler_process(*args, **kwargs)


def install(cls, options):
    from .reserve import release_reserves
    from autellix.runtime.policy import PolicyConfig, ProgramTable, RuntimeScheduler
    import torch

    original_init = cls.__init__
    original_extend = cls._extend_requests_to_queue
    original_add = cls._add_request_to_queue
    original_prefill = cls.get_new_batch_prefill
    original_run = cls.run_batch
    original_result = cls.process_batch_result
    original_abort = cls.abort_request
    original_check_memory = cls.check_memory
    original_update_running = getattr(cls, "update_running_batch", None)

    def init(self, *a, **kw):
        original_init(self, *a, **kw)
        self.autellix = RuntimeScheduler(ProgramTable(options["table_path"]),
                                        PolicyConfig(**options["config"]), options.get("trace"))
        self._autellix_steps = 0
        self._autellix_paper = self.autellix.config.implementation == "paper"
        if self._autellix_paper:
            # Mixed execution prevents native prefill preference from excluding
            # higher-priority decodes already admitted by the global plan.
            self.is_mixed_chunk = True
        self._autellix_rids = {}
        self._autellix_reserve = {}
        self._autellix_window = None
        if options.get("capacity"):
            from autellix.runtime.window import SchedulingWindow
            self._autellix_window = SchedulingWindow(self.autellix, options["capacity"])
        if options.get("host_bytes"):
            from .swap import HostKV
            self._autellix_host = HostKV(self.token_to_kv_pool_allocator, self.tree_cache, options["host_bytes"])
        else:
            self._autellix_host = None
        # The underlying FCFS policy must not overwrite Autellix queue ordering.
        def priority(waiting_queue, *args, **kwargs):
            if self._autellix_paper and self._autellix_window is not None:
                order = {rid: i for i, rid in enumerate(self._autellix_window.cohort)}
                waiting_queue.sort(key=lambda req: order[req.rid])
            else:
                waiting_queue.sort(key=lambda req: self.autellix.key(req.rid))
            return False
        self.policy.calc_priority = priority

    def extend(self, reqs, is_retracted=False):
        for req in reqs:
            if req.rid not in self.autellix.calls:
                pid, external_rid = decode_request(req.rid)
                self.autellix.admit(req.rid, pid, metadata=self.autellix.table.take_context(external_rid))
                self._autellix_rids[req.rid] = external_rid
        return original_extend(self, reqs, is_retracted=is_retracted)

    def add(self, req):
        if req.rid not in self.autellix.calls:
            pid, external_rid = decode_request(req.rid)
            self.autellix.admit(req.rid, pid, metadata=self.autellix.table.take_context(external_rid))
            self._autellix_rids[req.rid] = external_rid
        return original_add(self, req)

    def prefill(self):
        ctl = self.autellix
        # Decode results retire requests before native update_running_batch
        # filters its tensors. We inspect this batch earlier, so apply the
        # native filter first, including on iterations without a priority refresh.
        batch = self.running_batch
        previous_size = len(batch.reqs)
        batch.filter_batch()
        if len(batch.reqs) < previous_size:
            batch.batch_is_full = False
        self._autellix_steps += 1
        selected = None
        if self._autellix_window is not None:
            old_remaining = self._autellix_window.remaining
            selected, _ = self._autellix_window.select(
                [r.rid for r in batch.reqs + self.waiting_queue],
                [r.rid for r in batch.reqs] + list(self._autellix_reserve))
            for rid in list(self._autellix_reserve):
                if rid not in self._autellix_window.cohort:
                    release_reserves(self, [rid])
            if (self._autellix_paper and old_remaining > 0 and
                    selected == {r.rid for r in batch.reqs}):
                ctl.emit("continue_plan", requests=[r.rid for r in batch.reqs])
                return None
        if selected is not None or (self._autellix_steps - 1) % ctl.config.schedule_interval == 0:
            if selected is None:
                ctl.refresh()
            batch = self.running_batch
            if batch.reqs and (self.waiting_queue or selected is not None):
                limit = self.server_args.max_running_requests
                keep = selected if selected is not None else {
                    r.rid for r in sorted(batch.reqs + self.waiting_queue, key=lambda r: ctl.key(r.rid))[:limit]}
                if self._autellix_paper and self._autellix_window is not None:
                    order = {rid: i for i, rid in enumerate(self._autellix_window.cohort)}
                    pending = [order[r.rid] for r in self.waiting_queue if r.rid in keep]
                    if pending:
                        # Native admission accounts running requests before new
                        # prefills. Release the lower-priority running suffix so
                        # it cannot consume the space needed by a higher arrival.
                        # Re-admit that suffix in the same global FIFO order.
                        cutoff = min(pending)
                        keep = {rid for rid in keep if order[rid] < cutoff}
                victims = [i for i, r in enumerate(batch.reqs) if r.rid not in keep]
                # Use the pinned engine's native retraction ownership rules.
                # Generated output_ids are preserved by reset_for_retract().
                lengths = batch.seq_lens.cpu().tolist()
                retracted = []
                for i in victims:
                    req = batch.reqs[i]
                    if (len(self._autellix_reserve) < ctl.config.overprovision and
                            (self._autellix_window is None or req.rid in self._autellix_window.reserve)):
                        # Preserve computed GPU KV as a locked radix prefix;
                        # only the request slot is released. Unlike recompute,
                        # admission can reuse all previously computed tokens.
                        req.fill_ids = (req.origin_input_ids + req.output_ids)[:lengths[i]]
                        req._autellix_reserved_tokens = list(req.fill_ids)
                        batch.tree_cache.cache_unfinished_req(req)
                        self._autellix_reserve[req.rid] = req.last_node
                        batch.req_to_token_pool.free(req.req_pool_idx)
                        ctl.emit("reserve", rid=req.rid)
                    else:
                        if self._autellix_host is not None:
                            all_indices = batch.req_to_token_pool.req_to_token[req.req_pool_idx, :lengths[i]]
                            tokens = (req.origin_input_ids + req.output_ids)[:lengths[i]]
                            self._autellix_host.save(req.rid, tokens, all_indices)
                            ctl.emit("swap_out", rid=req.rid, tokens=len(tokens))
                        start = len(req.prefix_indices)
                        indices = batch.req_to_token_pool.req_to_token[req.req_pool_idx, start:lengths[i]]
                        batch.token_to_kv_pool_allocator.free(indices)
                        batch.req_to_token_pool.free(req.req_pool_idx)
                        batch.tree_cache.dec_lock_ref(req.last_node)
                    req.reset_for_retract()
                    retracted.append(req)
                    ctl.emit("retract", rid=req.rid)
                if victims:
                    batch.filter_batch(keep_indices=[i for i in range(len(batch.reqs)) if i not in victims])
                    batch.batch_is_full = False
                    self._extend_requests_to_queue(retracted, is_retracted=True)
        slots = max(0, self.server_args.max_running_requests - len(self.running_batch.reqs))
        ready = sorted((r for r in self.waiting_queue if selected is None or r.rid in selected),
                       key=lambda r: ctl.key(r.rid))[:slots]
        if self._autellix_paper and self._autellix_window is not None and not ready:
            # Decode continuation: no native prefill admission / priority pass.
            return None
        blocked = set()
        for ready_index, req in enumerate(ready):
            if (self._autellix_paper and self._autellix_window is not None
                    and req.rid not in self._autellix_window.cohort):
                blocked.add(req.rid)
                continue
            if self._autellix_host is not None and req.rid in self._autellix_host.saved:
                restored = self._autellix_host.restore(req.rid)
                if not restored and self._autellix_reserve:
                    inactive = (self._autellix_window.reserve & self._autellix_reserve.keys()
                                if self._autellix_window is not None else self._autellix_reserve.keys())
                    release_reserves(self, inactive)
                    if self._autellix_window is not None:
                        self._autellix_window.drop_reserve()
                    restored = self._autellix_host.restore(req.rid)
                if restored:
                    ctl.emit("swap_in", rid=req.rid)
                else:
                    if not self.running_batch.reqs and ready_index == 0:
                        raise RuntimeError("insufficient GPU KV capacity to restore one request")
                    # Do not admit a lower-priority request past a failed
                    # restoration. A successfully prepared prefix can still run.
                    blocked.update(r.rid for r in ready[ready_index:])
                    break
            # Keep a promoted reserve pinned until native admission acquires
            # its own prefix lock; restoring another request must not evict it.
        held = [r for r in self.waiting_queue if (selected is not None and r.rid not in selected) or r.rid in blocked]
        self.waiting_queue = [r for r in self.waiting_queue if r not in held]
        def hold_released_reserves():
            if self._autellix_paper and self._autellix_window is not None:
                released = [r for r in self.waiting_queue if r.rid not in self._autellix_window.cohort]
                held.extend(released)
                self.waiting_queue = [r for r in self.waiting_queue if r not in released]
        try:
            hold_released_reserves()
            if self._autellix_paper and ready:
                self.running_batch.batch_is_full = False
            result = original_prefill(self)
            if result is None and slots and self.waiting_queue and self._autellix_reserve:
                # Held reserves are temporarily outside waiting_queue here.
                current_waiting = self.waiting_queue
                self.waiting_queue = current_waiting + held
                try:
                    release_reserves(self)
                finally:
                    self.waiting_queue = current_waiting
                if self._autellix_window is not None:
                    self._autellix_window.drop_reserve()
                hold_released_reserves()
                self.running_batch.batch_is_full = False
                result = original_prefill(self)
        finally:
            self.waiting_queue.extend(held)
        if self._autellix_paper and self._autellix_window is not None:
            admitted = {r.rid for r in self.running_batch.reqs}
            if result is not None:
                admitted.update(r.rid for r in result.reqs)
            ready_ids = {r.rid for r in ready}
            for i, rid in enumerate(self._autellix_window.cohort):
                if rid in ready_ids and rid not in admitted:
                    if i == 0:
                        raise RuntimeError("highest-priority request cannot fit the configured token/KV budget")
                    self._autellix_window.retain_prefix(i)
                    break
        if result is not None:
            for req in result.reqs:
                node = self._autellix_reserve.pop(req.rid, None)
                if node is not None:
                    self.tree_cache.dec_lock_ref(node)
                    ctl.emit("resume_cached", rid=req.rid)
                if self._autellix_host is not None:
                    self._autellix_host.discard(req.rid)
        return result

    def run(self, batch):
        rids = [req.rid for req in batch.reqs]
        self.autellix.begin(rids)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = original_run(self, batch)
        end.record()
        end.synchronize()
        self.autellix.executed(rids, start.elapsed_time(end) / 1000.)
        if self._autellix_window is not None:
            only_prefill = batch.forward_mode.is_extend() and not getattr(batch, "decoding_reqs", None)
            ctl = self.autellix
            if self._autellix_paper:
                ctl.emit("plan_step", requests=rids, decode=not only_prefill,
                         remaining=self._autellix_window.remaining)
            self._autellix_window.executed(only_prefill)
        return result

    def update_running(self, batch):
        if not self._autellix_paper:
            return original_update_running(self, batch)
        batch.filter_batch()
        if batch.is_empty():
            batch.batch_is_full = False
            return batch
        # Advance the admitted plan; never use native length-based retraction
        # to replace paper priorities in the middle of a scheduling window.
        from .pressure import make_decode_room
        make_decode_room(self, batch)
        batch.prepare_for_decode()
        return batch

    def process(self, batch, result, *a, **kw):
        reqs = list(batch.reqs)
        ret = original_result(self, batch, result, *a, **kw)
        for req in reqs:
            if req.finished() and req.rid in self.autellix.calls:
                metrics = self.autellix.finish(req.rid)
                rid = self._autellix_rids.pop(req.rid)
                self.autellix.table.save_metrics(rid, metrics)
        return ret

    def abort(self, request):
        queued = [r.rid for r in self.waiting_queue
                  if request.abort_all or r.rid.startswith(request.rid)]
        ret = original_abort(self, request)
        for internal in queued:
            if self._autellix_host is not None:
                self._autellix_host.discard(internal)
            node = self._autellix_reserve.pop(internal, None)
            if node is not None:
                self.tree_cache.dec_lock_ref(node)
            metrics = self.autellix.finish(internal, "cancelled")
            external = self._autellix_rids.pop(internal, None)
            if external:
                self.autellix.table.save_metrics(external, metrics)
        return ret

    def check_memory(self):
        # Native idle accounting assumes there are no protected request KV
        # blocks. Reserves are cache entries, so unpin them when the executor
        # becomes idle; the radix cache can still reuse them on next admission.
        for rid, node in self._autellix_reserve.items():
            self.tree_cache.dec_lock_ref(node)
            self.autellix.emit("release_reserve", rid=rid)
        self._autellix_reserve.clear()
        return original_check_memory(self)

    cls.__init__ = init
    cls._extend_requests_to_queue = extend
    cls._add_request_to_queue = add
    cls.get_new_batch_prefill = prefill
    cls.run_batch = run
    cls.process_batch_result = process
    cls.abort_request = abort
    cls.check_memory = check_memory
    if original_update_running is not None:
        cls.update_running_batch = update_running
