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

    def init(self, *a, **kw):
        original_init(self, *a, **kw)
        self.autellix = RuntimeScheduler(ProgramTable(options["table_path"]),
                                        PolicyConfig(**options["config"]), options.get("trace"))
        self._autellix_steps = 0
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
            selected, _ = self._autellix_window.select(
                [r.rid for r in batch.reqs + self.waiting_queue],
                [r.rid for r in batch.reqs] + list(self._autellix_reserve))
            for rid in list(self._autellix_reserve):
                if rid not in self._autellix_window.cohort:
                    self.tree_cache.dec_lock_ref(self._autellix_reserve.pop(rid))
        if selected is not None or (self._autellix_steps - 1) % ctl.config.schedule_interval == 0:
            if selected is None:
                ctl.refresh()
            batch = self.running_batch
            if batch.reqs and (self.waiting_queue or selected is not None):
                ranked = sorted(batch.reqs + self.waiting_queue, key=lambda r: ctl.key(r.rid))
                limit = self.server_args.max_running_requests
                keep = selected if selected is not None else {r.rid for r in ranked[:limit]}
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
        blocked = set()
        for req in ready:
            if self._autellix_host is not None and req.rid in self._autellix_host.saved:
                restored = self._autellix_host.restore(req.rid)
                if not restored and self._autellix_reserve:
                    for node in self._autellix_reserve.values():
                        self.tree_cache.dec_lock_ref(node)
                    self._autellix_reserve.clear()
                    if self._autellix_window is not None:
                        self._autellix_window.drop_reserve()
                    restored = self._autellix_host.restore(req.rid)
                if restored:
                    ctl.emit("swap_in", rid=req.rid)
                else:
                    if not self.running_batch.reqs:
                        raise RuntimeError("insufficient GPU KV capacity to restore one request")
                    blocked.add(req.rid)
            node = self._autellix_reserve.pop(req.rid, None)
            if node is not None:
                self.tree_cache.dec_lock_ref(node)
                ctl.emit("resume_cached", rid=req.rid)
        held = [r for r in self.waiting_queue if (selected is not None and r.rid not in selected) or r.rid in blocked]
        self.waiting_queue = [r for r in self.waiting_queue if r not in held]
        try:
            result = original_prefill(self)
            if result is None and slots and self.waiting_queue and self._autellix_reserve:
                for node in self._autellix_reserve.values():
                    self.tree_cache.dec_lock_ref(node)
                self._autellix_reserve.clear()
                if self._autellix_window is not None:
                    self._autellix_window.drop_reserve()
                self.running_batch.batch_is_full = False
                result = original_prefill(self)
        finally:
            self.waiting_queue.extend(held)
        if result is not None and self._autellix_host is not None:
            for req in result.reqs:
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
            self._autellix_window.executed(batch.forward_mode.is_extend())
        return result

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
