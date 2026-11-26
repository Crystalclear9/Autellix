from __future__ import annotations

import base64
import json
from time import perf_counter


def encode_request(pid, rid):
    return "autellix_" + base64.urlsafe_b64encode(json.dumps([pid, rid]).encode()).decode()


def decode_request(value):
    if not value.startswith("autellix_"):
        raise ValueError("request lacks Autellix program metadata")
    pid, rid = json.loads(base64.urlsafe_b64decode(value[9:]))
    return pid, rid


def run_scheduler_process(*args, autellix_options, **kwargs):
    """Spawn-safe process target: install hooks inside the actual GPU process."""
    if autellix_options.get("legacy_gpu"):
        from .legacy_gpu import install_native_ops
        install_native_ops(prefix_attention=autellix_options.get("prefix_attention", True),
                           attention_score_elements=autellix_options.get("attention_score_elements", 2**26),
                           mlp_chunk_size=autellix_options.get("mlp_chunk_size", 0))
    if autellix_options.get("pipeline"):
        from .pipeline import install_pipeline_embeddings
        install_pipeline_embeddings()
    import sglang.srt.managers.scheduler as module
    if autellix_options["config"].get("implementation", "paper") == "paper":
        from sglang.srt.managers import schedule_policy
        # Admission reserves the current decode window. Generation still uses
        # each request's original max_new_tokens and normal stopping rules.
        schedule_policy.CLIP_MAX_NEW_TOKENS_ESTIMATION = autellix_options["config"]["schedule_interval"]
    install(module.Scheduler, autellix_options)
    from .pipeline import install as install_pipeline
    install_pipeline(module.Scheduler, timing_metadata=autellix_options.get("pipeline_timing_metadata", True))
    from .profiling import install as install_profile
    install_profile(module.Scheduler)
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
    measure_timing = options.get("measure_timing", False)
    defer_model_timing = options.get("defer_model_timing", True)
    min_decode_steps = options.get("min_decode_steps", 0)

    def init(self, *a, **kw):
        original_init(self, *a, **kw)
        self.pp_size = getattr(self, "pp_size", 1)
        if self.pp_size > 1:
            from .pipeline import configure
            configure(self, options)
        else:
            self._autellix_model_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            self._autellix_pending_timing = None
            self.autellix = RuntimeScheduler(ProgramTable(options["table_path"]),
                                            PolicyConfig(**options["config"]), options.get("trace"),
                                            publish_activity_at_window=options.get("publish_activity_at_window", False),
                                            publish_activity=options.get("publish_activity", True),
                                            min_decode_steps=min_decode_steps)
        self._autellix_steps = 0
        self._autellix_paper = self.autellix.config.implementation == "paper"
        if self._autellix_paper:
            # Mixed execution prevents native prefill preference from excluding
            # higher-priority decodes already admitted by the global plan.
            self.is_mixed_chunk = options.get("mix_prefill_decode", True)
            self.new_token_ratio = self.init_new_token_ratio = 1.0
        self._autellix_rids = {}
        self._autellix_reserve = {}
        self._autellix_window = None
        if options.get("capacity") and self.pp_size == 1:
            from autellix.runtime.window import SchedulingWindow
            self._autellix_window = SchedulingWindow(self.autellix, options["capacity"],
                                                      refill_idle_slots=options.get("refill_idle_slots", False),
                                                      resident_reserve_only=options.get("resident_reserve_only", False))
        if options.get("host_bytes"):
            from .swap import HostKV
            self._autellix_host = HostKV(self.token_to_kv_pool_allocator, self.tree_cache, options["host_bytes"],
                                         recompute_on_full=options.get("host_kv_recompute", False),
                                         fused=options.get("fused_swap", False),
                                         observe=self.autellix.emit if measure_timing else None)
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
        restored_nodes = []
        other_reserves = set()
        if self.pp_size > 1:
            other_reserves = {rid for reserves in self._autellix_reserves
                              if reserves is not self._autellix_reserve for rid in reserves}
        held = []
        if other_reserves:
            held = [req for req in self.waiting_queue if req.rid in other_reserves]
            self.waiting_queue = [req for req in self.waiting_queue if req.rid not in other_reserves]
        try:
            return prefill_plan(self, restored_nodes)
        finally:
            self.waiting_queue.extend(held)
            # Native admission has acquired its own locks for accepted requests.
            # Others retain their host backup and must remain evictable on retry.
            for _, node in reversed(restored_nodes):
                self.tree_cache.dec_lock_ref(node)

    def prefill_plan(self, restored_nodes):
        ctl = self.autellix
        window = self._autellix_window
        # SGLang owns a partial prefill outside the waiting and running queues.
        # Keep its frozen cohort until native admission finishes the chunk.
        if getattr(self, "chunked_req", None) is not None:
            waiting = self.waiting_queue
            eligible = []
            if options.get("fill_chunk_tail") and self._autellix_paper and window is not None:
                req = self.chunked_req
                remaining = len(req.origin_input_ids) + len(req.output_ids) - len(req.prefix_indices)
                mixed_decode = len(self.running_batch.reqs) if self.is_mixed_chunk else 0
                if remaining < self.chunked_prefill_size - mixed_decode:
                    queued = {req.rid: req for req in waiting}
                    if window.refill_idle_slots and window.remaining > 0 and req.rid in window.active:
                        running = {r.rid for r in self.running_batch.reqs} | {req.rid}
                        window.select(running.union(queued), running.union(self._autellix_reserve))
                    host = self._autellix_host
                    for rid in window.cohort:
                        if rid not in window.active or rid not in queued:
                            continue
                        if rid in self._autellix_reserve or (host is not None and rid in host.saved):
                            break
                        eligible.append(queued[rid])
            considered = {req.rid for req in eligible}
            self.waiting_queue = eligible
            try:
                ctl.emit("continue_prefill", rid=self.chunked_req.rid)
                return original_prefill(self)
            finally:
                admitted = considered.difference(req.rid for req in self.waiting_queue)
                self.waiting_queue = [req for req in waiting if req.rid not in admitted] if admitted else waiting
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
        if window is not None:
            running = {r.rid for r in batch.reqs}
            if self._autellix_paper and window.can_continue_decode(running, self._autellix_reserve):
                ctl.emit("continue_plan", requests=[r.rid for r in batch.reqs])
                return None
            old_remaining = window.remaining
            selected, _ = window.select(running.union(r.rid for r in self.waiting_queue),
                                        running.union(self._autellix_reserve))
            departing = [rid for rid in self._autellix_reserve if rid not in window.cohort]
            if departing:
                # A rotation leaves the prefix resident and evictable in radix.
                # Use the same residency policy as other paused requests.
                release_reserves(self, departing, preserve=not options.get("cache_paused_kv"))
            if self._autellix_paper and old_remaining > 0 and selected == running:
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
                    pending_reqs = [r for r in self.waiting_queue if r.rid in keep]
                    if pending_reqs:
                        # Native admission accounts running requests before new
                        # prefills. Release the lower-priority running suffix so
                        # it cannot consume the space needed by a higher arrival.
                        # Re-admit that suffix in the same global FIFO order.
                        cutoff = min(order[r.rid] for r in pending_reqs)
                        lower_running = [r.rid for r in batch.reqs if r.rid in keep and order[r.rid] >= cutoff]
                        from .admission import can_keep_running
                        if (lower_running and options.get("retain_running", True)
                                and can_keep_running(self, pending_reqs, keep, order=order)):
                            ctl.emit("retain_running", requests=lower_running,
                                     pending=[r.rid for r in pending_reqs])
                        else:
                            keep = {rid for rid in keep if order[rid] < cutoff}
                victims = [i for i, r in enumerate(batch.reqs) if r.rid not in keep]
                # Use the pinned engine's native retraction ownership rules.
                # Generated output_ids are preserved by reset_for_retract().
                lengths = batch.seq_lens.cpu().tolist() if victims else ()
                retracted = []
                for i in victims:
                    req = batch.reqs[i]
                    reserve = (len(self._autellix_reserve) < ctl.config.overprovision and
                               (self._autellix_window is None or req.rid in self._autellix_window.reserve))
                    if reserve or options.get("cache_paused_kv"):
                        # Transfer ownership to radix. A reserve holds a lock;
                        # other paused prefixes are reclaimed under pressure.
                        req.fill_ids = (req.origin_input_ids + req.output_ids)[:lengths[i]]
                        batch.tree_cache.cache_unfinished_req(req)
                        batch.req_to_token_pool.free(req.req_pool_idx)
                        if reserve:
                            req._autellix_reserved_tokens = list(req.fill_ids)
                            self._autellix_reserve[req.rid] = req.last_node
                            ctl.emit("reserve", rid=req.rid)
                        else:
                            batch.tree_cache.dec_lock_ref(req.last_node)
                            ctl.emit("pause_cached", rid=req.rid, tokens=lengths[i])
                    else:
                        if self._autellix_host is not None:
                            all_indices = batch.req_to_token_pool.req_to_token[req.req_pool_idx, :lengths[i]]
                            tokens = (req.origin_input_ids + req.output_ids)[:lengths[i]]
                            saved = self._autellix_host.save(req.rid, tokens, all_indices)
                            ctl.emit("recompute_retract" if saved is False else "swap_out", rid=req.rid, tokens=len(tokens))
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
        order = ({rid: i for i, rid in enumerate(self._autellix_window.cohort)}
                 if selected is not None and self._autellix_paper else None)
        ready = sorted((r for r in self.waiting_queue if selected is None or r.rid in selected),
                       key=lambda r: ctl.key(r.rid) if order is None else order[r.rid])[:slots]
        if self._autellix_paper and self._autellix_window is not None and not ready:
            # Decode continuation: no native prefill admission / priority pass.
            return None
        host = self._autellix_host
        last_restore = -1
        if host is not None:
            last_restore = max((i for i, req in enumerate(ready)
                                if req.rid in host.saved or req.rid in self._autellix_reserve), default=-1)
        blocked = set()
        for ready_index, req in enumerate(ready):
            if (self._autellix_paper and self._autellix_window is not None
                    and req.rid not in self._autellix_window.cohort):
                blocked.add(req.rid)
                continue
            if host is not None and req.rid in host.saved:
                restored = host.restore(req.rid)
                if not restored and self._autellix_reserve:
                    if self._autellix_window is not None:
                        order = {rid: i for i, rid in enumerate(self._autellix_window.cohort)}
                        inactive = [rid for rid in self._autellix_reserve if order[rid] > order[req.rid]]
                    else:
                        inactive = list(self._autellix_reserve)
                    release_reserves(self, inactive)
                    if self._autellix_window is not None:
                        self._autellix_window.drop_reserve()
                    restored = host.restore(req.rid)
                if restored:
                    # A later restoration can evict radix entries before native
                    # admission runs. Protect every restored higher-priority
                    # prefix until that admission has taken ownership.
                    tokens, _ = host.saved[req.rid]
                    matched = self.tree_cache.match_prefix(tokens)
                    if len(matched.device_indices) != len(tokens):
                        raise RuntimeError("restored KV prefix is incomplete before admission")
                    self.tree_cache.inc_lock_ref(matched.last_device_node)
                    restored_nodes.append((req.rid, matched.last_device_node))
                    ctl.emit("swap_in", rid=req.rid)
                else:
                    from .pressure import other_pipeline_requests
                    if (not self.running_batch.reqs and ready_index == 0
                            and not other_pipeline_requests(self, self.running_batch)):
                        raise RuntimeError("insufficient GPU KV capacity to restore one request")
                    # Do not admit a lower-priority request past a failed
                    # restoration. A successfully prepared prefix can still run.
                    blocked.update(r.rid for r in ready[ready_index:])
                    break
            elif ready_index < last_restore:
                # A new call can already have a long cached prefix. Preserve it
                # while lower-priority host copies are restored below.
                req.init_next_round_input(self.tree_cache)
                self.tree_cache.inc_lock_ref(req.last_node)
                restored_nodes.append((req.rid, req.last_node))
            # Keep a promoted reserve pinned until native admission acquires
            # its own prefix lock; restoring another request must not evict it.
        held = [r for r in self.waiting_queue if (selected is not None and r.rid not in selected) or r.rid in blocked]
        held_ids = {r.rid for r in held}
        self.waiting_queue = [r for r in self.waiting_queue if r.rid not in held_ids]
        def release_prepared(rid):
            for entry in restored_nodes[:]:
                if entry[0] == rid:
                    self.tree_cache.dec_lock_ref(entry[1])
                    restored_nodes.remove(entry)
        def hold_released_reserves():
            if self._autellix_paper and self._autellix_window is not None:
                released = [r for r in self.waiting_queue if r.rid not in self._autellix_window.cohort]
                for req in released:
                    release_prepared(req.rid)
                held.extend(released)
                released_ids = {r.rid for r in released}
                self.waiting_queue = [r for r in self.waiting_queue if r.rid not in released_ids]
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
            # Restored lower prefixes must not occupy space needed for a new
            # higher call's uncached tokens. Release the lower suffix and let
            # native admission evict it, keeping its host backup for later.
            while result is None and slots and len(self.waiting_queue) > 1 and restored_nodes:
                victim = self.waiting_queue.pop()
                held.append(victim)
                blocked.add(victim.rid)
                release_prepared(victim.rid)
                ctl.emit("defer_prefill", rid=victim.rid, reason="higher_prefill_capacity")
                self.running_batch.batch_is_full = False
                result = original_prefill(self)
        finally:
            self.waiting_queue.extend(held)
        # Native NO_TOKEN/slot exhaustion sets batch_is_full; hitting only
        # the per-forward prefill budget can leave a batch and pending work.
        # Keep those requests in the cohort for the next iteration.
        forward_limited = (self.pp_size == 1 and result is not None
                           and not self.running_batch.batch_is_full and not blocked)
        if (self._autellix_paper and self._autellix_window is not None and not forward_limited
                and getattr(self, "chunked_req", None) is None):
            admitted = {r.rid for r in self.running_batch.reqs}
            if result is not None:
                admitted.update(r.rid for r in result.reqs)
            ready_ids = {r.rid for r in ready}
            for i, rid in enumerate(self._autellix_window.cohort):
                if rid in ready_ids and rid not in admitted:
                    if i == 0:
                        from .pressure import other_pipeline_requests
                        if not other_pipeline_requests(self, self.running_batch):
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
            if options.get("decode_cached_resumes") and self._autellix_paper and self.pp_size == 1:
                use_decode_for_cached_resumes(self, result)
        return result

    def use_decode_for_cached_resumes(self, batch):
        # Native admission has already allocated one KV slot per query and
        # populated the request-to-token table. A single uncached token per
        # resumed call uses the same positions and KV writes as a decode step.
        if (getattr(self, "chunked_req", None) is not None or self.running_batch.reqs
                or batch.return_logprob or batch.has_grammar or batch.return_hidden_states
                or batch.model_config.is_encoder_decoder or batch.input_embeds is not None
                or not batch.sampling_info.is_all_greedy or any(length != 1 for length in batch.extend_lens)):
            return
        continuing = len(batch.decoding_reqs or ())
        resumed = batch.reqs[:len(batch.reqs) - continuing]
        if not resumed or any(not req.output_ids or req.multimodal_inputs is not None for req in resumed):
            return
        batch.forward_mode = type(batch.forward_mode).DECODE
        self.running_batch = batch
        self.autellix.emit("decode_cached_resumes", requests=[req.rid for req in resumed])

    def account_model_time(self, rids, prefill_count, phase, *, synchronize):
        start, end = self._autellix_model_events
        if synchronize or not end.query():
            end.synchronize()
        seconds = start.elapsed_time(end) / 1000.
        self.autellix.executed(rids, seconds, prefill_count=prefill_count)
        if phase is not None:
            self.autellix.emit("model_phase_time", seconds=seconds, requests=rids, **phase)

    def run(self, batch):
        rids = [req.rid for req in batch.reqs]
        input_tokens = batch.input_ids.numel()
        if self.pp_size > 1:
            self._autellix_batches[self._autellix_microbatch] = batch
            self.autellix.begin(rids, active_rids=self._autellix_inflight)
            self._autellix_inflight.update(rids)
            result = original_run(self, batch)
        else:
            self.autellix.begin(rids)
            prefill_count = None
            phase = None
            if measure_timing or min_decode_steps:
                decoding = (len(getattr(batch, "decoding_reqs", None) or ())
                            if batch.forward_mode.is_extend() else len(rids))
                prefill_count = len(rids) - decoding
                if measure_timing:
                    prefill = (batch.extend_lens or ())[:prefill_count]
                    phase = dict(phase="mixed" if prefill_count and decoding else "prefill" if prefill_count else "decode",
                                 input_tokens=input_tokens, prefill_tokens=sum(prefill),
                                 resumed_prefill_tokens=sum(n for req, n in zip(batch.reqs, prefill) if req.output_ids))
            start, end = self._autellix_model_events
            start.record()
            result = original_run(self, batch)
            end.record()
            if defer_model_timing:
                # The non-overlap loop processes this result before its next
                # scheduling decision. Native token readback usually completes
                # the events without a separate blocking wait.
                self._autellix_pending_timing = (rids, prefill_count if min_decode_steps else None, phase)
            else:
                account_model_time(self, rids, prefill_count if min_decode_steps else None, phase, synchronize=True)
        if self._autellix_window is not None:
            only_prefill = batch.forward_mode.is_extend() and not getattr(batch, "decoding_reqs", None)
            ctl = self.autellix
            if self._autellix_paper:
                ctl.emit("plan_step", requests=rids, decode=not only_prefill,
                         remaining=self._autellix_window.remaining, input_tokens=input_tokens)
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
        if batch.is_empty():
            return batch
        batch.prepare_for_decode()
        return batch

    def process(self, batch, result, *a, **kw):
        reqs = list(batch.reqs)
        if self.pp_size > 1:
            seconds = self._autellix_tape.value("gpu_seconds", lambda: self._autellix_completed_seconds)
            self.autellix.executed([req.rid for req in reqs], seconds)
            self._autellix_inflight.difference_update(req.rid for req in reqs)
        ret = original_result(self, batch, result, *a, **kw)
        if self.pp_size == 1 and self._autellix_pending_timing is not None:
            rids, prefill_count, phase = self._autellix_pending_timing
            self._autellix_pending_timing = None
            account_model_time(self, rids, prefill_count, phase, synchronize=False)
        for req in reqs:
            if req.finished() and req.rid in self.autellix.calls:
                reason = "cancelled" if getattr(req, "to_abort", False) else "finished"
                metrics = self.autellix.finish(req.rid, reason)
                rid = self._autellix_rids.pop(req.rid)
                self.autellix.table.save_metrics(rid, metrics)
        return ret

    def abort(self, request):
        queued = [r.rid for r in self.waiting_queue
                  if request.abort_all or r.rid.startswith(request.rid)]
        ret = original_abort(self, request)
        if self.pp_size > 1:
            from .pipeline import abort_inflight
            abort_inflight(self, request)
        for internal in queued:
            if self._autellix_host is not None:
                self._autellix_host.discard(internal)
            pools = self._autellix_reserves if self.pp_size > 1 else [self._autellix_reserve]
            for reserves in pools:
                node = reserves.pop(internal, None)
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
        pools = self._autellix_reserves if self.pp_size > 1 else [self._autellix_reserve]
        for reserves in pools:
            for rid, node in reserves.items():
                self.tree_cache.dec_lock_ref(node)
                if self.pp_size == 1:
                    self.autellix.emit("release_reserve", rid=rid)
            reserves.clear()
        return original_check_memory(self)

    cls.__init__ = init
    cls._extend_requests_to_queue = extend
    cls._add_request_to_queue = add
    cls.get_new_batch_prefill = prefill
    cls.run_batch = run
    cls.process_batch_result = process
    cls.abort_request = abort
    cls.check_memory = check_memory
    if measure_timing:
        original_next = cls.get_next_batch_to_run

        def timed_next(self):
            host = self._autellix_host
            before = host.transfer_seconds if host is not None else 0.
            started = perf_counter()
            batch = original_next(self)
            if batch is not None or self.waiting_queue:
                swap_seconds = host.transfer_seconds - before if host is not None else 0.
                self.autellix.emit("schedule_time", seconds=max(0., perf_counter() - started - swap_seconds))
            return batch

        cls.get_next_batch_to_run = timed_next
    if original_update_running is not None:
        cls.update_running_batch = update_running
