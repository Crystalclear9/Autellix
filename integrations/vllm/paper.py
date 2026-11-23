"""Global-priority batch admission, independent of vLLM's phase scheduling.

Only window boundaries select a new cohort. Other invocations materialize the
next execution step of that plan, including prepared-reserve replacement.
Native block management and sequence metadata remain the source of KV ownership.
"""
from autellix.runtime.window import SchedulingWindow


class PaperPlan:
    def __init__(self, scheduler, controller, capacity, keep_resident_kv=False, refill_idle_slots=False,
                 chunked_prefill=False, resident_reserve_only=False, recompute_on_swap_full=False,
                 resident_decode_headroom=False, prefix_cache=None):
        self.scheduler, self.controller = scheduler, controller
        self.window = SchedulingWindow(controller, capacity, refill_idle_slots=refill_idle_slots,
                                       resident_reserve_only=resident_reserve_only)
        self.keep_resident_kv = keep_resident_kv
        self.chunked_prefill = chunked_prefill
        self.recompute_on_swap_full = recompute_on_swap_full
        self.resident_decode_headroom = resident_decode_headroom
        self.prefix_cache = prefix_cache
        self.prepared = set()
        self.reserved_slots = set()

    def next_step(self):
        from vllm.core.scheduler import SchedulerOutputs, ScheduledSequenceGroup
        from vllm.core.interfaces import AllocStatus
        from vllm.sequence import SequenceStatus
        s, ctl, window = self.scheduler, self.controller, self.window
        incoming, outgoing, copies, ignored = [], [], [], []
        uncached_counts = {}

        def output(groups=()):
            groups = list(groups)
            # Attention kernels require prefill metadata before decode metadata;
            # this does not change admission priority or the selected batch.
            groups.sort(key=lambda g: not g.seq_group.is_prefill())
            s._autellix_execution = [g.seq_group.request_id for g in groups]
            s._autellix_num_prefill = sum(g.seq_group.is_prefill() for g in groups)
            s._autellix_is_prefill = s._autellix_num_prefill == len(groups)
            # vLLM's statistics consume logical progress, including cached tokens.
            num_tokens = sum(g.token_chunk_size for g in groups)
            uncached_tokens = sum(uncached_counts.get(g.seq_group.request_id, g.token_chunk_size)
                                  for g in groups) if uncached_counts else num_tokens
            ctl.emit("plan_step", requests=s._autellix_execution,
                     decode=not s._autellix_is_prefill, prefill_count=s._autellix_num_prefill,
                     remaining=window.remaining, tokens=uncached_tokens)
            ctl.begin(s._autellix_execution)
            return SchedulerOutputs(
                scheduled_seq_groups=groups,
                num_prefill_groups=s._autellix_num_prefill,
                num_batched_tokens=num_tokens,
                blocks_to_swap_in=incoming, blocks_to_swap_out=outgoing,
                blocks_to_copy=copies, ignored_seq_groups=ignored,
                num_lookahead_slots=0, running_queue_size=len(s.running), preempted=0)

        def continue_decode(running):
            groups = []
            for rid in window.cohort[:window.capacity]:
                group = running[rid]
                for seq in group.get_seqs():
                    copies.extend(s.block_manager.append_slots(seq, 0))
                group.init_multi_step(num_scheduler_steps=1)
                groups.append(ScheduledSequenceGroup(seq_group=group, token_chunk_size=1))
            ctl.emit("continue_plan", requests=[g.seq_group.request_id for g in groups])
            return output(groups)

        running = {g.request_id: g for g in s.running}

        def prepared_decode(cohort):
            return bool(cohort) and all(
                rid in self.reserved_slots and rid in self.prepared and rid in running
                and not running[rid].is_prefill() for rid in cohort)

        # A full prepared window can continue without scanning waiting and
        # swapped queues, whose size is independent of the active GPU batch.
        if (window.remaining and (not window.refill_idle_slots or len(window.active) == window.capacity)
                and prepared_decode(window.cohort)):
            return continue_decode(running)
        if window.remaining and not window.refill_idle_slots:
            # Eviction removes the ready and reserved flags. A missing request
            # that still has both flags finished and can leave its slot now.
            live = [rid for rid in window.cohort if rid in running]
            if (prepared_decode(live)
                    and all(rid in self.reserved_slots and rid in self.prepared for rid in window.cohort)):
                window.select(live, live)
                return continue_decode(running)

        by_id = {**running, **{g.request_id: g for g in s.waiting}, **{g.request_id: g for g in s.swapped}}
        boundary = window.remaining == 0 or not any(r in by_id for r in window.cohort)
        if boundary:
            self.reserved_slots.clear()
        chosen, resident = window.select(by_id, self.prepared & running.keys())
        self.prepared.intersection_update(by_id)

        def evict(group):
            if self.recompute_on_swap_full and not s.block_manager.can_swap_out(group):
                s._preempt_by_recompute(group)
                s.waiting.appendleft(group)
                ctl.emit("recompute_retract", rid=group.request_id, reason="host_kv_full")
            else:
                s._preempt_by_swap(group, outgoing)
                s.swapped.append(group)
                ctl.emit("swap_out", rid=group.request_id)
            s.running.remove(group)
            self.prepared.discard(group.request_id)
            self.reserved_slots.discard(group.request_id)

        def evict_for(index, fits):
            cohort = set(window.cohort)
            inactive = sorted((g for g in s.running if g.request_id not in cohort),
                              key=lambda g: ctl.key(g.request_id), reverse=True)
            suffix = [by_id[r] for r in reversed(window.cohort[index + 1:]) if r in running]
            # The allocator releases GPU block ownership immediately. Collect
            # all required victim mappings before the single transfer barrier.
            for victim in inactive + suffix:
                evict(victim)
                if fits():
                    break
            return bool(inactive or suffix)

        def allocation_status(group, lookahead):
            if group in s.waiting:
                return s.block_manager.can_allocate(group)
            if group in s.swapped:
                return s.block_manager.can_swap_in(group, lookahead)
            return (AllocStatus.OK if s.block_manager.can_append_slots(group, lookahead)
                    else AllocStatus.LATER)

        if not self.keep_resident_kv:
            # Flush outgoing copies before reusing their GPU blocks for swap-in.
            for group in list(s.running):
                if group.request_id not in window.cohort:
                    evict(group)
            if outgoing:
                return output()

        if not boundary and prepared_decode(window.cohort):
            return continue_decode(running)

        groups, prepared, tokens = [], [], 0

        def run_admitted():
            self.prepared.update(prepared)
            return output(groups)

        resident_decodes = set()
        defer_unreserved = False
        if self.chunked_prefill:
            from vllm.core.scheduler import SchedulingBudget
            budget = SchedulingBudget(s.scheduler_config.max_num_batched_tokens,
                                      s.scheduler_config.max_num_seqs)
            if self.resident_decode_headroom:
                resident_decodes = {rid for rid in window.active & running.keys()
                                    if not running[rid].is_prefill()}
                min_prefill = s.cache_config.block_size if s.cache_config.enable_prefix_caching else 1
                if budget.remaining_token_budget() < min_prefill + len(resident_decodes):
                    resident_decodes.clear()
                for rid in resident_decodes:
                    budget.add_num_batched_tokens(rid, 1)
        for index, rid in enumerate(window.cohort):
            group = by_id[rid]
            if defer_unreserved and rid not in resident_decodes:
                continue
            if rid not in chosen and rid in self.reserved_slots:
                continue
            prefill = group.is_prefill()
            executes = rid not in resident and not (rid in window.reserve and not prefill)
            count = (group.get_seqs()[0].data.get_num_uncomputed_tokens() if prefill else 1) if executes else 0
            partial = False
            cached = 0
            if self.chunked_prefill and prefill and count:
                seq = group.get_seqs()[0]
                if self.prefix_cache is None:
                    chunk = s._get_num_new_tokens(group, seq.status, True, budget)
                else:
                    cached = self.prefix_cache.reusable_tokens(seq)
                    available = budget.remaining_token_budget()
                    size = self.prefix_cache.block_size
                    aligned = (available // size) * size
                    chunk = count if count - cached <= available else (cached + aligned if aligned else 0)
                if chunk == 0:
                    if not groups:
                        raise RuntimeError("token budget must allow one prefix-cache block")
                    # This forward consumed the chunk budget. The remaining
                    # cohort can still fit in a later forward of this window.
                    if resident_decodes:
                        defer_unreserved = True
                        continue
                    break
                partial = chunk < count
                count = chunk
            uncached = count - cached
            if (resident_decodes and rid not in resident_decodes
                    and uncached > budget.remaining_token_budget()):
                defer_unreserved = True
                continue
            if tokens + uncached > s.scheduler_config.max_num_batched_tokens:
                if groups and not self.chunked_prefill:
                    window.retain_prefix(index)
                break
            lookahead = window.remaining if rid not in self.reserved_slots and not partial else 0
            status = allocation_status(group, lookahead)
            if status != AllocStatus.OK:
                # Send the current transfer with the admitted prefix and
                # retry this request once the worker has completed it.
                if status == AllocStatus.LATER and (incoming or evict_for(
                        index, lambda: allocation_status(group, lookahead) == AllocStatus.OK)):
                    return run_admitted()
                if status == AllocStatus.NEVER:
                    for seq in group.get_seqs():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    s.free_seq(group.get_seqs()[0])
                    for queue in (s.waiting, s.running, s.swapped):
                        if group in queue:
                            queue.remove(group)
                    ignored.append(group)
                if groups:
                    window.retain_prefix(index)
                break
            if group in s.waiting:
                s.waiting.remove(group)
                s._allocate_and_set_running(group)
                s.running.append(group)
            elif group in s.swapped:
                s._swap_in(group, incoming)
                s.swapped.remove(group)
                s.running.append(group)
                ctl.emit("resume", rid=rid)
                if not prefill:
                    self.prepared.add(rid)
            # Reserve future KV capacity once per window. Subsequent append
            # calls commit actual token IDs / prefix-cache bookkeeping only.
            if not s.block_manager.can_append_slots(group, lookahead):
                if incoming or evict_for(index, lambda: s.block_manager.can_append_slots(group, lookahead)):
                    return run_admitted()
                if not groups:
                    raise RuntimeError("insufficient KV capacity for one scheduling window; reduce schedule_interval")
                window.retain_prefix(index)
                break
            for seq in group.get_seqs():
                copies.extend(s.block_manager.append_slots(seq, lookahead))
            if not partial:
                self.reserved_slots.add(rid)
            group.init_multi_step(num_scheduler_steps=1)
            if rid in resident or (rid in window.reserve and not prefill):
                continue
            groups.append(ScheduledSequenceGroup(seq_group=group, token_chunk_size=count))
            if cached:
                uncached_counts[rid] = uncached
            if not partial:
                prepared.append(rid)
            tokens += uncached
            if self.chunked_prefill and rid not in resident_decodes:
                budget.add_num_batched_tokens(rid, uncached)
            if rid in window.reserve:
                ctl.emit("reserve", rid=rid)
        if not groups and not incoming and not ignored and by_id:
            raise RuntimeError("highest-priority request cannot fit the configured token/KV budget")
        return run_admitted()
