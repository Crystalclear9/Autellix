"""Global-priority batch admission, independent of vLLM's phase scheduling.

Only window boundaries select a new cohort. Other invocations materialize the
next execution step of that plan, including prepared-reserve replacement.
Native block management and sequence metadata remain the source of KV ownership.
"""
from autellix.runtime.window import SchedulingWindow


class PaperPlan:
    def __init__(self, scheduler, controller, capacity, keep_resident_kv=False, refill_idle_slots=False,
                 chunked_prefill=False):
        self.scheduler, self.controller = scheduler, controller
        self.window = SchedulingWindow(controller, capacity, refill_idle_slots=refill_idle_slots)
        self.keep_resident_kv = keep_resident_kv
        self.chunked_prefill = chunked_prefill
        self.prepared = set()
        self.reserved_slots = set()

    def next_step(self):
        from vllm.core.scheduler import SchedulerOutputs, ScheduledSequenceGroup
        from vllm.core.interfaces import AllocStatus
        from vllm.sequence import SequenceStatus
        s, ctl, window = self.scheduler, self.controller, self.window
        incoming, outgoing, copies, ignored = [], [], [], []

        def output(groups=()):
            groups = list(groups)
            # Attention kernels require prefill metadata before decode metadata;
            # this does not change admission priority or the selected batch.
            groups.sort(key=lambda g: not g.seq_group.is_prefill())
            s._autellix_execution = [g.seq_group.request_id for g in groups]
            s._autellix_is_prefill = not any(not g.seq_group.is_prefill() for g in groups)
            num_tokens = sum(g.token_chunk_size for g in groups)
            ctl.emit("plan_step", requests=s._autellix_execution,
                     decode=not s._autellix_is_prefill, remaining=window.remaining, tokens=num_tokens)
            ctl.begin(s._autellix_execution)
            return SchedulerOutputs(
                scheduled_seq_groups=groups,
                num_prefill_groups=sum(g.seq_group.is_prefill() for g in groups),
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

        def prepared_decode():
            return bool(window.cohort) and all(
                rid in self.reserved_slots and rid in self.prepared and rid in running
                and not running[rid].is_prefill() for rid in window.cohort)

        # A full prepared window can continue without scanning waiting and
        # swapped queues, whose size is independent of the active GPU batch.
        if (window.remaining and (not window.refill_idle_slots or len(window.active) == window.capacity)
                and prepared_decode()):
            return continue_decode(running)

        by_id = {**running, **{g.request_id: g for g in s.waiting}, **{g.request_id: g for g in s.swapped}}
        boundary = window.remaining == 0 or not any(r in by_id for r in window.cohort)
        if boundary:
            self.reserved_slots.clear()
        chosen, resident = window.select(by_id, self.prepared & running.keys())
        self.prepared.intersection_update(by_id)

        def evict(group):
            s._preempt_by_swap(group, outgoing)
            s.running.remove(group)
            s.swapped.append(group)
            self.prepared.discard(group.request_id)
            self.reserved_slots.discard(group.request_id)
            ctl.emit("swap_out", rid=group.request_id)

        def evict_for(index, fits):
            if incoming:
                return False
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

        if not boundary and prepared_decode():
            return continue_decode(running)

        groups, prepared, tokens = [], [], 0
        if self.chunked_prefill:
            from vllm.core.scheduler import SchedulingBudget
            budget = SchedulingBudget(s.scheduler_config.max_num_batched_tokens,
                                      s.scheduler_config.max_num_seqs)
        for index, rid in enumerate(window.cohort):
            group = by_id[rid]
            if rid not in chosen and rid in self.reserved_slots:
                continue
            prefill = group.is_prefill()
            executes = rid not in resident and not (rid in window.reserve and not prefill)
            count = (group.get_seqs()[0].data.get_num_uncomputed_tokens() if prefill else 1) if executes else 0
            partial = False
            if self.chunked_prefill and prefill and count:
                chunk = s._get_num_new_tokens(group, group.get_seqs()[0].status, True, budget)
                if chunk == 0:
                    if not groups:
                        raise RuntimeError("token budget must allow one prefix-cache block")
                    window.retain_prefix(index)
                    break
                partial = chunk < count
                count = chunk
            if tokens + count > s.scheduler_config.max_num_batched_tokens:
                if groups:
                    window.retain_prefix(index)
                break  # Algorithm 1: stop at the first request that cannot fit.
            lookahead = window.remaining if rid not in self.reserved_slots and not partial else 0
            status = allocation_status(group, lookahead)
            if status != AllocStatus.OK:
                # A lower-priority resident must never block a higher-priority
                # arrival or swapped request. Retry after the transfer barrier.
                if status == AllocStatus.LATER and evict_for(
                        index, lambda: allocation_status(group, lookahead) == AllocStatus.OK):
                    return output()
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
                if evict_for(index, lambda: s.block_manager.can_append_slots(group, lookahead)):
                    return output()
                if incoming:
                    return output()  # retry capacity after the incoming barrier
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
            if not partial:
                prepared.append(rid)
            tokens += count
            if self.chunked_prefill:
                budget.add_num_batched_tokens(rid, count)
            if rid in window.reserve:
                ctl.emit("reserve", rid=rid)
        if not groups and not incoming and not ignored and by_id:
            raise RuntimeError("highest-priority request cannot fit the configured token/KV budget")
        self.prepared.update(prepared)
        return output(groups)
