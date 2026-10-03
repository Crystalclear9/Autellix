"""Global-priority batch admission, independent of vLLM's phase scheduling.

Only window boundaries select a new cohort. Other invocations materialize the
next execution step of that plan, including prepared-reserve replacement.
Native block management and sequence metadata remain the source of KV ownership.
"""
from autellix.runtime.window import SchedulingWindow


class PaperPlan:
    def __init__(self, scheduler, controller, capacity):
        self.scheduler, self.controller = scheduler, controller
        self.window = SchedulingWindow(controller, capacity)
        self.prepared = set()
        self.reserved_slots = set()

    def next_step(self):
        from vllm.core.scheduler import SchedulerOutputs, ScheduledSequenceGroup
        from vllm.core.interfaces import AllocStatus
        from vllm.sequence import SequenceStatus
        s, ctl, window = self.scheduler, self.controller, self.window
        available = list(s.running) + list(s.waiting) + list(s.swapped)
        by_id = {g.request_id: g for g in available}
        boundary = window.remaining == 0 or not any(r in by_id for r in window.cohort)
        if boundary:
            self.reserved_slots.clear()
        chosen, resident = window.select(by_id, self.prepared & {g.request_id for g in s.running})
        self.prepared.intersection_update(by_id)
        incoming, outgoing, copies, ignored = [], [], [], []

        def output(groups=()):
            groups = list(groups)
            # Attention kernels require prefill metadata before decode metadata;
            # this does not change admission priority or the selected batch.
            groups.sort(key=lambda g: not g.seq_group.is_prefill())
            s._autellix_execution = [g.seq_group.request_id for g in groups]
            s._autellix_is_prefill = not any(not g.seq_group.is_prefill() for g in groups)
            ctl.emit("plan_step", requests=s._autellix_execution,
                     decode=not s._autellix_is_prefill, remaining=window.remaining)
            ctl.begin(s._autellix_execution)
            return SchedulerOutputs(
                scheduled_seq_groups=groups,
                num_prefill_groups=sum(g.seq_group.is_prefill() for g in groups),
                num_batched_tokens=sum(g.token_chunk_size for g in groups),
                blocks_to_swap_in=incoming, blocks_to_swap_out=outgoing,
                blocks_to_copy=copies, ignored_seq_groups=ignored,
                num_lookahead_slots=0, running_queue_size=len(s.running), preempted=0)

        def evict(group):
            s._preempt_by_swap(group, outgoing)
            s.running.remove(group)
            s.swapped.append(group)
            self.prepared.discard(group.request_id)
            self.reserved_slots.discard(group.request_id)
            ctl.emit("swap_out", rid=group.request_id)

        # Flush outgoing copies before reusing their GPU blocks for swap-in.
        for group in list(s.running):
            if group.request_id not in window.cohort:
                evict(group)
        if outgoing:
            return output()

        if (not boundary and window.cohort and
                all(r in self.prepared and r in self.reserved_slots and by_id[r] in s.running
                    and not by_id[r].is_prefill() for r in window.cohort)):
            # No admission, fit checks, victim selection or native scheduling
            # inside a prepared decode window. Only commit the latest tokens to
            # the preallocated block tables and materialize model metadata.
            groups = []
            for rid in window.cohort:
                if rid not in window.active:
                    continue
                group = by_id[rid]
                for seq in group.get_seqs():
                    copies.extend(s.block_manager.append_slots(seq, 0))
                group.init_multi_step(num_scheduler_steps=1)
                groups.append(ScheduledSequenceGroup(seq_group=group, token_chunk_size=1))
            ctl.emit("continue_plan", requests=[g.seq_group.request_id for g in groups])
            return output(groups)

        groups, tokens = [], 0
        for rid in window.cohort:
            group = by_id[rid]
            if rid not in chosen and rid in self.reserved_slots:
                continue
            prefill = group.is_prefill()
            count = group.get_seqs()[0].data.get_num_uncomputed_tokens() if prefill else 1
            if tokens + count > s.scheduler_config.max_num_batched_tokens:
                if groups:
                    window.retain_prefix(window.cohort.index(rid))
                break  # Algorithm 1: stop at the first request that cannot fit.
            lookahead = max(0, window.remaining) if rid not in self.reserved_slots else 0
            if group in s.waiting:
                status = s.block_manager.can_allocate(group)
            elif group in s.swapped:
                status = s.block_manager.can_swap_in(group, lookahead)
            else:
                status = (AllocStatus.OK if s.block_manager.can_append_slots(group, lookahead)
                          else AllocStatus.LATER)
            if status != AllocStatus.OK:
                # A lower-priority resident must never block a higher-priority
                # arrival or swapped request. Retry after the transfer barrier.
                victims = [by_id[r] for r in window.cohort[window.cohort.index(rid) + 1:]
                           if by_id[r] in s.running]
                if victims and not incoming:
                    evict(victims[-1])
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
                    window.retain_prefix(window.cohort.index(rid))
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
                victims = [by_id[r] for r in window.cohort[window.cohort.index(rid) + 1:]
                           if by_id[r] in s.running]
                if victims and not incoming:
                    evict(victims[-1])
                    return output()
                if incoming:
                    return output()  # retry capacity after the incoming barrier
                if not groups:
                    raise RuntimeError("insufficient KV capacity for one scheduling window; reduce schedule_interval")
                window.retain_prefix(window.cohort.index(rid))
                break
            for seq in group.get_seqs():
                copies.extend(s.block_manager.append_slots(seq, lookahead))
            self.reserved_slots.add(rid)
            group.init_multi_step(num_scheduler_steps=1)
            if rid in resident or (rid in window.reserve and not prefill):
                continue
            groups.append(ScheduledSequenceGroup(seq_group=group, token_chunk_size=count))
            tokens += count
            if rid in window.reserve:
                ctl.emit("reserve", rid=rid)
        if not groups and not incoming and not ignored and available:
            raise RuntimeError("highest-priority request cannot fit the configured token/KV budget")
        self.prepared.update(g.seq_group.request_id for g in groups)
        return output(groups)
