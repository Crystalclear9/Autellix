"""vLLM 0.6.1 scheduler integration. No backend imports at module import time."""
from collections import deque


def attach_scheduler(native, controller):
    from vllm.core.scheduler import Scheduler, SchedulerOutputs

    if type(native) is not Scheduler:
        raise TypeError("expected the unmodified vLLM 0.6.1 scheduler")

    class ProgramAwareScheduler(Scheduler):
        def _schedule(self):
            ctl = self.autellix
            self._autellix_step += 1
            available = list(self.running) + list(self.waiting) + list(self.swapped)
            for group in available:
                if group.request_id not in ctl.calls:
                    raise RuntimeError("request entered vLLM without Autellix metadata")
            # The native engine caches this schedule for num_scheduler_steps.
            # Each invocation here is already a multi-step policy boundary.
            ctl.refresh()
            ordered = sorted(available, key=lambda g: ctl.key(g.request_id))
            selected = []
            slots = 0
            for group in ordered:
                count = group.get_max_num_running_seqs()
                if slots + count > self.scheduler_config.max_num_seqs:
                    break
                selected.append(group.request_id)
                slots += count
            chosen = set(selected)
            reserve = []
            if not self._autellix_evict_reserve:
                reserve = [g.request_id for g in ordered if g in self.running and
                           g.request_id not in chosen][:ctl.config.overprovision]
            self._autellix_evict_reserve = False
            keep_resident = chosen | set(reserve)
            for rid in reserve:
                if rid not in self._autellix_resident_reserve:
                    ctl.emit("reserve", rid=rid)
            self._autellix_resident_reserve = set(reserve)
            blocks = []
            for group in list(self.running):
                if group.request_id not in keep_resident:
                    # Use native block ownership transitions; never drop output
                    # tokens or manufacture a new generation request.
                    self._preempt_by_swap(group, blocks)
                    self.running.remove(group)
                    self.swapped.append(group)
                    ctl.emit("swap_out", rid=group.request_id)
            self._autellix_selected = chosen
            if blocks:
                self._autellix_was_swapped.update(g.request_id for g in self.swapped)
                # v0.6.1 forbids simultaneous swap-in/out in one SchedulerOutputs.
                # Execute the outgoing transfer before scheduling the next batch.
                self._autellix_execution = []
                ctl.begin([])
                if self.scheduler_config.num_scheduler_steps > 1:
                    # MultiStepWorker rejects an empty sequence metadata list.
                    # Flush this transfer-only barrier directly through its
                    # cache engine; there is no model computation in this phase.
                    self._autellix_transfer_out(blocks)
                    blocks = []
                return SchedulerOutputs(
                    scheduled_seq_groups=[], num_prefill_groups=0,
                    num_batched_tokens=0, blocks_to_swap_in=[],
                    blocks_to_swap_out=blocks, blocks_to_copy=[],
                    ignored_seq_groups=[], num_lookahead_slots=0,
                    running_queue_size=len(self.running), preempted=0)

            held = {}
            for name in ("running", "waiting", "swapped"):
                queue = getattr(self, name)
                held[name] = [g for g in queue if g.request_id not in chosen]
                setattr(self, name, deque(sorted(
                    (g for g in queue if g.request_id in chosen),
                    key=lambda g: ctl.key(g.request_id))))
            try:
                outputs = super()._schedule()
            finally:
                for name, groups in held.items():
                    getattr(self, name).extend(groups)
            self._autellix_execution = [g.seq_group.request_id for g in outputs.scheduled_seq_groups]
            ctl.begin(self._autellix_execution)
            if not self._autellix_execution and reserve:
                self._autellix_evict_reserve = True
            for group in outputs.scheduled_seq_groups:
                if group.seq_group.request_id in self._autellix_was_swapped:
                    ctl.emit("resume", rid=group.seq_group.request_id)
            self._autellix_was_swapped = {g.request_id for g in self.swapped}
            return outputs

    native.__class__ = ProgramAwareScheduler
    native.autellix = controller
    native._autellix_step = 0
    native._autellix_selected = set()
    native._autellix_execution = []
    native._autellix_was_swapped = set()
    native._autellix_evict_reserve = False
    native._autellix_resident_reserve = set()
    return native
