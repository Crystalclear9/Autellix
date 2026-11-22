"""Admission semantics independent of native phase-preference heuristics."""
import unittest
from collections import deque
from types import SimpleNamespace as NS
from unittest.mock import patch

from autellix.runtime.policy import PolicyConfig
from integrations.vllm.paper import PaperPlan


class Group:
    def __init__(self, rid, prefill=False):
        self.request_id, self.prefill = rid, prefill
        self.seq = NS(data=NS(get_num_uncomputed_tokens=lambda: 4))

    def is_prefill(self):
        return self.prefill

    def get_seqs(self):
        return [self.seq]

    def init_multi_step(self, **kwargs):
        pass


class Budget:
    def __init__(self, tokens, seqs):
        self.remaining = tokens

    def add_num_batched_tokens(self, rid, count):
        self.remaining -= count


class PaperPlanTests(unittest.TestCase):
    def setUp(self):
        self.status = NS(OK=0, LATER=1, NEVER=2)
        self.modules = patch.dict("sys.modules", {
            "vllm.core.scheduler": NS(SchedulerOutputs=NS, ScheduledSequenceGroup=NS, SchedulingBudget=Budget),
            "vllm.core.interfaces": NS(AllocStatus=self.status),
            "vllm.sequence": NS(SequenceStatus=NS(FINISHED_IGNORED=9))})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.events, self.refreshes = [], []
        self.priority = {}
        self.ctl = NS(config=PolicyConfig(schedule_interval=3, overprovision=1),
                      refresh=lambda: self.refreshes.append(1), key=lambda r: self.priority[r],
                      emit=lambda event, **kw: self.events.append((event, kw)), begin=lambda _: None)
        self.s = NS(running=deque(), waiting=deque(), swapped=deque(),
                    scheduler_config=NS(max_num_batched_tokens=32),
                    block_manager=NS(can_allocate=lambda _: 0, can_swap_in=lambda *a: 0,
                                     can_append_slots=lambda *a: True, append_slots=lambda *a: []),
                    _allocate_and_set_running=lambda _: None, free_seq=lambda _: None,
                    _swap_in=lambda group, blocks: blocks.append((1, 2)),
                    _preempt_by_swap=lambda group, blocks: blocks.append((3, 4)))

    def test_prefill_cannot_displace_higher_priority_decode(self):
        self.ctl.config = PolicyConfig(schedule_interval=3, overprovision=0)
        self.priority.update(high=0, low=1)
        self.s.running.append(Group("high"))
        self.s.waiting.append(Group("low", True))
        plan = PaperPlan(self.s, self.ctl, 1)
        out = plan.next_step()
        self.assertEqual([g.seq_group.request_id for g in out.scheduled_seq_groups], ["high"])
        self.assertEqual(len(self.s.waiting), 1)

    def test_mixed_batch_keeps_both_admitted_phases(self):
        self.ctl.config = PolicyConfig(overprovision=0)
        self.priority.update(high=0, low=1)
        self.s.running.append(Group("high"))
        self.s.waiting.append(Group("low", True))
        out = PaperPlan(self.s, self.ctl, 2).next_step()
        self.assertEqual({g.seq_group.request_id for g in out.scheduled_seq_groups}, {"high", "low"})
        self.assertEqual(out.num_prefill_groups, 1)
        self.assertEqual(out.num_batched_tokens, 5)

    def test_swap_state_does_not_override_global_priority(self):
        self.ctl.config = PolicyConfig(overprovision=0)
        self.priority.update(high=0, low=1)
        self.s.running.append(Group("low"))
        self.s.swapped.append(Group("high"))
        plan = PaperPlan(self.s, self.ctl, 1)
        self.assertTrue(plan.next_step().blocks_to_swap_out)
        out = plan.next_step()
        self.assertEqual([g.seq_group.request_id for g in out.scheduled_seq_groups], ["high"])
        self.assertTrue(out.blocks_to_swap_in)
        self.assertEqual(len(self.refreshes), 1)

    def test_inactive_kv_stays_resident_without_changing_the_selected_batch(self):
        self.ctl.config = PolicyConfig(overprovision=0)
        self.priority.update(high=0, low=1)
        inactive, waiting = Group("low"), Group("high", True)
        self.s.running.append(inactive)
        self.s.waiting.append(waiting)
        out = PaperPlan(self.s, self.ctl, 1, keep_resident_kv=True).next_step()
        self.assertEqual([g.seq_group.request_id for g in out.scheduled_seq_groups], ["high"])
        self.assertIn(inactive, self.s.running)
        self.assertFalse(out.blocks_to_swap_out)

    def test_allocation_pressure_evicts_lowest_priority_inactive_kv(self):
        self.ctl.config = PolicyConfig(overprovision=0)
        self.priority.update(high=0, low=1, lowest=2)
        low, lowest = Group("low"), Group("lowest")
        self.s.running.extend([lowest, low])
        self.s.waiting.append(Group("high", True))
        self.s.block_manager.can_allocate = lambda _: self.status.LATER if lowest in self.s.running else self.status.OK
        plan = PaperPlan(self.s, self.ctl, 1, keep_resident_kv=True)
        self.assertTrue(plan.next_step().blocks_to_swap_out)
        self.assertEqual(list(self.s.swapped), [lowest])
        out = plan.next_step()
        self.assertEqual([g.seq_group.request_id for g in out.scheduled_seq_groups], ["high"])
        self.assertIn(low, self.s.running)

    def test_one_barrier_batches_multiple_capacity_evictions(self):
        self.ctl.config = PolicyConfig(overprovision=0)
        self.priority.update(high=0, low=1, lowest=2)
        low, lowest = Group("low"), Group("lowest")
        self.s.running.extend([low, lowest])
        self.s.waiting.append(Group("high", True))
        self.s.block_manager.can_allocate = lambda _: self.status.LATER if self.s.running else self.status.OK
        plan = PaperPlan(self.s, self.ctl, 1, keep_resident_kv=True)
        out = plan.next_step()
        self.assertEqual(len(out.blocks_to_swap_out), 2)
        self.assertEqual(list(self.s.swapped), [lowest, low])
        self.assertFalse(out.scheduled_seq_groups)
        out = plan.next_step()
        self.assertEqual([g.seq_group.request_id for g in out.scheduled_seq_groups], ["high"])

    def test_host_swap_pressure_requeues_the_victim_for_recompute(self):
        self.ctl.config = PolicyConfig(overprovision=0)
        self.priority.update(high=0, low=1)
        high, low = Group("high", True), Group("low")
        self.s.running.append(low)
        self.s.waiting.append(high)
        self.s.block_manager.can_allocate = lambda _: self.status.LATER if low in self.s.running else self.status.OK
        self.s.block_manager.can_swap_out = lambda _: False
        self.s._preempt_by_recompute = lambda group: setattr(group, "prefill", True)
        plan = PaperPlan(self.s, self.ctl, 1, keep_resident_kv=True, recompute_on_swap_full=True)
        plan.prepared.add("low")
        self.assertFalse(plan.next_step().blocks_to_swap_out)
        self.assertNotIn(low, self.s.running)
        self.assertFalse(self.s.swapped)
        self.assertNotIn("low", plan.prepared)
        self.assertTrue(low.is_prefill())
        self.assertIn(("recompute_retract", {"rid": "low", "reason": "host_kv_full"}), self.events)
        self.assertEqual([g.seq_group for g in plan.next_step().scheduled_seq_groups], [high])
        self.s.running.remove(high)
        self.assertEqual([g.seq_group for g in plan.next_step().scheduled_seq_groups], [low])

    def test_window_refills_without_readmitting_new_arrival(self):
        self.priority.update(a=0, b=1, urgent=-1)
        a, b = Group("a", True), Group("b", True)
        self.s.waiting.extend([a, b])
        plan = PaperPlan(self.s, self.ctl, 1)
        plan.next_step()  # reserve prefill
        a.prefill = b.prefill = False
        # Prepared continuation must not run fit/admission checks again.
        self.s.block_manager.can_append_slots = lambda *a: self.fail("readmission inside window")
        plan.next_step()
        plan.window.executed()
        self.s.running.remove(a)  # early completion
        self.s.waiting.append(Group("urgent", True))
        out = plan.next_step()
        self.assertEqual([g.seq_group.request_id for g in out.scheduled_seq_groups], ["b"])
        self.assertEqual(len(self.refreshes), 1)
        self.assertTrue(any(event == "refill" for event, _ in self.events))
        plan.window.executed()
        self.s.block_manager.can_append_slots = lambda *a: True
        plan.next_step()
        plan.window.executed()
        plan.next_step()
        self.assertEqual(len(self.refreshes), 2)

    def test_invalid_implementation_rejected(self):
        with self.assertRaises(ValueError):
            PolicyConfig(implementation="silently-fallback")

    def test_idle_refill_preserves_live_decode_and_window_boundary(self):
        self.ctl.config = PolicyConfig(schedule_interval=3, overprovision=0)
        self.priority.update(a=1, b=0)
        active = Group("a")
        self.s.running.append(active)
        plan = PaperPlan(self.s, self.ctl, 2, keep_resident_kv=True, refill_idle_slots=True)
        plan.next_step()
        plan.window.executed()
        self.s.waiting.append(Group("b", True))
        out = plan.next_step()
        self.assertEqual({g.seq_group.request_id for g in out.scheduled_seq_groups}, {"a", "b"})
        self.assertEqual(plan.window.cohort, ["a", "b"])
        self.assertEqual(plan.window.remaining, 2)
        self.assertEqual(len(self.refreshes), 1)

    def test_reserve_uses_ready_kv_without_lower_priority_prefill(self):
        self.priority.update(active=0, waiting=1, resident=2)
        self.s.running.extend([Group("active"), Group("resident")])
        self.s.waiting.append(Group("waiting", True))
        plan = PaperPlan(self.s, self.ctl, 1, keep_resident_kv=True, resident_reserve_only=True)
        plan.prepared.update(["active", "resident"])
        first = plan.next_step()
        self.assertEqual(plan.window.cohort, ["active", "resident"])
        self.assertEqual([g.seq_group.request_id for g in first.scheduled_seq_groups], ["active"])
        plan.window.executed()
        self.s.running.popleft()
        refill = plan.next_step()
        self.assertEqual([g.seq_group.request_id for g in refill.scheduled_seq_groups], ["resident"])
        self.assertEqual([g.request_id for g in self.s.waiting], ["waiting"])

    def test_idle_refill_does_not_bypass_capacity_limited_prefix(self):
        self.ctl.config = PolicyConfig(schedule_interval=3, overprovision=0)
        self.priority.update(a=0, b=1)
        self.s.scheduler_config.max_num_batched_tokens = 5
        self.s.waiting.extend([Group("a", True), Group("b", True)])
        plan = PaperPlan(self.s, self.ctl, 2, keep_resident_kv=True, refill_idle_slots=True)
        plan.next_step()
        self.assertEqual(plan.window.cohort, ["a"])
        out = plan.next_step()
        self.assertEqual([g.seq_group.request_id for g in out.scheduled_seq_groups], ["a"])
        self.assertEqual(len(self.refreshes), 1)

    def test_resident_reserve_does_not_consume_decode_token_budget(self):
        self.priority.update(a=0, reserve=1)
        self.s.scheduler_config.max_num_batched_tokens = 1
        self.s.running.extend([Group("a"), Group("reserve")])
        plan = PaperPlan(self.s, self.ctl, 1)
        plan.prepared.update(["a", "reserve"])
        out = plan.next_step()  # New window re-reserves future KV slots.
        self.assertEqual(plan.window.cohort, ["a", "reserve"])
        self.assertEqual(out.num_batched_tokens, 1)
        self.assertEqual([g.seq_group.request_id for g in out.scheduled_seq_groups], ["a"])

    def test_full_prepared_window_does_not_rescan_backlog(self):
        class UnreadableQueue(deque):
            def __iter__(self):
                raise AssertionError("a prepared decode step scanned the backlog")

        self.ctl.config = PolicyConfig(schedule_interval=3, overprovision=0)
        self.priority.update(a=0)
        self.s.running.append(Group("a"))
        plan = PaperPlan(self.s, self.ctl, 1, keep_resident_kv=True, refill_idle_slots=True)
        plan.next_step()
        plan.window.executed()
        self.s.waiting = UnreadableQueue([Group("new", True)])
        self.s.swapped = UnreadableQueue([Group("paused")])
        out = plan.next_step()
        self.assertEqual([g.seq_group.request_id for g in out.scheduled_seq_groups], ["a"])
        self.assertEqual(len(self.refreshes), 1)

    def test_chunked_prefill_can_exceed_single_iteration_token_budget(self):
        self.ctl.config = PolicyConfig(schedule_interval=3, overprovision=0)
        self.priority.update(long=0)
        group = Group("long", True)
        group.seq.status = "waiting"
        remaining = 10
        group.seq.data.get_num_uncomputed_tokens = lambda: remaining
        self.s.waiting.append(group)
        self.s.scheduler_config = NS(max_num_batched_tokens=4, max_num_seqs=1)
        self.s._get_num_new_tokens = lambda group, status, chunk, budget: min(remaining, budget.remaining)
        plan = PaperPlan(self.s, self.ctl, 1, chunked_prefill=True)
        for count in (4, 4, 2):
            out = plan.next_step()
            self.assertEqual(out.num_batched_tokens, count)
            self.assertEqual("long" in plan.reserved_slots, count == 2)
            self.assertEqual("long" in plan.prepared, count == 2)
            remaining -= count
        self.assertEqual(len(self.refreshes), 1)

    def test_chunk_budget_keeps_pending_prefill_and_resident_decode_in_window(self):
        self.ctl.config = PolicyConfig(schedule_interval=3, overprovision=0)
        self.priority.update(first=0, pending=1, running=2)
        first, pending, running = Group("first", True), Group("pending", True), Group("running")
        first.seq.status = pending.seq.status = "waiting"
        pending.seq.data.get_num_uncomputed_tokens = lambda: 1
        self.s.waiting.extend([first, pending])
        self.s.running.append(running)
        self.s.scheduler_config = NS(max_num_batched_tokens=4, max_num_seqs=3)
        self.s._get_num_new_tokens = lambda group, status, chunk, budget: min(
            group.seq.data.get_num_uncomputed_tokens(), budget.remaining)
        plan = PaperPlan(self.s, self.ctl, 3, chunked_prefill=True)

        first_step = plan.next_step()
        self.assertEqual([g.seq_group for g in first_step.scheduled_seq_groups], [first])
        self.assertEqual(plan.window.cohort, ["first", "pending", "running"])
        first.prefill = False
        second_step = plan.next_step()
        self.assertEqual([g.seq_group for g in second_step.scheduled_seq_groups], [pending, first, running])
        self.assertFalse(second_step.blocks_to_swap_out)
        self.assertEqual(len(self.refreshes), 1)

    def test_full_prefill_budget_keeps_resident_decode_in_window(self):
        self.ctl.config = PolicyConfig(schedule_interval=3, overprovision=0)
        self.priority.update(first=0, running=1)
        first, running = Group("first", True), Group("running")
        first.seq.status = "waiting"
        self.s.waiting.append(first)
        self.s.running.append(running)
        self.s.scheduler_config = NS(max_num_batched_tokens=4, max_num_seqs=2)
        self.s._get_num_new_tokens = lambda group, status, chunk, budget: budget.remaining
        plan = PaperPlan(self.s, self.ctl, 2, chunked_prefill=True)

        plan.next_step()
        first.prefill = False
        out = plan.next_step()
        self.assertFalse(out.blocks_to_swap_out)
        self.assertEqual([g.seq_group for g in out.scheduled_seq_groups], [first, running])

    def test_chunked_plan_still_shrinks_at_true_kv_pressure(self):
        self.ctl.config = PolicyConfig(schedule_interval=3, overprovision=0)
        self.priority.update(first=0, blocked=1)
        first, blocked = Group("first"), Group("blocked", True)
        blocked.seq.status = "waiting"
        self.s.running.append(first)
        self.s.waiting.append(blocked)
        self.s.scheduler_config.max_num_seqs = 2
        self.s._get_num_new_tokens = lambda *args: 4
        self.s.block_manager.can_allocate = lambda _: self.status.LATER
        plan = PaperPlan(self.s, self.ctl, 2, chunked_prefill=True)

        out = plan.next_step()
        self.assertEqual([g.seq_group for g in out.scheduled_seq_groups], [first])
        self.assertEqual(plan.window.cohort, ["first"])
        self.assertTrue(plan.window.capacity_limited)

    def test_oversized_prefill_rejected_before_program_admission(self):
        from integrations.vllm.backend import VLLMBackend
        from unittest.mock import Mock
        backend = VLLMBackend.__new__(VLLMBackend)
        backend.scheduler = NS(_autellix_paper=NS(chunked_prefill=False), scheduler_config=NS(max_num_batched_tokens=4))
        backend.engine = NS(model_config=NS(max_model_len=16, hf_config=NS(vocab_size=100)), add_request=Mock())
        backend.controller, backend.table = Mock(), Mock()
        with patch.dict("sys.modules", {"vllm": NS(SamplingParams=Mock())}):
            with self.assertRaisesRegex(ValueError, "whole-prefill token budget"):
                backend.add("r", "p", [1] * 5, {})
            backend.controller.admit.assert_not_called()
            backend.engine.add_request.assert_not_called()
            backend.add("next", "p", [1] * 4, {})
            backend.controller.admit.assert_called_once()
            backend.engine.add_request.assert_called_once()

    def test_swapped_reserve_restores_without_decoding_until_refill(self):
        self.priority.update(a=0, reserve=1)
        a, reserve = Group("a"), Group("reserve")
        self.s.running.append(a)
        self.s.swapped.append(reserve)
        plan = PaperPlan(self.s, self.ctl, 1)
        out = plan.next_step()
        self.assertTrue(out.blocks_to_swap_in)
        self.assertEqual([g.seq_group.request_id for g in out.scheduled_seq_groups], ["a"])
        self.assertIn("reserve", plan.prepared)
        plan.window.executed()
        out = plan.next_step()
        self.assertEqual([g.seq_group.request_id for g in out.scheduled_seq_groups], ["a"])
        self.s.running.remove(a)
        out = plan.next_step()
        self.assertEqual([g.seq_group.request_id for g in out.scheduled_seq_groups], ["reserve"])

    def test_first_nonfitting_request_ends_window_admission(self):
        self.ctl.config = PolicyConfig(schedule_interval=3, overprovision=0)
        self.priority.update(a=0, b=1)
        self.s.scheduler_config.max_num_batched_tokens = 4
        self.s.running.append(Group("a"))
        self.s.waiting.append(Group("b", True))
        plan = PaperPlan(self.s, self.ctl, 2)
        plan.next_step()
        self.assertEqual(plan.window.cohort, ["a"])
        plan.window.executed()
        self.s.block_manager.can_append_slots = lambda *a: self.fail("capacity rechecked inside window")
        out = plan.next_step()
        self.assertEqual([g.seq_group.request_id for g in out.scheduled_seq_groups], ["a"])
        self.assertEqual(len(self.refreshes), 1)
