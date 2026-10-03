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


class PaperPlanTests(unittest.TestCase):
    def setUp(self):
        self.status = NS(OK=0, LATER=1, NEVER=2)
        self.modules = patch.dict("sys.modules", {
            "vllm.core.scheduler": NS(SchedulerOutputs=NS, ScheduledSequenceGroup=NS),
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

    def test_oversized_prefill_rejected_before_program_admission(self):
        from integrations.vllm.backend import VLLMBackend
        from unittest.mock import Mock
        backend = VLLMBackend.__new__(VLLMBackend)
        backend.scheduler = NS(_autellix_paper=object(), scheduler_config=NS(max_num_batched_tokens=4))
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
