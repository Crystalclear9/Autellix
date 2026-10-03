"""Paper scheduling semantics, independent of simulated execution costs."""
import unittest

from autellix.core.models import CallSpec, CallState, EngineState, ProcessEntry, ProgramSpec
from autellix.core.load_balancer import LocalityAwareLoadBalancer
from autellix.core.schedulers import make_scheduler
from autellix.core.simulator import Simulator
from autellix.core.execution import ExecutionModel


class PaperSimulationTests(unittest.TestCase):
    def test_new_placement_does_not_invent_cached_program_history(self):
        engine = EngineState(1, batch_size=1, queue_count=1)
        table = {"p": ProcessEntry("p", arrival_time=0, completed_call_ids={"first"},
                                   engine_ids={0}, completed_engine_ids={0})}
        spec = CallSpec("next", "p", 1, prefill_tokens=2000)
        LocalityAwareLoadBalancer().assign(spec, [engine], table)
        model = ExecutionModel()
        call = CallState(spec)
        self.assertEqual(table["p"].engine_ids, {0, 1})
        self.assertLess(model.cache_hit_rate(call, table["p"], 1),
                        model.cache_hit_rate(call, table["p"], 0))

    def test_routing_uses_known_input_only_at_2048_boundary(self):
        balancer = LocalityAwareLoadBalancer()
        engines = [EngineState(i, batch_size=1, queue_count=1) for i in range(2)]
        table = {"p": ProcessEntry("p", arrival_time=0, engine_id=0)}
        engines[0].queues[0].append(CallState(CallSpec("busy", "other", 1)))
        for output_length in (0, 1, 10000):
            short = CallSpec("short", "p", 1, prefill_tokens=2048, decode_tokens=output_length)
            long = CallSpec("long", "p", 1, prefill_tokens=2049, decode_tokens=output_length)
            self.assertEqual(balancer.assign(short, engines, table).engine_id, 1)
            self.assertEqual(balancer.assign(long, engines, table).engine_id, 0)

    def test_quantum_expiry_cannot_interrupt_multi_step_window(self):
        programs = [ProgramSpec(pid, (CallSpec("call", pid, 6),)) for pid in ("a", "b")]
        result = Simulator(programs, scheduler="round-robin", batch_size=1,
                           schedule_interval=3).run()
        first = [(row["time"], row["program_id"]) for row in result.gantt[:6]]
        self.assertEqual(first, [(0, "a"), (1, "a"), (2, "a"),
                                 (3, "b"), (4, "b"), (5, "b")])

    def test_higher_priority_arrival_preempts_at_boundary_before_quantum_expiry(self):
        scheduler = make_scheduler("plas", priority_boundaries=(0, 1, float("inf")),
                                   queue_quanta=(1, 100), anti_starvation_beta=1e9)
        programs = [ProgramSpec("a", (CallSpec("long", "a", 10),)),
                    ProgramSpec("b", (CallSpec("short", "b", 1, submit_time=4),))]
        result = Simulator(programs, scheduler=scheduler, batch_size=1, schedule_interval=3).run()
        self.assertEqual(result.calls[("b", "short")].start_time, 6)
        self.assertEqual([r["program_id"] for r in result.gantt[:7]], ["a"] * 6 + ["b"])

    def test_running_request_keeps_fifo_position_across_boundaries(self):
        programs = [ProgramSpec("a", (CallSpec("first", "a", 4),)),
                    ProgramSpec("b", (CallSpec("second", "b", 1, submit_time=1),))]
        result = Simulator(programs, scheduler="fcfs", batch_size=1).run()
        self.assertEqual(result.calls[("b", "second")].start_time, 4)

    def test_window_begins_at_arrival_instead_of_global_clock_multiple(self):
        program = ProgramSpec("p", (CallSpec("call", "p", 2),), arrival_time=2)
        result = Simulator([program], schedule_interval=8, batch_size=1).run()
        self.assertEqual(result.calls[("p", "call")].start_time, 2)

    def test_mlfq_starvation_does_not_inherit_program_wait(self):
        scheduler = make_scheduler("mlfq", priority_boundaries=(0, 1, float("inf")),
                                   queue_quanta=(1, 10), anti_starvation_beta=2)
        engine = EngineState(0, batch_size=1, queue_count=2)
        table = {"p": ProcessEntry("p", arrival_time=0, waiting_time=100)}
        call = CallState(CallSpec("c", "p", 10))
        scheduler.enqueue(call, engine, table, 0)
        engine.queues[0].remove(call)
        scheduler.demote(call, engine)
        scheduler.refresh(engine, table)
        self.assertEqual(call.queue_index, 1)
