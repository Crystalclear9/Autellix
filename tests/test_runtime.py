import json
import math
import os
import tempfile
import unittest
from dataclasses import replace

from autellix.runtime.policy import PolicyConfig, ProgramTable, RuntimeScheduler
from autellix.core.execution import ExecutionModel
from autellix.core.models import CallSpec, ProgramSpec
from autellix.core.simulator import Simulator


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class RuntimePolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.table = ProgramTable(os.path.join(self.tmp.name, "state.sqlite"))
        self.clock = Clock()
        self.config = PolicyConfig(boundaries=(0, 2, 4, math.inf), quanta=(1, 2, 4))
        self.ctl = RuntimeScheduler(self.table, self.config, clock=self.clock)
        self.table.open("p")

    def tearDown(self):
        self.table.close()
        self.tmp.cleanup()

    def run_call(self, rid, duration):
        self.ctl.admit(rid, "p")
        self.ctl.begin([rid])
        self.clock.now += duration
        self.ctl.executed([rid], duration)
        return self.ctl.finish(rid)

    def test_online_inheritance_does_not_require_parent_ids(self):
        self.run_call("first", 3)
        self.ctl.admit("next", "p")
        self.assertEqual(self.ctl.calls["next"].inherited, 3)
        self.assertEqual(self.ctl.calls["next"].queue, 1)

    def test_concurrent_atlas_completions_take_max_not_sum(self):
        self.ctl.admit("a", "p")
        self.ctl.admit("b", "p")
        self.ctl.begin(["a", "b"])
        self.clock.now = 2
        self.ctl.executed(["a", "b"], 2)
        self.ctl.finish("a")
        self.ctl.finish("b")
        self.assertEqual(self.table.snapshot()["p"][0], 2)

    def test_cross_connection_program_statistics_and_close(self):
        peer = ProgramTable(self.table.path)
        try:
            ctl = RuntimeScheduler(peer, self.config, clock=self.clock)
            self.run_call("a", 3)
            ctl.admit("b", "p")
            self.assertEqual(ctl.calls["b"].inherited, 3)
            self.table.end("p")
            with self.assertRaises(ValueError):
                self.ctl.admit("late", "p")
            ctl.finish("b")
            self.assertNotIn("p", self.table.snapshot())
        finally:
            peer.close()

    def test_promotion_resets_windows_not_lifetime_service(self):
        self.run_call("history", 3)
        self.ctl.admit("slow", "p")
        self.ctl.begin(["slow"])
        self.clock.now += 2
        self.ctl.executed(["slow"], 2)
        self.ctl.refresh()
        self.clock.now += 80
        self.ctl.refresh()
        call = self.ctl.calls["slow"]
        self.assertEqual(call.queue, 0)
        self.assertEqual(call.executed, 2)
        self.assertEqual(call.wait_window, 0)
        self.assertEqual(call.run_window, 0)
        self.ctl.finish("slow")
        self.assertEqual(self.table.snapshot()["p"][0], 5)

    def test_duplicate_admission_does_not_corrupt_active_count(self):
        self.ctl.admit("a", "p")
        with self.assertRaises(ValueError):
            self.ctl.admit("a", "p")
        self.table.end("p")
        self.ctl.finish("a")
        self.ctl.finish("a")
        self.assertNotIn("p", self.table.snapshot())

    def test_swap_cost_changes_completion_time(self):
        programs = [ProgramSpec("p", (CallSpec("a", "p", 8),))]
        model = ExecutionModel(fixed_model_time=True)
        base = Simulator(programs, scheduler="round-robin", batch_size=1, execution_model=model).run()
        swapped = Simulator(programs, scheduler="round-robin", batch_size=1,
                            execution_model=replace(model, swap_penalty_per_preemption=10)).run()
        self.assertEqual(base.makespan, 8)
        self.assertEqual(swapped.makespan, 78)
        self.assertEqual(swapped.program_metrics["p"].execution_time, 8)

    def test_simulator_online_atlas_and_explicit_dag_are_distinct(self):
        programs = [ProgramSpec("p", (CallSpec("a", "p", 8), CallSpec("b", "p", 1, submit_time=10)))]
        online = Simulator(programs, scheduler="atlas", batch_size=1).run()
        dag = Simulator(programs, scheduler="atlas-dag", batch_size=1).run()
        self.assertEqual(online.calls[("p", "b")].service_priority, 8)
        self.assertEqual(dag.calls[("p", "b")].service_priority, 0)

    def test_invalid_policy_parameters(self):
        for options in ({"beta": float("nan")}, {"schedule_interval": 0},
                        {"schedule_interval": 1.5}, {"overprovision": True},
                        {"boundaries": (0, math.nan, math.inf), "quanta": (1, 2)},
                        {"boundaries": (0, 2, 1, math.inf), "quanta": (1, 2, 3)}):
            with self.assertRaises(ValueError):
                PolicyConfig(**options)

    def test_sglang_request_metadata_roundtrip(self):
        from integrations.sglang.scheduler import encode_request, decode_request
        self.assertEqual(decode_request(encode_request("程序/1", "call-1")), ("程序/1", "call-1"))
        with self.assertRaises(ValueError):
            decode_request("unannotated")

    def test_benchmark_validates_dag_without_using_model_time(self):
        from autellix.runtime.benchmark import validate_workload
        validate_workload([{"program_id": "p", "calls": [
            {"call_id": "a", "prompt": "hello"},
            {"call_id": "b", "prompt": "continue", "parents": ["a"]}]}])
        with self.assertRaisesRegex(ValueError, "cyclic"):
            validate_workload([{"program_id": "p", "calls": [
                {"call_id": "a", "prompt": "hello", "parents": ["a"]}]}])


if __name__ == "__main__":
    unittest.main()
