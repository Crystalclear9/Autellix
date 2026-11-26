import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

from autellix.runtime import ReplicaConfig
from autellix.runtime.policy import PolicyConfig, ProgramTable, RuntimeScheduler
from integrations.sglang.pipeline import PipelineTable, PolicyTape, abort_inflight, configure


class PipelinePolicyTests(unittest.TestCase):
    def test_pipeline_applies_controller_and_window_options(self):
        packages = [ModuleType(name) for name in (
            "sglang", "sglang.srt", "sglang.srt.managers", "sglang.srt.managers.scheduler")]
        for parent, child in zip(packages, packages[1:]):
            setattr(parent, child.__name__.rsplit(".", 1)[-1], child)
        packages[-1].point_to_point_pyobj = lambda *args: None
        stage = NS(pp_rank=1, pp_size=2, recv_requests=lambda: None)
        with patch.dict(sys.modules, {module.__name__: module for module in packages}):
            configure(stage, dict(config=asdict(PolicyConfig()), capacity=8, publish_activity=False,
                                  publish_activity_at_window=True, refill_idle_slots=True,
                                  resident_reserve_only=True))
        self.assertFalse(stage.autellix.publish_activity)
        self.assertTrue(stage.autellix.publish_activity_at_window)
        self.assertEqual(len(stage._autellix_windows), 2)
        for window in stage._autellix_windows:
            self.assertEqual(window.capacity, 4)
            self.assertTrue(window.refill_idle_slots)
            self.assertTrue(window.resident_reserve_only)

    def test_all_stages_replay_policy_with_one_database_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            table = ProgramTable(str(Path(directory, "programs.sqlite")))
            table.open("p")
            leader = PolicyTape(True)
            follower = PolicyTape(False)
            first = RuntimeScheduler(PipelineTable(leader, table), PolicyConfig(), clock=leader.clock)
            second = RuntimeScheduler(PipelineTable(follower, None), PolicyConfig(), clock=follower.clock)

            def step(controller):
                controller.admit("a", "p")
                controller.admit("b", "p")
                controller.refresh()
                controller.begin(["a"])
                controller.begin(["b"], active_rids=["a"])
                self.assertTrue(controller.calls["a"].scheduled)
                controller.executed(["a"], .03)
                controller.refresh()
                return controller.finish("a")

            expected = step(first)
            follower.reset(leader.entries)
            self.assertEqual(step(second), expected)
            self.assertEqual(asdict(first.calls["b"]), asdict(second.calls["b"]))
            self.assertEqual(table.db.execute("SELECT active FROM programs").fetchone()[0], 1)
            follower.reset()
            table.close()

    def test_replica_group_covers_all_model_parallel_ranks(self):
        ReplicaConfig("sglang", "model", "0,1,2", {"pp_size": 3})
        ReplicaConfig("vllm", "model", "0,1,2", {"tensor_parallel_size": 3})
        with self.assertRaises(ValueError):
            ReplicaConfig("sglang", "model", "0", {"pp_size": 3})

    def test_mismatched_stage_policy_is_reported(self):
        follower = PolicyTape(False)
        follower.reset([("table.admit", (0, 0))])
        with self.assertRaisesRegex(RuntimeError, "diverged"):
            follower.clock()

    def test_cancel_reaches_all_forward_batches_including_pending_prefill(self):
        requests = [NS(rid=f"p-{i}", to_abort=False, finished=lambda: False) for i in range(3)]
        finished = NS(rid="p-old", to_abort=False, finished=lambda: True)
        scheduler = NS(_autellix_batches=[NS(reqs=[requests[0], finished]),
                                         NS(reqs=[requests[1]]), NS(reqs=[requests[2]])])
        abort_inflight(scheduler, NS(rid="p-2", abort_all=False))
        self.assertEqual([r.to_abort for r in requests], [False, False, True])
        abort_inflight(scheduler, NS(rid="", abort_all=True))
        self.assertTrue(all(r.to_abort for r in requests))
        self.assertFalse(finished.to_abort)


if __name__ == "__main__":
    unittest.main()
