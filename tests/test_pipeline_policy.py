import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock, patch

from autellix.runtime import ReplicaConfig
from autellix.runtime.policy import PolicyConfig, ProgramTable, RuntimeScheduler
from integrations.sglang.pipeline import (PipelineTable, PolicyTape, abort_inflight,
                                          compact_control_transfer, configure, host_control_transfer, install)


class PipelinePolicyTests(unittest.TestCase):
    def test_host_control_relays_only_through_the_current_ranks_endpoints(self):
        frame = (["request"], [("clock", 12.34), ("table.snapshot", {"p": (.031, .025)})])
        incoming, outgoing, unused_read, unused_write = Mock(), Mock(), Mock(), Mock()
        incoming.recv.return_value = frame
        relay = host_control_transfer(1, [(incoming, unused_write), (unused_read, outgoing)])
        value = relay([], 1, None, 0, 1)
        self.assertEqual(value, frame)
        self.assertEqual(relay(value, 1, None, 1, 2), frame)
        outgoing.send.assert_called_once_with(frame)
        unused_read.close.assert_called_once_with()
        unused_write.close.assert_called_once_with()
        incoming.close.assert_not_called()
        outgoing.close.assert_not_called()

    def test_compact_control_replays_clocks_and_full_policy_payloads(self):
        try:
            import torch
        except ImportError:
            self.skipTest("requires tensors for pipeline control transport")
        wire = []
        leader, follower = compact_control_transfer("cpu"), compact_control_transfer("cpu")

        def recv(tensor, **kwargs):
            tensor.copy_(wire.pop(0))

        frames = [
            ([], [], 1),
            ([], [("clock", 12.34), ("clock", 12.35)], 1),
            (["arrival"], [("table.take_context", {"call_id": "first"})], 2),
            ([], [("clock", 12.36), ("table.snapshot", {"p": (.031, .025)})], 2),
        ]
        with patch.object(torch.distributed, "send", side_effect=lambda tensor, **kw: wire.append(tensor.clone())), \
             patch.object(torch.distributed, "recv", side_effect=recv):
            for requests, entries, messages in frames:
                leader((requests, entries), 0, None, 0, 1)
                self.assertEqual(len(wire), messages)
                self.assertEqual(follower([], 1, None, 0, 1), (requests, entries))
                self.assertEqual(wire, [])

    def test_pipeline_timing_follows_activations_and_sampled_tokens(self):
        for metadata in (True, False):
            with self.subTest(metadata=metadata):
                wires = [[], [], []]

                class Stage:
                    def __init__(self, rank):
                        self.pp_size, self.device = len(wires), "cuda"
                        self.pp_group = NS(
                            is_first_rank=rank == 0,
                            is_last_rank=rank == len(wires) - 1,
                            send_tensor_dict=wires[rank].append,
                            recv_tensor_dict=lambda: wires[rank - 1].pop(0),
                        )
                        self.tp_worker = NS(model_runner=NS(forward=lambda: "model output"))

                synchronize = Mock()

                def event(**kwargs):
                    return Mock(elapsed_time=lambda _: 5., synchronize=synchronize)

                cuda_event = Mock(side_effect=event)
                tensor = Mock(side_effect=lambda value, **kwargs: NS(item=lambda: value))
                fake_torch = NS(cuda=NS(Event=cuda_event), tensor=tensor, float64="float64")
                with patch.dict(sys.modules, {"torch": fake_torch}):
                    install(Stage, timing_metadata=metadata)
                first, middle, last = (Stage(i) for i in range(len(wires)))

                for _ in range(2):
                    for source, target, key in ((first, middle, "hidden_states"), (middle, last, "hidden_states"),
                                                (last, first, "next_token_ids")):
                        previous_waits = synchronize.call_count
                        self.assertEqual(source.tp_worker.model_runner.forward(), "model output")
                        self.assertEqual(synchronize.call_count, previous_waits)
                        source.pp_group.send_tensor_dict({key: [7]})
                        self.assertEqual(synchronize.call_count, previous_waits + (source is not first))
                        received = target.pp_group.recv_tensor_dict()
                        self.assertEqual(synchronize.call_count,
                                         previous_waits + (source is not first) + (target is first))
                        self.assertEqual(received[key], [7])
                        self.assertEqual("autellix_gpu_seconds" in received, key == "next_token_ids")
                    self.assertAlmostEqual(first._autellix_completed_seconds, .015)
                    first.pp_group.send_tensor_dict(received)
                    middle.pp_group.recv_tensor_dict()
                    self.assertAlmostEqual(middle._autellix_completed_seconds, .015)
                self.assertEqual(cuda_event.call_count, 2 * (2 * len(wires) - 1))
                self.assertEqual(tensor.call_count, 0 if metadata else 2 * len(wires))

    def test_first_stage_pairs_overlapping_forwards_with_fifo_results(self):
        returned = iter([{"next_token_ids": [1], "autellix_gpu_seconds": .010},
                         {"next_token_ids": [2], "autellix_gpu_seconds": .020}])
        sent = []

        class FirstStage:
            def __init__(self):
                self.pp_size = 2
                self.pp_group = NS(is_first_rank=True, is_last_rank=False,
                                   recv_tensor_dict=lambda: next(returned), send_tensor_dict=sent.append)
                self.tp_worker = NS(model_runner=NS(forward=Mock()))

        start_one, end_one = Mock(elapsed_time=lambda end: 3.), Mock()
        start_two, end_two = Mock(elapsed_time=lambda end: 7.), Mock()
        fake_torch = NS(cuda=NS(Event=Mock(side_effect=[start_one, end_one, start_two, end_two])))
        with patch.dict(sys.modules, {"torch": fake_torch}):
            install(FirstStage)
        stage = FirstStage()
        for token in (1, 2):
            stage.tp_worker.model_runner.forward()
            stage.pp_group.send_tensor_dict({"hidden_states": [token]})
        self.assertEqual(sent, [{"hidden_states": [1]}, {"hidden_states": [2]}])
        end_one.synchronize.assert_not_called()
        end_two.synchronize.assert_not_called()
        self.assertAlmostEqual(stage.pp_group.recv_tensor_dict()["autellix_gpu_seconds"], .013)
        end_one.synchronize.assert_called_once()
        end_two.synchronize.assert_not_called()
        self.assertAlmostEqual(stage.pp_group.recv_tensor_dict()["autellix_gpu_seconds"], .027)
        end_two.synchronize.assert_called_once()

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
