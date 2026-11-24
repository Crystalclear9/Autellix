"""IPC contract tests with a deliberately fake worker; no model claims."""
import os
import tempfile
import unittest
from unittest.mock import patch

from autellix.runtime.benchmark import run_workload
from autellix.runtime.engine import InferenceEngine, ReplicaConfig


def fake_transport_worker(index, replica, policy, table_path, trace, commands, events):
    events.put((index, "ready", None, None))
    while True:
        command, rid, payload = commands.get()
        if command == "shutdown":
            return
        if command == "tokenize":
            events.put((index, "tokenized", rid, [1, 2]))
        if command in {"add", "add-group"}:
            requests = payload if command == "add-group" else [(rid, payload)]
            for rid, request in requests:
                if request["sampling"].get("crash"):
                    os._exit(3)
                if request["sampling"].get("hold"):
                    continue
                if not replica.engine_args.get("completion_only"):
                    events.put((index, "progress", rid, {"text": "worker"}))
                events.put((index, "result", rid, {"text": "worker result", "token_ids": [7],
                            "prompt_tokens": len(request["prompt"]), "finish_reason": "stop",
                            "metrics": {"admission_size": len(requests)}}))
        if command == "cancel":
            events.put((index, "cancelled", rid, None))


class IPCTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = patch("autellix.runtime.engine._worker", fake_transport_worker)
        self.patch.start()
        self.engine = InferenceEngine([ReplicaConfig("vllm", "fake") for _ in range(2)],
                                     state_dir=self.tmp.name, locality_threshold=3)
        self.engine.wait_ready(30)

    def tearDown(self):
        self.engine.shutdown()
        self.patch.stop()
        self.tmp.cleanup()

    def test_result_comes_from_worker_and_session_closes(self):
        pid = self.engine.start_session()
        result = self.engine.submit(pid, prompt="hi", sampling={"max_tokens": 1}).result(10)
        self.assertEqual(result["text"], "worker result")
        self.engine.end_session(pid)
        self.assertEqual(self.engine.table.snapshot(), {})

    def test_ready_siblings_share_one_worker_admission(self):
        workload = [dict(program_id="p", calls=[
            dict(call_id="a", prompt="branch", input_ids=[1] * 4),
            dict(call_id="b", prompt="branch", input_ids=[1] * 4),
            dict(call_id="join", prompt="join", input_ids=[1] * 4,
                 parents=["a", "b"], append_parent_outputs=False)])]
        results = run_workload(self.engine, workload, group_ready_calls=True)["calls"]
        self.assertEqual({r["workload_call"]: r["metrics"]["admission_size"] for r in results},
                         {"a": 2, "b": 2, "join": 1})
        self.assertEqual(self.engine.table.snapshot(), {})

    def test_cancel_before_group_delivery_skips_worker_admission(self):
        pid = self.engine.start_session()
        with self.engine.group_pretokenized_submissions():
            cancelled = self.engine.submit(pid, input_ids=[1] * 4, sampling={"crash": True})
            self.assertTrue(cancelled.cancel())
            live = self.engine.submit(pid, input_ids=[2] * 4)
        self.assertEqual(live.result(10)["metrics"]["admission_size"], 1)
        self.engine.end_session(pid)
        self.assertEqual(self.engine.table.snapshot(), {})

    def test_untraced_engine_removes_its_temporary_state(self):
        with InferenceEngine([ReplicaConfig("vllm", "fake")], scheduler_trace=False) as engine:
            state = engine.state_dir
            self.assertTrue((state / "programs.sqlite").is_file())
        self.assertFalse(state.exists())

    def test_completion_only_preserves_final_result_without_recording_ttft(self):
        replica = ReplicaConfig("vllm", "fake", engine_args={"completion_only": True})
        with InferenceEngine([replica], scheduler_trace=False) as engine:
            pid = engine.start_session()
            future = engine.submit(pid, input_ids=[1])
            result = future.result(10)
            self.assertEqual(result["token_ids"], [7])
            self.assertIsNone(result["ttft_seconds"])
            self.assertTrue(future.progress.empty())

    def test_short_balances_long_pins_and_cancel_releases_load(self):
        pid = self.engine.start_session()
        first = self.engine.submit(pid, input_ids=[1], sampling={"hold": True})
        second = self.engine.submit(pid, input_ids=[1], sampling={"hold": True})
        self.assertEqual(self.engine.loads, [1, 1])
        third = self.engine.submit(pid, input_ids=[1]*4, sampling={"hold": True})
        fourth = self.engine.submit(pid, input_ids=[1]*4, sampling={"hold": True})
        self.assertEqual(self.engine.loads, [3, 1])
        self.engine.end_session(pid)
        for future in (first, second, third, fourth):
            self.assertTrue(future.cancel())
        # pending is removed before SQLite/session cleanup; wait under the
        # coordinator condition so the assertions observe completed cleanup.
        self.engine.wait_idle(10)
        self.assertEqual(self.engine.loads, [0, 0])
        self.assertEqual(self.engine.table.snapshot(), {})

    def test_dead_worker_fails_future(self):
        pid = self.engine.start_session()
        future = self.engine.submit(pid, input_ids=[1], sampling={"crash": True})
        with self.assertRaisesRegex(RuntimeError, "code 3"):
            future.result(10)
        self.assertEqual(self.engine.loads, [0, 0])
        # The surviving replica remains usable.
        result = self.engine.submit(pid, input_ids=[1], sampling={"max_tokens": 1}).result(10)
        self.assertEqual(result["engine_id"], 1)


if __name__ == "__main__":
    unittest.main()
