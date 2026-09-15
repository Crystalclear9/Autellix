"""IPC contract tests with a deliberately fake worker; no model claims."""
import os
import tempfile
import time
import unittest
from unittest.mock import patch

from autellix.runtime.engine import InferenceEngine, ReplicaConfig


def fake_transport_worker(index, replica, policy, table_path, trace, commands, events):
    events.put((index, "ready", None, None))
    while True:
        command, rid, payload = commands.get()
        if command == "shutdown":
            return
        if command == "tokenize":
            events.put((index, "tokenized", rid, [1, 2]))
        if command == "add":
            if payload["sampling"].get("crash"):
                os._exit(3)
            if payload["sampling"].get("hold"):
                continue
            events.put((index, "progress", rid, {"text": "worker"}))
            events.put((index, "result", rid, {"text": "worker result", "token_ids": [7],
                        "prompt_tokens": len(payload["prompt"]), "finish_reason": "stop", "metrics": {}}))
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
        deadline = time.monotonic() + 10
        while self.engine.pending and time.monotonic() < deadline:
            time.sleep(.01)
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
