import concurrent.futures
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from autellix.runtime.client import InferenceClient
from autellix.runtime.engine import InferenceEngine, InferenceFuture
from autellix.runtime.policy import PolicyConfig, ProgramTable, RuntimeScheduler
from autellix.runtime.window import SchedulingWindow


class RuntimeRegressions(unittest.TestCase):
    def test_poisson_trace_is_reproducible_and_rejects_invalid_rates(self):
        from autellix.runtime.benchmark import poisson_trace
        source = [{"program_id": "p", "calls": [{"call_id": "a", "prompt": "real prompt"}]}]
        first = poisson_trace(source, 20, 2., seed=42)
        self.assertEqual(first, poisson_trace(source, 20, 2., seed=42))
        self.assertEqual(len({p["program_id"] for p in first}), 20)
        self.assertTrue(all(a["arrival_time"] < b["arrival_time"] for a, b in zip(first, first[1:])))
        self.assertNotIn("arrival_time", source[0])
        for rate in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                poisson_trace(source, 20, rate)

    def test_program_token_latency_uses_dag_critical_path(self):
        from autellix.runtime.benchmark import run_workload
        clock = [0.]
        durations = dict(root=2, left=3, right=5, join=4)
        def submit(pid, **kwargs):
            duration = durations[kwargs["call_id"]]
            clock[0] += duration
            future = concurrent.futures.Future()
            future.set_result(dict(text="output", token_ids=[1] * duration))
            return future
        engine = SimpleNamespace(start_session=lambda: "session", end_session=lambda _: None,
                                 submit=submit, policy=SimpleNamespace(policy="atlas"),
                                 replicas=[SimpleNamespace(backend="test")], model="test")
        workload = [{"program_id": "p", "calls": [
            {"call_id": "root", "prompt": "root"},
            {"call_id": "left", "prompt": "left", "parents": ["root"]},
            {"call_id": "right", "prompt": "right", "parents": ["root"]},
            {"call_id": "join", "prompt": "join", "parents": ["left", "right"]}]}]
        with patch("autellix.runtime.benchmark.time.monotonic", lambda: clock[0]):
            result = run_workload(engine, workload)
        self.assertEqual(result["program_critical_path_seconds"]["p"], 11)
        self.assertEqual(result["programs"]["p"], 14)
        self.assertAlmostEqual(result["program_token_latencies"]["p"], 11 / 14)

    def test_cancel_during_result_keeps_cleanup_and_next_result_alive(self):
        with tempfile.TemporaryDirectory() as directory:
            table = ProgramTable(str(Path(directory, "state.db")))
            engine = InferenceEngine.__new__(InferenceEngine)
            engine.table, engine.rpc, engine.sessions = table, {}, {}
            engine.lock = threading.RLock()
            engine.idle = threading.Condition(engine.lock)
            engine.loads = [2]
            first, second = InferenceFuture(), InferenceFuture()
            engine.pending = {"a": (first, "p", 0, "a"), "b": (second, "p", 0, "b")}
            entered, release = threading.Event(), threading.Event()
            original = first.update
            def update(value):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError()
                original(value)
            first.update = update
            errors = []
            def resolve():
                try:
                    with engine.lock:
                        engine._resolve("a", "result", {"text": "a"})
                        engine._resolve("b", "result", {"text": "b"})
                except BaseException as exc:
                    errors.append(exc)
            thread = threading.Thread(target=resolve)
            thread.start()
            self.assertTrue(entered.wait(5))
            first.cancel()
            release.set()
            thread.join(5)
            try:
                self.assertEqual(errors, [])
                self.assertEqual(second.result(1)["text"], "b")
                self.assertEqual(engine.pending, {})
                self.assertEqual(engine.loads, [0])
            finally:
                table.close()

    def test_automatic_program_shared_by_threads_and_closed_on_error(self):
        calls = []
        def request(client, method, path, body=None):
            calls.append((method, path, body))
            return {"session_id": "program"} if path == "/sessions" else {}
        with patch.object(InferenceClient, "_request", request):
            with self.assertRaisesRegex(ValueError, "application"):
                with InferenceClient() as client:
                    with concurrent.futures.ThreadPoolExecutor(3) as pool:
                        list(pool.map(lambda _: client.chat([{"role": "user", "content": "hi"}]), range(6)))
                    raise ValueError("application")
        self.assertEqual(sum(path == "/sessions" for _, path, _ in calls), 1)
        chats = [body for _, path, body in calls if path.endswith("completions")]
        self.assertEqual({c["session_id"] for c in chats}, {"program"})
        self.assertEqual(len({c["call_id"] for c in chats}), 6)
        self.assertTrue(all(c["thread_id"] for c in chats))
        self.assertEqual(calls[-1][:2], ("DELETE", "/sessions/program"))

    def test_cross_process_activity_and_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            table = ProgramTable(str(Path(directory, "state.db")))
            peer = ProgramTable(table.path)
            try:
                table.open("p")
                context = dict(request_id="r", engine_id=2, thread_id="thread", call_id="call")
                table.record_submission("r", "p", context)
                ctl = RuntimeScheduler(peer)
                ctl.admit("internal-r", "p", context)
                ctl.begin(["internal-r"])
                ctl.executed(["internal-r"], .25)
                state = table.describe()["p"]
                self.assertEqual(state["engine_ids"], [2])
                self.assertEqual(state["calls"][0]["executed"], .25)
                self.assertEqual(state["calls"][0]["thread_id"], "thread")
                ctl.finish("internal-r")
                table.complete_activity("r")
                state = peer.describe()["p"]
                self.assertGreaterEqual(state["last_completion"], state["last_arrival"])
                self.assertEqual(state["calls"], [])
            finally:
                peer.close()
                table.close()

    def test_window_refills_resident_reserve_before_next_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            table = ProgramTable(str(Path(directory, "state.db")))
            try:
                table.open("p")
                ctl = RuntimeScheduler(table, PolicyConfig(schedule_interval=4, overprovision=1))
                for rid in ("a", "b", "c"):
                    ctl.admit(rid, "p")
                window = SchedulingWindow(ctl, 1)
                chosen, _ = window.select(["a", "b", "c"], [])
                self.assertEqual(chosen, {"a", "b"})  # prefill reserve
                window.executed(is_prefill=True)
                chosen, reserve = window.select(["a", "b", "c"], ["a", "b"])
                self.assertEqual((chosen, reserve), ({"a"}, {"b"}))
                window.executed()
                ctl.finish("a")
                chosen, _ = window.select(["b", "c"], ["b"])
                self.assertEqual(chosen, {"b"})
                self.assertEqual(window.remaining, 3)
                self.assertNotIn("c", window.cohort)
            finally:
                table.close()
