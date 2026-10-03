"""HTTP protocol tests; engine stub isolates protocol from GPU inference."""
import json
import concurrent.futures
import threading
import unittest
from types import SimpleNamespace
from autellix.runtime.engine import InferenceFuture

try:
    from fastapi.testclient import TestClient
except ImportError:
    TestClient = None


class ProtocolEngine:
    model = "test-model"
    closed = False
    failed = {}
    loads = [0]
    replicas = [SimpleNamespace(backend="test")]
    policy = SimpleNamespace(policy="atlas")

    def __init__(self):
        self.sessions = set()

    def wait_ready(self):
        pass

    def shutdown(self):
        self.closed = True

    def start_session(self, pid=None):
        pid = pid or "automatic"
        self.sessions.add(pid)
        return pid

    def end_session(self, pid):
        self.sessions.discard(pid)

    def submit(self, pid, **kwargs):
        if pid not in self.sessions:
            raise ValueError("unknown session")
        result = dict(text="hello", token_ids=[1], prompt_tokens=2,
                      finish_reason="stop", metrics={}, engine_id=0)
        future = InferenceFuture()
        future.update(result)
        future.set_result(result)
        return future


@unittest.skipUnless(TestClient, "requires fastapi/httpx")
class HTTPTests(unittest.TestCase):
    def test_duplicate_and_cancel_while_submission_is_pending(self):
        from autellix.runtime.server import create_app
        engine = ProtocolEngine()
        entered, release = threading.Event(), threading.Event()
        pending = InferenceFuture()
        def submit(pid, **kwargs):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test admission was not released")
            return pending
        engine.submit = submit
        body = dict(messages=[{"role": "user", "content": "hi"}],
                    session_id="p", request_id="same-id")
        with TestClient(create_app(engine)) as client:
            client.post("/sessions", json={"program_id": "p"})
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                first = pool.submit(client.post, "/v1/chat/completions", json=body)
                try:
                    self.assertTrue(entered.wait(5))
                    duplicate = client.post("/v1/chat/completions", json=body)
                    self.assertEqual(duplicate.status_code, 409)
                    self.assertEqual(client.delete("/requests/same-id").status_code, 200)
                finally:
                    release.set()
                self.assertEqual(first.result(5).status_code, 499)
            self.assertTrue(pending.cancelled())
            self.assertEqual(client.delete("/requests/same-id").status_code, 404)

    def test_sessions_usage_and_errors(self):
        from autellix.runtime.server import create_app
        engine = ProtocolEngine()
        with TestClient(create_app(engine)) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            sid = client.post("/sessions", json={"program_id": "p"}).json()["session_id"]
            body = dict(messages=[{"role": "user", "content": "hi"}], session_id=sid)
            result = client.post("/v1/chat/completions", json=body)
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(result.json()["usage"]["total_tokens"], 3)
            self.assertEqual(result.json()["choices"][0]["message"]["content"], "hello")
            self.assertEqual(client.post("/v1/chat/completions", json={**body, "model": "absent"}).status_code, 404)
            client.delete("/sessions/p")
            self.assertEqual(engine.sessions, set())

    def test_streaming_sse_and_automatic_session_cleanup(self):
        from autellix.runtime.server import create_app
        engine = ProtocolEngine()
        with TestClient(create_app(engine)) as client:
            response = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "hi"}], "stream": True})
            self.assertEqual(response.status_code, 200)
            chunks = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
            self.assertEqual(json.loads(chunks[0])["choices"][0]["delta"]["content"], "hello")
            self.assertEqual(chunks[-1], "[DONE]")
            self.assertEqual(engine.sessions, set())
