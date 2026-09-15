"""Stateful client for the real HTTP service (standard library only)."""
from contextlib import contextmanager
import json
from urllib.request import Request, urlopen
from urllib.parse import quote


class InferenceClient:
    def __init__(self, base_url="http://127.0.0.1:8000", timeout=600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = Request(self.base_url + path, data=data, method=method,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=self.timeout) as response:
            return json.load(response)

    @contextmanager
    def session(self, program_id=None):
        sid = self._request("POST", "/sessions", {"program_id": program_id})["session_id"]
        try:
            yield sid
        finally:
            self._request("DELETE", "/sessions/" + quote(sid, safe=""))

    def chat(self, messages, *, session_id=None, **options):
        return self._request("POST", "/v1/chat/completions",
                             dict(messages=messages, session_id=session_id, **options))
