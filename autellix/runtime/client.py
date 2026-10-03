"""Stateful client for the real HTTP service (standard library only)."""
from contextlib import contextmanager
from contextvars import ContextVar
import atexit
import json
import os
import threading
import uuid
from urllib.request import Request, urlopen
from urllib.parse import quote


class InferenceClient:
    def __init__(self, base_url="http://127.0.0.1:8000", timeout=600, program_id=None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.program_id = program_id
        self._sid = None
        self._closed = False
        self._lock = threading.RLock()
        self._session = ContextVar("autellix_session", default=None)
        self._owner = os.getpid()
        atexit.register(self._exit_cleanup)

    def _check_open(self):
        if os.getpid() != self._owner:
            raise RuntimeError("create a new InferenceClient after forking")
        if self._closed:
            raise RuntimeError("client is closed")

    def _automatic_session(self):
        self._check_open()  # Check ownership before touching an inherited lock.
        with self._lock:
            self._check_open()
            if self._sid is None:
                self._sid = self._request("POST", "/sessions", {"program_id": self.program_id})["session_id"]
            return self._sid

    def close(self):
        if os.getpid() != self._owner:
            return
        with self._lock:
            if self._closed or os.getpid() != self._owner:
                return
            if self._sid is not None:
                self._request("DELETE", "/sessions/" + quote(self._sid, safe=""))
            self._sid = None
            self._closed = True
            atexit.unregister(self._exit_cleanup)

    def _exit_cleanup(self):
        try:
            self.close()
        except Exception:
            pass  # Interpreter exit cannot guarantee network availability.

    def __enter__(self):
        self._automatic_session()
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except Exception as cleanup_error:
            if exc is None:
                raise
            raise exc.with_traceback(traceback) from cleanup_error

    def _request(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = Request(self.base_url + path, data=data, method=method,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=self.timeout) as response:
            return json.load(response)

    @contextmanager
    def session(self, program_id=None):
        self._check_open()
        with self._lock:
            self._check_open()
            sid = self._request("POST", "/sessions", {"program_id": program_id})["session_id"]
        token = self._session.set(sid)
        error = None
        try:
            yield sid
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._session.reset(token)
            # A forked child must never end the parent's program.
            if os.getpid() == self._owner:
                try:
                    self._request("DELETE", "/sessions/" + quote(sid, safe=""))
                except Exception as cleanup_error:
                    if error is None:
                        raise
                    raise error from cleanup_error

    def chat(self, messages, *, session_id=None, **options):
        self._check_open()
        session_id = session_id or self._session.get() or self._automatic_session()
        options.setdefault("thread_id", str(threading.get_ident()))
        options.setdefault("call_id", uuid.uuid4().hex)
        return self._request("POST", "/v1/chat/completions",
                             dict(messages=messages, session_id=session_id, **options))
