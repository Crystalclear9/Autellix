from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
import os
import queue
import tempfile
import threading
import time
import uuid
import traceback
import json
from dataclasses import dataclass, field
from pathlib import Path

from .policy import PolicyConfig, ProgramTable


class InferenceFuture(concurrent.futures.Future):
    """Final result plus a bounded, coalesced stream of cumulative outputs."""
    def __init__(self):
        super().__init__()
        self.progress = queue.Queue(maxsize=1)
        self.submitted_at = time.monotonic()
        self.first_token_at = None

    def update(self, value):
        if self.first_token_at is None and value.get("text"):
            self.first_token_at = time.monotonic()
        try:
            self.progress.put_nowait(value)
        except queue.Full:
            try:
                self.progress.get_nowait()
            except queue.Empty:
                pass
            self.progress.put_nowait(value)


@dataclass(frozen=True)
class ReplicaConfig:
    backend: str
    model: str
    device: str = "0"
    engine_args: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.backend not in {"vllm", "sglang"}:
            raise ValueError("backend must be vllm or sglang")


def _worker(index, replica, policy, table_path, trace, commands, events):
    # Set device visibility before importing either CUDA runtime.
    os.environ["CUDA_VISIBLE_DEVICES"] = replica.device
    backend = None
    try:
        if replica.backend == "vllm":
            from integrations.vllm.backend import VLLMBackend as Backend
        else:
            from integrations.sglang.backend import SGLangBackend as Backend
        backend = Backend(replica.model, table_path, policy, trace, **replica.engine_args)
        events.put((index, "ready", None, None))
        closing = False
        while not closing:
            # Bound admission work so a busy producer cannot starve generation.
            for _ in range(64):
                try:
                    command, rid, payload = commands.get(timeout=.02 if not backend.has_work() else 0)
                except queue.Empty:
                    break
                if command == "shutdown":
                    closing = True
                    break
                try:
                    if command == "add":
                        backend.add(rid=rid, **payload)
                    elif command == "cancel":
                        backend.cancel(rid)
                        events.put((index, "cancelled", rid, None))
                    elif command == "tokenize":
                        if "messages" in payload:
                            tokenizer = (backend.engine.get_tokenizer() if replica.backend == "vllm"
                                         else backend.engine.tokenizer_manager.tokenizer)
                            tokens = tokenizer.apply_chat_template(payload["messages"], tokenize=True,
                                                                   add_generation_prompt=True)
                        else:
                            tokens = backend.tokenize(payload["prompt"])
                        events.put((index, "tokenized", rid, tokens))
                except Exception as exc:
                    events.put((index, "error", rid, repr(exc)))
            if not closing and backend.has_work():
                for result in backend.step():
                    event = "error" if "error" in result else ("result" if result.get("finished", True) else "progress")
                    events.put((index, event, result["request_id"], result.get("error", result)))
        events.put((index, "stopped", None, None))
    except BaseException as exc:
        traceback.print_exc()
        events.put((index, "fatal", None, repr(exc)))
    finally:
        if backend is not None:
            backend.close()


class InferenceEngine:
    """Real replica coordinator. Futures are resolved exclusively by worker IPC.

    The coordinator never simulates model execution. Independent processes can
    target different GPUs (or share one GPU with explicit memory budgets).
    """

    def __init__(self, replicas: list[ReplicaConfig], *, policy: PolicyConfig | None = None,
                 state_dir: str | None = None, locality_threshold=2048):
        if not replicas:
            raise ValueError("at least one replica is required")
        if len({r.model for r in replicas}) != 1:
            raise ValueError("replicas must serve the same model and tokenizer")
        if locality_threshold < 0:
            raise ValueError("locality_threshold must be nonnegative")
        self.replicas = replicas
        self.model = replicas[0].model
        self.policy = policy or PolicyConfig()
        self.state_dir = Path(state_dir or tempfile.mkdtemp(prefix="autellix-"))
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.table = ProgramTable(str(self.state_dir / "programs.sqlite"))
        self.threshold = locality_threshold
        self.lock = threading.RLock()
        self.idle = threading.Condition(self.lock)
        self.sessions = {}
        self.pending = {}
        self.rpc = {}
        self.loads = [0] * len(replicas)
        self.failed = {}
        self.closed = False
        ctx = mp.get_context("spawn")
        self.events = ctx.Queue()
        self.commands = [ctx.Queue() for _ in replicas]
        self.ready = [concurrent.futures.Future() for _ in replicas]
        self.processes = []
        for index, replica in enumerate(replicas):
            process = ctx.Process(target=_worker, args=(index, replica, self.policy,
                self.table.path, str(self.state_dir / f"replica-{index}.jsonl"),
                self.commands[index], self.events), name=f"autellix-{index}")
            process.start()
            self.processes.append(process)
        self.listener = threading.Thread(target=self._listen, daemon=True, name="autellix-results")
        self.listener.start()

    def wait_ready(self, timeout=600):
        deadline = time.monotonic() + timeout
        for ready in self.ready:
            ready.result(timeout=max(0, deadline-time.monotonic()))
        return self

    def start_session(self, program_id=None):
        with self.lock:
            if self.closed:
                raise RuntimeError("engine is closed")
            pid = program_id or uuid.uuid4().hex
            self.table.open(pid)
            self.sessions[pid] = {"closing": False, "engine": None, "calls": set()}
            return pid

    def end_session(self, pid):
        with self.lock:
            session = self.sessions.get(pid)
            if session is None:
                return
            session["closing"] = True
            self._cleanup_session(pid)

    def _cleanup_session(self, pid):
        if (pid in self.sessions and self.sessions[pid]["closing"] and
                not any(item[1] == pid for item in self.pending.values())):
            self.table.end(pid)
            del self.sessions[pid]

    def tokenize(self, *, prompt=None, messages=None, timeout=120):
        if (prompt is None) == (messages is None):
            raise ValueError("supply prompt or messages")
        self.wait_ready(timeout)
        with self.lock:
            if self.closed:
                raise RuntimeError("engine is closed")
            available = [i for i in range(len(self.replicas)) if i not in self.failed]
            if not available:
                raise RuntimeError("no healthy replicas")
            index = min(available, key=lambda i: self.loads[i])
            rid = uuid.uuid4().hex
            future = concurrent.futures.Future()
            self.rpc[rid] = (future, index)
            payload = {"prompt": prompt} if prompt is not None else {"messages": messages}
            self.commands[index].put(("tokenize", rid, payload))
        try:
            return future.result(timeout)
        finally:
            with self.lock:
                self.rpc.pop(rid, None)

    def submit(self, pid, *, prompt=None, messages=None, input_ids=None, sampling=None, call_id=None,
               thread_id=None, metadata=None):
        if sum(x is not None for x in (prompt, messages, input_ids)) != 1:
            raise ValueError("supply exactly one of prompt, messages, input_ids")
        tokens = input_ids if input_ids is not None else self.tokenize(prompt=prompt, messages=messages)
        if not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("input_ids must be nonempty nonnegative integers")
        with self.lock:
            session = self.sessions.get(pid)
            if self.closed or session is None or session["closing"]:
                raise ValueError("session is not open")
            call_id = call_id or uuid.uuid4().hex
            if call_id in session["calls"]:
                raise ValueError("duplicate call ID in session")
            available = [i for i in range(len(self.replicas)) if i not in self.failed]
            if not available:
                raise RuntimeError("no healthy replicas")
            index = min(available, key=lambda i: (self.loads[i], i))
            if len(tokens) > self.threshold:
                if session["engine"] in available:
                    index = session["engine"]
                else:
                    session["engine"] = index
            rid = uuid.uuid4().hex
            context = dict(call_id=call_id, thread_id=thread_id, metadata=metadata or {})
            json.dumps(context)  # validate before admission or updating load
            self.table.set_context(rid, context)
            future = InferenceFuture()
            self.pending[rid] = (future, pid, index, call_id)
            session["calls"].add(call_id)
            self.loads[index] += 1
            self.commands[index].put(("add", rid, dict(pid=pid, prompt=list(tokens), sampling=sampling or {})))
            def on_done(done):
                if done.cancelled():
                    self.commands[index].put(("cancel", rid, {}))
            future.add_done_callback(on_done)
            return future

    def _resolve(self, rid, event, payload):
        if event == "progress":
            item = self.pending.get(rid)
            if item and not item[0].done():
                item[0].update(payload)
            return
        if rid in self.rpc:
            future, _ = self.rpc.pop(rid)
            if not future.done():
                if event == "tokenized":
                    future.set_result(payload)
                else:
                    future.set_exception(RuntimeError(str(payload)))
            return
        item = self.pending.pop(rid, None)
        if not item:
            return
        future, pid, index, call_id = item
        self.loads[index] -= 1
        self.table.take_context(rid)
        if event in {"error", "cancelled"}:
            # Reclaim admission bookkeeping even if a worker died mid-request.
            self.table.finish(pid, rid, 0, 0, 0, self.policy.policy)
            if self.replicas[index].backend == "sglang":
                from integrations.sglang.scheduler import encode_request
                self.table.finish(pid, encode_request(pid, rid), 0, 0, 0, self.policy.policy)
        if not future.done():
            if event == "result":
                payload.update(program_id=pid, call_id=call_id, engine_id=index)
                future.update(payload)
                payload["latency_seconds"] = time.monotonic() - future.submitted_at
                payload["ttft_seconds"] = (future.first_token_at - future.submitted_at
                                           if future.first_token_at is not None else None)
                future.set_result(payload)
            elif event == "cancelled":
                future.cancel()
            else:
                future.set_exception(RuntimeError(str(payload)))
        self._cleanup_session(pid)
        self.idle.notify_all()

    def wait_idle(self, timeout=600):
        """Wait for all results/cancellation acknowledgements and session cleanup."""
        deadline = time.monotonic() + timeout
        with self.idle:
            while self.pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("requests are still active")
                self.idle.wait(remaining)

    def _fail_replica(self, index, message):
        if index in self.failed:
            return
        self.failed[index] = message
        if not self.ready[index].done():
            self.ready[index].set_exception(RuntimeError(message))
        for rid, item in list(self.pending.items()):
            if item[2] == index:
                self._resolve(rid, "error", message)
        for rid, (future, owner) in list(self.rpc.items()):
            if owner == index:
                self._resolve(rid, "error", message)

    def _listen(self):
        while not self.closed:
            try:
                index, event, rid, payload = self.events.get(timeout=.1)
                with self.lock:
                    if event == "ready" and not self.ready[index].done():
                        self.ready[index].set_result(True)
                    elif event in {"fatal", "stopped"}:
                        self._fail_replica(index, str(payload or "replica stopped"))
                    else:
                        self._resolve(rid, event, payload)
            except queue.Empty:
                with self.lock:
                    for i, process in enumerate(self.processes):
                        if process.exitcode is not None:
                            self._fail_replica(i, f"replica exited with code {process.exitcode}")

    def shutdown(self):
        with self.lock:
            if self.closed:
                return
            for command in self.commands:
                command.put(("shutdown", None, {}))
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        with self.lock:
            for index in range(len(self.replicas)):
                self._fail_replica(index, "engine shutdown")
            for pid in list(self.sessions):
                self.end_session(pid)
            self.closed = True
        self.listener.join(timeout=2)
        self.table.close()
        for channel in self.commands + [self.events]:
            channel.close()
            channel.cancel_join_thread()

    def __enter__(self):
        try:
            return self.wait_ready()
        except BaseException:
            self.shutdown()
            raise

    def __exit__(self, *_):
        self.shutdown()
