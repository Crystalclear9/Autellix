from __future__ import annotations

import json
import math
import sqlite3
import time
import threading
from functools import wraps
from dataclasses import asdict, dataclass, field
from pathlib import Path


def serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return call


@dataclass(frozen=True)
class PolicyConfig:
    policy: str = "atlas"
    boundaries: tuple[float, ...] = (0, .02, .04, .08, .16, .32, .64, math.inf)
    quanta: tuple[float, ...] = (.01, .02, .04, .08, .16, .32, .64)
    beta: float = 8.0
    schedule_interval: int = 1
    overprovision: int = 0

    def __post_init__(self):
        if self.policy not in {"fcfs", "mlfq", "plas", "atlas"}:
            raise ValueError("policy must be fcfs, mlfq, plas, or atlas")
        if (len(self.boundaries) != len(self.quanta) + 1 or
                self.boundaries[0] != 0 or self.boundaries[-1] != math.inf or
                any(math.isnan(b) for b in self.boundaries) or
                any(a >= b for a, b in zip(self.boundaries, self.boundaries[1:]))):
            raise ValueError("boundaries must strictly increase from 0 to infinity")
        if any(not math.isfinite(q) or q <= 0 for q in self.quanta):
            raise ValueError("quanta must be finite positive seconds")
        if not math.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("beta must be finite and positive")
        if (type(self.schedule_interval) is not int or type(self.overprovision) is not int or
                self.schedule_interval < 1 or self.overprovision < 0):
            raise ValueError("invalid schedule_interval or overprovision")

    def queue(self, service: float) -> int:
        return next(i for i, hi in enumerate(self.boundaries[1:]) if service < hi)


class ProgramTable:
    """Transactional program statistics shared by engine processes on one host.

    SQLite is used only at arrival/completion and policy refresh boundaries.
    Store this database on a local Linux filesystem, not an NFS/WSL mount.
    Session IDs are unique: callers must explicitly open and close sessions.
    """

    def __init__(self, path: str):
        self.path = path
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, timeout=30, isolation_level=None,
                                  check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS programs (
          pid TEXT PRIMARY KEY, service REAL NOT NULL DEFAULT 0,
          wait REAL NOT NULL DEFAULT 0, active INTEGER NOT NULL DEFAULT 0,
          closing INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS requests (
          rid TEXT PRIMARY KEY, pid TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS results (rid TEXT PRIMARY KEY, metrics TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS contexts (rid TEXT PRIMARY KEY, context TEXT NOT NULL);
        """)

    @serialized
    def open(self, pid: str):
        if not pid:
            raise ValueError("program ID must not be empty")
        self.db.execute("INSERT INTO programs(pid) VALUES (?)", (pid,))

    @serialized
    def admit(self, pid: str, rid: str) -> tuple[float, float]:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT service,wait,closing FROM programs WHERE pid=?", (pid,)).fetchone()
            if row is None or row[2]:
                raise ValueError(f"session {pid!r} is not open")
            self.db.execute("INSERT INTO requests VALUES (?,?)", (rid, pid))
            self.db.execute("UPDATE programs SET active=active+1 WHERE pid=?", (pid,))
            self.db.execute("COMMIT")
            return row[0], row[1]
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    @serialized
    def snapshot(self) -> dict[str, tuple[float, float]]:
        return {pid: (service, wait) for pid, service, wait in
                self.db.execute("SELECT pid,service,wait FROM programs")}

    @serialized
    def finish(self, pid: str, rid: str, inherited: float, executed: float,
               waited: float, policy: str):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            removed = self.db.execute("DELETE FROM requests WHERE rid=? AND pid=?", (rid, pid)).rowcount
            if removed:
                expression = "max(service, ?)" if policy == "atlas" else "service + ?"
                amount = inherited + executed if policy == "atlas" else executed
                self.db.execute(f"UPDATE programs SET service={expression}, wait=wait+?, active=active-1 WHERE pid=?",
                                (amount, waited, pid))
                self.db.execute("DELETE FROM programs WHERE pid=? AND closing=1 AND active=0", (pid,))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    @serialized
    def end(self, pid: str):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute("UPDATE programs SET closing=1 WHERE pid=?", (pid,))
            self.db.execute("DELETE FROM programs WHERE pid=? AND active=0", (pid,))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    @serialized
    def save_metrics(self, rid, metrics):
        self.db.execute("INSERT OR REPLACE INTO results VALUES (?,?)", (rid, json.dumps(metrics)))

    @serialized
    def set_context(self, rid, context):
        self.db.execute("INSERT INTO contexts VALUES (?,?)", (rid, json.dumps(context)))

    @serialized
    def take_context(self, rid):
        row = self.db.execute("SELECT context FROM contexts WHERE rid=?", (rid,)).fetchone()
        if row:
            self.db.execute("DELETE FROM contexts WHERE rid=?", (rid,))
        return json.loads(row[0]) if row else {}

    @serialized
    def take_metrics(self, rid):
        row = self.db.execute("SELECT metrics FROM results WHERE rid=?", (rid,)).fetchone()
        if row:
            self.db.execute("DELETE FROM results WHERE rid=?", (rid,))
            return json.loads(row[0])
        return None

    @serialized
    def close(self):
        self.db.close()


@dataclass
class RuntimeCall:
    rid: str
    pid: str
    inherited: float
    queue: int
    order: int
    quantum: float
    last_update: float
    executed: float = 0.0
    waited: float = 0.0
    run_window: float = 0.0
    wait_window: float = 0.0
    scheduled: bool = False
    metadata: dict = field(default_factory=dict)


class RuntimeScheduler:
    """Online Algorithm 1; all service/quanta values are measured seconds.

    A batched model-executor duration is charged to each participating call,
    matching elapsed model service rather than dividing time by batch size.
    Queue movement never resets lifetime execution statistics.
    """

    def __init__(self, table: ProgramTable, config: PolicyConfig | None = None,
                 trace: str | None = None, clock=time.monotonic):
        self.table = table
        self.config = config or PolicyConfig()
        self.calls: dict[str, RuntimeCall] = {}
        self.clock = clock
        self.counter = 0
        self.trace = trace
        if trace:
            Path(trace).parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields):
        if self.trace:
            with open(self.trace, "a", encoding="utf-8") as f:
                f.write(json.dumps({"event": event, "time": self.clock(), **fields}) + "\n")

    def admit(self, rid: str, pid: str, metadata=None):
        if rid in self.calls:
            raise ValueError(f"duplicate request {rid}")
        service, _ = self.table.admit(pid, rid)
        initial = service if self.config.policy in {"plas", "atlas"} else 0
        q = self.config.queue(initial)
        self.counter += 1
        self.calls[rid] = RuntimeCall(rid, pid, initial, q, self.counter,
                                      self.config.quanta[q], self.clock(), metadata=dict(metadata or {}))
        self.emit("admit", rid=rid, pid=pid, inherited=initial, queue=q, metadata=metadata or {})

    def _wait_until(self, call, now):
        delta = max(0., now - call.last_update)
        if not call.scheduled:
            call.waited += delta
            call.wait_window += delta
        call.last_update = now

    def refresh(self):
        now = self.clock()
        totals = self.table.snapshot()
        for call in self.calls.values():
            self._wait_until(call, now)
            if self.config.policy == "fcfs":
                continue
            if call.quantum <= 0:
                call.queue = min(call.queue + 1, len(self.config.quanta) - 1)
                call.quantum = self.config.quanta[call.queue]
                self.counter += 1
                call.order = self.counter
                self.emit("demote", rid=call.rid, queue=call.queue)
            service, wait = totals.get(call.pid, (0., 0.))
            if self.config.policy == "mlfq":
                service, wait = 0., 0.
            if call.queue and (wait + call.wait_window) / max(1e-9, service + call.run_window) >= self.config.beta:
                call.queue = 0
                call.quantum = self.config.quanta[0]
                call.wait_window = call.run_window = 0.
                self.counter += 1
                call.order = self.counter
                self.emit("promote", rid=call.rid, queue=0)

    def key(self, rid):
        call = self.calls[rid]
        return call.queue, call.order

    def begin(self, rids):
        now = self.clock()
        selected = set(rids)
        for call in self.calls.values():
            self._wait_until(call, now)
            call.scheduled = call.rid in selected
        self.emit("batch", requests=list(rids))

    def executed(self, rids, seconds: float):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("execution time must be finite and nonnegative")
        now = self.clock()
        for rid in rids:
            call = self.calls[rid]
            call.executed += seconds
            call.run_window += seconds
            call.quantum -= seconds
            call.scheduled = False
            call.last_update = now
        self.emit("execute", requests=list(rids), seconds=seconds)

    def finish(self, rid: str, reason="finished") -> dict:
        call = self.calls.pop(rid, None)
        if call is None:
            return {}
        self._wait_until(call, self.clock())
        self.table.finish(call.pid, rid, call.inherited, call.executed,
                          call.waited, self.config.policy)
        metrics = asdict(call)
        self.emit(reason, **metrics)
        return metrics

    def fail_all(self):
        for rid in list(self.calls):
            self.finish(rid, "failed")
