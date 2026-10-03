"""Native scheduler lifecycle regressions; GPU generation is tested separately."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from integrations.sglang.scheduler import encode_request, install


class CompletedDecodeTests(unittest.TestCase):
    def test_restored_prefix_is_pinned_through_admission_and_unlocked_on_error(self):
        for fail_admission in (False, True):
            with self.subTest(fail_admission=fail_admission), tempfile.TemporaryDirectory() as directory:
                resident, locks, admitted = set(), set(), []
                class Native:
                    def __init__(self):
                        self.policy = SimpleNamespace()
                        self.server_args = SimpleNamespace(max_running_requests=2)
                        self.waiting_queue = []
                    _add_request_to_queue = Mock()
                    _extend_requests_to_queue = Mock()
                    run_batch = Mock()
                    process_batch_result = Mock()
                    abort_request = Mock()
                    check_memory = Mock()
                    def get_new_batch_prefill(self):
                        if resident != {"a"} or locks != {"a"}:
                            raise AssertionError("higher-priority restored prefix was evicted")
                        if fail_admission:
                            raise ValueError("native admission failed")
                        admitted.extend(r.rid for r in self.waiting_queue)
                        result = SimpleNamespace(reqs=list(self.waiting_queue))
                        self.waiting_queue = []
                        return result
                with patch.dict("sys.modules", {"torch": Mock()}):
                    install(Native, {"table_path": str(Path(directory, "state.sqlite")),
                                     "config": {"overprovision": 0}, "capacity": 2})
                scheduler = Native()
                try:
                    scheduler.autellix.table.open("p")
                    for rid in ("a", "b"):
                        scheduler.autellix.admit(rid, "p")
                    scheduler.running_batch = SimpleNamespace(reqs=[], filter_batch=lambda: None)
                    scheduler.waiting_queue = [SimpleNamespace(rid="a"), SimpleNamespace(rid="b")]
                    # One-prefix GPU capacity: restoring b would evict a unless
                    # a stays pinned across the entire restoration/admission pass.
                    def restore(rid):
                        if resident and locks:
                            return False
                        resident.clear()
                        resident.add(rid)
                        return True
                    saved = {r: ([r], None) for r in ("a", "b")}
                    scheduler._autellix_host = SimpleNamespace(
                        saved=saved, restore=restore, discard=lambda rid: saved.pop(rid))
                    scheduler.tree_cache = SimpleNamespace(
                        match_prefix=lambda tokens: SimpleNamespace(
                            device_indices=tokens if tokens[0] in resident else [], last_device_node=tokens[0]),
                        inc_lock_ref=locks.add, dec_lock_ref=locks.remove)
                    if fail_admission:
                        with self.assertRaisesRegex(ValueError, "native admission failed"):
                            scheduler.get_new_batch_prefill()
                        self.assertEqual(set(saved), {"a", "b"})
                    else:
                        scheduler.get_new_batch_prefill()
                        self.assertEqual(admitted, ["a"])
                        self.assertEqual(set(saved), {"b"})
                    self.assertEqual(locks, set())
                    self.assertIn("b", [r.rid for r in scheduler.waiting_queue])
                finally:
                    scheduler.autellix.table.close()

    def test_reserve_pin_survives_until_native_admission(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            class Native:
                def __init__(self):
                    self.policy = SimpleNamespace()
                    self.server_args = SimpleNamespace(max_running_requests=1)
                    self.waiting_queue = []
                _add_request_to_queue = Mock()
                _extend_requests_to_queue = Mock()
                run_batch = Mock()
                process_batch_result = Mock()
                abort_request = Mock()
                check_memory = Mock()
                def get_new_batch_prefill(self):
                    if "r" not in self._autellix_reserve:
                        raise AssertionError("reserve became evictable before native admission")
                    events.append("native_lock")
                    result = SimpleNamespace(reqs=list(self.waiting_queue))
                    self.waiting_queue = []
                    return result
            with patch.dict("sys.modules", {"torch": Mock()}):
                install(Native, {"table_path": str(Path(directory, "state.sqlite")),
                                 "config": {}, "capacity": 1})
            scheduler = Native()
            try:
                scheduler.autellix.table.open("p")
                scheduler.autellix.admit("r", "p")
                scheduler.running_batch = SimpleNamespace(reqs=[], filter_batch=lambda: None)
                scheduler.waiting_queue = [SimpleNamespace(rid="r")]
                scheduler._autellix_reserve["r"] = "node"
                scheduler.tree_cache = SimpleNamespace(dec_lock_ref=lambda _: events.append("release_reserve_lock"))
                scheduler.get_new_batch_prefill()
                self.assertEqual(events, ["native_lock", "release_reserve_lock"])
                self.assertEqual(scheduler._autellix_reserve, {})
            finally:
                scheduler.autellix.table.close()

    def test_failed_restore_stops_admission_after_successful_prefix(self):
        for running in (False, True):
            with self.subTest(running=running), tempfile.TemporaryDirectory() as directory:
                admitted = []
                class Native:
                    def __init__(self):
                        self.policy = SimpleNamespace()
                        self.server_args = SimpleNamespace(max_running_requests=3)
                        self.waiting_queue = []
                    _add_request_to_queue = Mock()
                    _extend_requests_to_queue = Mock()
                    run_batch = Mock()
                    process_batch_result = Mock()
                    abort_request = Mock()
                    check_memory = Mock()
                    def get_new_batch_prefill(self):
                        admitted.extend(r.rid for r in self.waiting_queue)
                        result = SimpleNamespace(reqs=list(self.waiting_queue)) if self.waiting_queue else None
                        self.waiting_queue = []
                        return result
                with patch.dict("sys.modules", {"torch": Mock()}):
                    install(Native, {"table_path": str(Path(directory, "state.sqlite")),
                                     "config": {"overprovision": 0}, "capacity": 3})
                scheduler = Native()
                ctl = scheduler.autellix
                try:
                    ctl.table.open("p")
                    for rid in ("higher", "a", "b"):
                        ctl.admit(rid, "p")
                    scheduler.running_batch = SimpleNamespace(
                        reqs=[SimpleNamespace(rid="higher")] if running else [],
                        filter_batch=lambda: None, batch_is_full=False,
                        seq_lens=SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: [5])))
                    scheduler.waiting_queue = [SimpleNamespace(rid="a"), SimpleNamespace(rid="b")]
                    scheduler._autellix_host = SimpleNamespace(
                        saved={"a": ([1], None), "b": ([2], None)}, discard=lambda _: None,
                        restore=lambda rid: rid == "a" and not running)
                    scheduler.tree_cache = SimpleNamespace(
                        match_prefix=lambda tokens: SimpleNamespace(device_indices=tokens, last_device_node="node"),
                        inc_lock_ref=lambda _: None, dec_lock_ref=lambda _: None)
                    scheduler.get_new_batch_prefill()
                    self.assertEqual(admitted, [] if running else ["a"])
                    self.assertEqual([r.rid for r in scheduler.waiting_queue], ["a", "b"] if running else ["b"])
                    self.assertEqual(scheduler._autellix_window.cohort, ["higher"] if running else ["a"])
                finally:
                    ctl.table.close()

    def test_prepared_window_bypasses_native_prefill_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            class Native:
                def __init__(self):
                    self.policy = SimpleNamespace()
                    self.server_args = SimpleNamespace(max_running_requests=1)
                    self.waiting_queue = []
                _add_request_to_queue = Mock()
                _extend_requests_to_queue = Mock()
                run_batch = Mock()
                process_batch_result = Mock()
                abort_request = Mock()
                check_memory = Mock()
                def get_new_batch_prefill(self):
                    raise AssertionError("native admission ran inside prepared window")
            with patch.dict("sys.modules", {"torch": Mock()}):
                install(Native, {"table_path": str(Path(directory, "state.sqlite")),
                                 "config": {"schedule_interval": 3, "overprovision": 0}, "capacity": 1})
            scheduler = Native()
            ctl = scheduler.autellix
            try:
                ctl.table.open("p")
                ctl.admit("active", "p")
                ctl.admit("waiting", "p")
                scheduler._autellix_window.select(["active", "waiting"], ["active"])
                scheduler._autellix_window.executed()
                scheduler.running_batch = SimpleNamespace(reqs=[SimpleNamespace(rid="active")],
                                                          filter_batch=lambda: None)
                scheduler.waiting_queue = [SimpleNamespace(rid="waiting")]
                self.assertIsNone(scheduler.get_new_batch_prefill())
                self.assertEqual(scheduler.waiting_queue[0].rid, "waiting")
                self.assertTrue(scheduler.is_mixed_chunk)
            finally:
                ctl.table.close()

    def test_retired_decode_is_filtered_before_priority_and_slot_counting(self):
        for step in (0, 1):  # Policy refresh and the intervening multi-step iteration.
            with self.subTest(step=step), tempfile.TemporaryDirectory() as directory:
                class NativeScheduler:
                    def __init__(self):
                        self.policy = SimpleNamespace()
                        self.server_args = SimpleNamespace(max_running_requests=2)
                        self.waiting_queue = []

                    _add_request_to_queue = Mock()
                    _extend_requests_to_queue = Mock()
                    run_batch = Mock()
                    process_batch_result = Mock()
                    abort_request = Mock()
                    check_memory = Mock()

                    def get_new_batch_prefill(self):
                        # Native prefill refuses admission while this flag is set.
                        if self.running_batch.batch_is_full:
                            return None
                        self.policy.calc_priority(self.waiting_queue)
                        return SimpleNamespace(reqs=list(self.running_batch.reqs))

                with patch.dict("sys.modules", {"torch": Mock()}):
                    install(NativeScheduler, {
                        "table_path": str(Path(directory, "state.sqlite")),
                        "config": {"schedule_interval": 3},
                    })
                scheduler = NativeScheduler()
                ctl = scheduler.autellix
                try:
                    ctl.table.open("program")
                    reqs = []
                    for rid in ("finished", "running", "waiting"):
                        internal = encode_request("program", rid)
                        ctl.admit(internal, "program")
                        reqs.append(SimpleNamespace(rid=internal, finished=lambda r=rid: r == "finished"))
                    ctl.finish(reqs[0].rid)  # process_batch_result already retired it.
                    batch = SimpleNamespace(reqs=reqs[:2], batch_is_full=True,
                                            seq_lens=SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: [5])))
                    def filter_batch():
                        batch.reqs = [r for r in batch.reqs if not r.finished()]
                    batch.filter_batch = filter_batch
                    scheduler.running_batch = batch
                    scheduler.waiting_queue = reqs[2:]
                    scheduler._autellix_steps = step
                    self.assertEqual(len(scheduler.get_new_batch_prefill().reqs), 1)
                    self.assertEqual(batch.reqs, reqs[1:2])
                    self.assertFalse(batch.batch_is_full)
                finally:
                    ctl.table.close()


if __name__ == "__main__":
    unittest.main()
