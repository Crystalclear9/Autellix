"""Native scheduler lifecycle regressions; GPU generation is tested separately."""
from contextlib import closing
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from integrations.sglang.scheduler import encode_request, install


class CompletedDecodeTests(unittest.TestCase):
    def test_single_stage_accounts_deferred_gpu_time_before_finish(self):
        for event_ready in (True, False):
            with self.subTest(event_ready=event_ready), tempfile.TemporaryDirectory() as directory:
                start = Mock()
                start.elapsed_time.return_value = 25.
                end = Mock()
                end.query.return_value = event_ready

                class Native:
                    def __init__(self):
                        self.policy = SimpleNamespace()

                    def process_batch_result(self, batch, result):
                        end.synchronize.assert_not_called()
                        batch.reqs[0].done = True
                        return result

                    _add_request_to_queue = Mock()
                    _extend_requests_to_queue = Mock()
                    get_new_batch_prefill = Mock()
                    run_batch = Mock()
                    abort_request = Mock()
                    check_memory = Mock()

                torch = SimpleNamespace(cuda=SimpleNamespace(Event=Mock(side_effect=[start, end])))
                with patch.dict("sys.modules", {"torch": torch}):
                    install(Native, {"table_path": str(Path(directory, "state.sqlite")), "config": {}})
                scheduler = Native()
                with closing(scheduler.autellix.table) as table:
                    table.open("program")
                    request = SimpleNamespace(rid="internal", done=False)
                    request.finished = lambda: request.done
                    scheduler.autellix.admit(request.rid, "program")
                    scheduler._autellix_rids[request.rid] = "external"
                    batch = SimpleNamespace(reqs=[request], input_ids=Mock())
                    result = scheduler.run_batch(batch)
                    self.assertEqual(scheduler.autellix.calls[request.rid].executed, 0.)
                    self.assertIs(scheduler.process_batch_result(batch, result), result)
                    self.assertEqual(end.synchronize.call_count, 0 if event_ready else 1)
                    self.assertEqual(table.take_metrics("external")["executed"], .025)

    def test_per_step_prefill_budget_preserves_the_decode_cohort(self):
        for admission_limit in ("partial", "chunk_budget", "kv_capacity"):
            with self.subTest(admission_limit=admission_limit):
                self._check_prefill_admission_limit(admission_limit)

    def _check_prefill_admission_limit(self, admission_limit):
        with tempfile.TemporaryDirectory() as directory:
            class Native:
                def __init__(self):
                    self.policy = SimpleNamespace()
                    self.server_args = SimpleNamespace(max_running_requests=3)
                    self.max_prefill_tokens, self.chunked_prefill_size = 16, 4
                    self.token_to_kv_pool_allocator = SimpleNamespace(available_size=lambda: 2048)
                    self.tree_cache = SimpleNamespace(evictable_size=lambda: 0, inc_lock_ref=Mock(), dec_lock_ref=Mock())
                    self.waiting_queue = []
                    self.chunked_req = None
                _add_request_to_queue = Mock()
                _extend_requests_to_queue = Mock()
                run_batch = Mock()
                process_batch_result = Mock()
                abort_request = Mock()
                check_memory = Mock()

                def get_new_batch_prefill(self):
                    if self.chunked_req is not None:
                        request, self.chunked_req = self.chunked_req, None
                    else:
                        request = self.waiting_queue.pop(0)
                        if request.rid == "first":
                            if admission_limit == "partial":
                                self.chunked_req = request
                            elif admission_limit == "kv_capacity":
                                self.running_batch.batch_is_full = True
                    return SimpleNamespace(reqs=[request, *self.running_batch.reqs])

            with patch.dict("sys.modules", {"torch": Mock()}):
                install(Native, {"table_path": str(Path(directory, "state.sqlite")),
                                 "config": {"overprovision": 0}, "capacity": 3})
            scheduler = Native()
            with closing(scheduler.autellix.table) as table:
                table.open("p")
                first, second, running = (SimpleNamespace(
                    rid=rid, output_ids=[], sampling_params=SimpleNamespace(max_new_tokens=8),
                    extend_input_len=4, init_next_round_input=lambda _: None, last_node=None)
                    for rid in ("first", "second", "running"))
                for request in (first, second, running):
                    scheduler.autellix.admit(request.rid, "p")
                scheduler.waiting_queue = [first, second]
                scheduler.running_batch = SimpleNamespace(reqs=[running], filter_batch=lambda: None)
                self.assertEqual(scheduler.get_new_batch_prefill().reqs, [first, running])
                window = scheduler._autellix_window
                if admission_limit == "kv_capacity":
                    self.assertEqual(window.cohort, ["first"])
                    self.assertTrue(window.capacity_limited)
                    return
                self.assertEqual(window.cohort, ["first", "second", "running"])
                self.assertFalse(window.capacity_limited)
                if admission_limit == "partial":
                    self.assertEqual(scheduler.get_new_batch_prefill().reqs, [first, running])
                scheduler.running_batch.reqs = [first, running]
                self.assertEqual(scheduler.get_new_batch_prefill().reqs, [second, first, running])

    def test_rotating_reserve_uses_the_configured_paused_kv_policy(self):
        for cache_paused in (True, False):
            with self.subTest(cache_paused=cache_paused), tempfile.TemporaryDirectory() as directory:
                class Native:
                    def __init__(self):
                        self.policy = SimpleNamespace()
                        self.server_args = SimpleNamespace(max_running_requests=2)
                        self.running_batch = SimpleNamespace(reqs=[], filter_batch=lambda: None)
                    _add_request_to_queue = Mock()
                    _extend_requests_to_queue = Mock()
                    run_batch = Mock()
                    process_batch_result = Mock()
                    abort_request = Mock()
                    check_memory = Mock()

                    def get_new_batch_prefill(self):
                        result = SimpleNamespace(reqs=list(self.waiting_queue))
                        self.waiting_queue = []
                        return result

                with patch.dict("sys.modules", {"torch": Mock()}):
                    install(Native, {"table_path": str(Path(directory, "state.sqlite")),
                                     "config": {"overprovision": 1}, "capacity": 1,
                                     "cache_paused_kv": cache_paused})
                scheduler = Native()
                with closing(scheduler.autellix.table) as table:
                    table.open("p")
                    scheduler.waiting_queue = [SimpleNamespace(rid=rid, last_node=rid, _autellix_reserved_tokens=[7],
                                                               init_next_round_input=lambda _: None)
                                               for rid in ("first", "next", "old-reserve")]
                    for req in scheduler.waiting_queue:
                        scheduler.autellix.admit(req.rid, "p")
                    scheduler._autellix_reserve["old-reserve"] = "old-reserve"
                    scheduler._autellix_host = SimpleNamespace(saved={}, save=Mock(return_value=True), discard=Mock())
                    scheduler.tree_cache = SimpleNamespace(inc_lock_ref=Mock(), dec_lock_ref=Mock(),
                        match_prefix=lambda _: SimpleNamespace(device_indices=[8]))

                    result = scheduler.get_new_batch_prefill()
                    self.assertEqual([req.rid for req in result.reqs], ["first", "next"])
                    self.assertEqual([req.rid for req in scheduler.waiting_queue], ["old-reserve"])
                    self.assertEqual(scheduler._autellix_reserve, {})
                    self.assertEqual(scheduler.tree_cache.dec_lock_ref.call_args_list.count((("old-reserve",), {})), 1)
                    self.assertEqual(scheduler._autellix_host.save.call_count, 0 if cache_paused else 1)

    def test_promoted_lower_reserve_yields_to_highest_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            locks = {"lower"}
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
                    result = SimpleNamespace(reqs=list(self.waiting_queue))
                    self.waiting_queue = []
                    return result
            with patch.dict("sys.modules", {"torch": Mock()}):
                install(Native, {"table_path": str(Path(directory, "state.sqlite")), "config": {}, "capacity": 2})
            scheduler = Native()
            try:
                scheduler.autellix.table.open("p")
                for rid in ("higher", "lower"):
                    scheduler.autellix.admit(rid, "p")
                scheduler.running_batch = SimpleNamespace(reqs=[], filter_batch=lambda: None)
                scheduler.waiting_queue = [SimpleNamespace(rid="higher"),
                    SimpleNamespace(rid="lower", _autellix_reserved_tokens=["lower"])]
                scheduler._autellix_reserve["lower"] = "lower"
                saved = {"higher": (["higher"], None)}
                scheduler._autellix_host = SimpleNamespace(saved=saved,
                    save=lambda rid, tokens, indices: saved.update({rid: (tokens, None)}),
                    restore=lambda rid: rid == "higher" and "lower" not in locks,
                    discard=lambda rid: saved.pop(rid))
                scheduler.tree_cache = SimpleNamespace(
                    match_prefix=lambda tokens: SimpleNamespace(device_indices=tokens, last_device_node=tokens[0]),
                    inc_lock_ref=locks.add, dec_lock_ref=locks.remove)
                result = scheduler.get_new_batch_prefill()
                self.assertEqual([r.rid for r in result.reqs], ["higher"])
                self.assertEqual(scheduler._autellix_window.cohort, ["higher"])
                self.assertEqual(scheduler._autellix_reserve, {})
                self.assertEqual(set(saved), {"lower"})
                self.assertEqual(locks, set())
            finally:
                scheduler.autellix.table.close()

    def test_lower_restoration_cannot_block_new_higher_prefill(self):
        for cached, host_restore in ((0, True), (3, True), (3, False)):
            with self.subTest(cached=cached, host_restore=host_restore), tempfile.TemporaryDirectory() as directory:
                locks, resident, attempts = set(), {"higher"}, []
                lengths = {"higher": cached, "lower": 5}
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
                        attempts.append([r.rid for r in self.waiting_queue])
                        if "higher" not in resident or ("higher" in locks) != host_restore:
                            raise AssertionError("unexpected temporary lock on the new higher call")
                        available = 10 - sum(lengths[r] for r in locks)
                        if 8 - cached >= available:
                            return None
                        result = SimpleNamespace(reqs=list(self.waiting_queue))
                        self.waiting_queue = []
                        return result
                with patch.dict("sys.modules", {"torch": Mock()}):
                    install(Native, {"table_path": str(Path(directory, "state.sqlite")),
                                     "config": {"overprovision": 0}, "capacity": 2})
                scheduler = Native()
                try:
                    scheduler.autellix.table.open("p")
                    for rid in ("higher", "lower"):
                        scheduler.autellix.admit(rid, "p")
                    scheduler.running_batch = SimpleNamespace(reqs=[], filter_batch=lambda: None)
                    scheduler.waiting_queue = [SimpleNamespace(rid="higher", last_node="higher",
                        init_next_round_input=lambda tree: None), SimpleNamespace(rid="lower")]
                    saved = {"lower": (["lower"]*5, None)} if host_restore else {}
                    def restore(rid):
                        if "higher" not in locks:
                            resident.discard("higher")
                        resident.add(rid)
                        return True
                    scheduler._autellix_host = SimpleNamespace(saved=saved, restore=restore,
                                                               discard=lambda rid: saved.pop(rid, None))
                    scheduler.tree_cache = SimpleNamespace(
                        match_prefix=lambda tokens: SimpleNamespace(device_indices=tokens, last_device_node=tokens[0]),
                        inc_lock_ref=locks.add, dec_lock_ref=locks.remove)
                    result = scheduler.get_new_batch_prefill()
                    admitted = ["higher"] if host_restore else ["higher", "lower"]
                    self.assertEqual([r.rid for r in result.reqs], admitted)
                    self.assertEqual(attempts, [["higher", "lower"], ["higher"]] if host_restore else [admitted])
                    self.assertEqual([r.rid for r in scheduler.waiting_queue], ["lower"] if host_restore else [])
                    self.assertEqual(scheduler._autellix_window.cohort, admitted)
                    self.assertEqual(set(saved), {"lower"} if host_restore else set())
                    self.assertEqual(locks, set())
                finally:
                    scheduler.autellix.table.close()

    def test_selected_decode_stays_resident_when_higher_prefill_fits(self):
        with tempfile.TemporaryDirectory() as directory:
            class Native:
                def __init__(self):
                    self.policy = SimpleNamespace()
                    self.server_args = SimpleNamespace(max_running_requests=2)
                    self.max_prefill_tokens = 8
                    self.token_to_kv_pool_allocator = SimpleNamespace(available_size=lambda: 2048)
                    self.tree_cache = SimpleNamespace(evictable_size=lambda: 0, inc_lock_ref=Mock(), dec_lock_ref=Mock())
                    self.waiting_queue = []
                _add_request_to_queue = Mock()
                _extend_requests_to_queue = Mock()
                run_batch = Mock()
                process_batch_result = Mock()
                abort_request = Mock()
                check_memory = Mock()
                def get_new_batch_prefill(self):
                    result = SimpleNamespace(reqs=self.waiting_queue + self.running_batch.reqs)
                    self.waiting_queue = []
                    return result
            with patch.dict("sys.modules", {"torch": Mock()}):
                install(Native, {"table_path": str(Path(directory, "state.sqlite")),
                                 "config": {"overprovision": 0}, "capacity": 2})
            scheduler = Native()
            try:
                scheduler.autellix.table.open("p")
                for rid in ("higher", "running"):
                    scheduler.autellix.admit(rid, "p")
                req = lambda rid: SimpleNamespace(rid=rid, origin_input_ids=[1]*10, output_ids=[2],
                                                 sampling_params=SimpleNamespace(max_new_tokens=8),
                                                 extend_input_len=2, init_next_round_input=lambda _: None, last_node=None)
                scheduler.waiting_queue = [req("higher")]
                scheduler.running_batch = SimpleNamespace(reqs=[req("running")], filter_batch=lambda: None,
                    seq_lens=SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: [10])))
                result = scheduler.get_new_batch_prefill()
                self.assertEqual([r.rid for r in result.reqs], ["higher", "running"])
                self.assertEqual([r.rid for r in scheduler.running_batch.reqs], ["running"])
            finally:
                scheduler.autellix.table.close()

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
