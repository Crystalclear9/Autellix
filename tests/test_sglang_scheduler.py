"""Native scheduler lifecycle regressions; GPU generation is tested separately."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from integrations.sglang.scheduler import encode_request, install


class CompletedDecodeTests(unittest.TestCase):
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
                        return len(self.running_batch.reqs)

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
                    self.assertEqual(scheduler.get_new_batch_prefill(), 1)
                    self.assertEqual(batch.reqs, reqs[1:2])
                    self.assertFalse(batch.batch_is_full)
                finally:
                    ctl.table.close()


if __name__ == "__main__":
    unittest.main()
