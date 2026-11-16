"""Three-GPU pipeline generation, swap/resume, accounting and cancellation."""
import json
import math
import os
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path

from autellix.runtime import InferenceEngine, PolicyConfig, ReplicaConfig


@unittest.skipUnless(os.environ.get("AUTELLIX_TEST_PP_MODEL"), "requires AUTELLIX_TEST_PP_MODEL and three GPUs")
class PipelineGPUTests(unittest.TestCase):
    def test_three_stages_swap_resume_history_and_cancel(self):
        options = dict(autellix_legacy_gpu=True, dtype="float32", pp_size=3,
                       context_length=512, max_total_tokens=2048,
                       max_running_requests=3, mem_fraction_static=.35, random_seed=0)
        policy = PolicyConfig(boundaries=(0, .001, .002, math.inf), quanta=(.0001, .0002, .0004),
                              beta=1e12, schedule_interval=2, overprovision=0)
        with tempfile.TemporaryDirectory(prefix="autellix-pp-test-") as directory:
            with InferenceEngine([ReplicaConfig("sglang", os.environ["AUTELLIX_TEST_PP_MODEL"],
                                                "0,1,2", options)], policy=policy, state_dir=directory) as engine:
                tokens = engine.tokenize(prompt="The capital of France is")
                sessions = [engine.start_session() for _ in range(8)]
                sampling = dict(temperature=0, max_tokens=32, ignore_eos=True)
                futures = [engine.submit(pid, input_ids=tokens, sampling=sampling) for pid in sessions]
                outputs = [future.result(240) for future in futures]
                self.assertTrue(all(row["completion_tokens"] == 32 for row in outputs))
                reference = engine.submit(sessions[0], input_ids=tokens, sampling=sampling).result(180)
                self.assertTrue(all(row["text"] == reference["text"] for row in outputs))
                self.assertGreater(reference["metrics"]["inherited"], 0)
                pending = [engine.submit(pid, input_ids=tokens,
                                         sampling=dict(temperature=0, max_tokens=128, ignore_eos=True))
                           for pid in sessions[:3]]
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    active = engine.table.describe()
                    if all(any(call["executed"] > 0 for call in active[pid]["calls"])
                           for pid in sessions[:3]):
                        break
                    time.sleep(.01)
                else:
                    self.fail("cancellation targets did not all execute on the GPU")
                for future in pending:
                    future.cancel()
                for pid in sessions:
                    engine.end_session(pid)
                engine.wait_idle(30)
                self.assertEqual(engine.table.describe(), {})
                events = [json.loads(line) for line in Path(directory, "replica-0.jsonl").read_text().splitlines()]
                counts = Counter(row["event"] for row in events)
                self.assertGreater(counts["swap_out"], 0)
                self.assertGreater(counts["swap_in"], 0)
                self.assertEqual(counts["cancelled"], 3)
                selected = {"batch", "execute", "finished", "cancelled", "swap_out", "swap_in"}
                expected = [row for row in events if row["event"] in selected]
                for rank in [1, 2]:
                    follower = [json.loads(line) for line in Path(directory, f"replica-0.jsonl.stage-{rank}").read_text().splitlines()]
                    self.assertEqual(follower, expected)
                print(json.dumps(dict(pp_size=3, greedy_resume_matches=True,
                                      inherited=reference["metrics"]["inherited"], trace_events=dict(counts),
                                      all_stage_traces_match=True, cancelled=counts["cancelled"])))


if __name__ == "__main__":
    unittest.main()
