import importlib.util
import copy
import json
import random
import sys
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest
from collections import Counter
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
try:
    spec = importlib.util.spec_from_file_location("matrix", Path(__file__).parents[1] / "scripts/run_sglang_matrix.py")
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
finally:
    sys.path.pop(0)


class MatrixResumeTests(unittest.TestCase):
    def test_timed_trace_uses_every_arrival_in_fixed_window(self):
        pool = [{"program_id": str(index)} for index in range(40)]
        trace = matrix.timed_trace(pool, 1.5, 27, 10)
        rng = random.Random(27)
        arrivals = []
        at = rng.expovariate(1.5)
        while at < 10:
            arrivals.append(at)
            at += rng.expovariate(1.5)
        self.assertEqual([row["arrival_time"] for row in trace], arrivals)
        self.assertEqual(len({row["program_id"] for row in trace}), len(arrivals))
        self.assertEqual(trace, matrix.timed_trace(pool, 1.5, 27, 10))

    def test_timed_mixed_trace_keeps_poisson_count_and_balances_sources(self):
        pool = [dict(program_id=f"{dataset}-{i}", dataset=dataset)
                for dataset in ("sharegpt", "bfcl", "lats") for i in range(20)]
        original = matrix.timed_trace(pool, 1.5, 27, 10)
        mixed = matrix.timed_trace(pool, 1.5, 27, 10, equal_dataset_counts=True)
        self.assertEqual([p["arrival_time"] for p in mixed], [p["arrival_time"] for p in original])
        sizes = Counter(p["dataset"] for p in mixed)
        self.assertEqual(len(sizes), 3)
        self.assertLessEqual(max(sizes.values()) - min(sizes.values()), 1)

    def test_distinct_mixed_trace_balances_datasets_at_every_rate(self):
        pool = [dict(program_id=f"{dataset}-{i}", dataset=dataset)
                for dataset, size in (("sharegpt", 9), ("bfcl", 7), ("lats", 3)) for i in range(size)]
        slow = matrix.unique_trace(pool, 9, .2, 29, equal_dataset_counts=True)
        fast = matrix.unique_trace(pool, 9, .4, 29, equal_dataset_counts=True)
        self.assertEqual(Counter(row["dataset"] for row in slow), {"sharegpt": 3, "bfcl": 3, "lats": 3})
        self.assertEqual(len({row["program_id"] for row in slow}), 9)
        for a, b in zip(slow, fast, strict=True):
            self.assertEqual(a["program_id"], b["program_id"])
            self.assertAlmostEqual(a["arrival_time"], 2 * b["arrival_time"])
        self.assertTrue(all("arrival_time" not in row for row in pool))
        with self.assertRaisesRegex(ValueError, "every dataset"):
            matrix.unique_trace(pool, 12, .2, 29, equal_dataset_counts=True)

    def test_poisson_window_can_exceed_the_number_of_source_programs(self):
        pool = [dict(program_id=kind, dataset=kind, calls=[dict(call_id="a", prompt="input")])
                for kind in ("sharegpt", "bfcl", "lats")]
        trace = matrix.timed_trace(pool, 1.5, 27, 10, equal_dataset_counts=True, unique_programs=False)
        self.assertGreater(len(trace), len(pool))
        self.assertEqual(len({row["program_id"] for row in trace}), len(trace))
        distribution = Counter(row["dataset"] for row in trace)
        self.assertEqual(len(distribution), 3)
        self.assertLessEqual(max(distribution.values()) - min(distribution.values()), 1)
        distinct_pool = [dict(program_id=str(i)) for i in range(40)]
        self.assertEqual([row["arrival_time"] for row in trace],
                         [row["arrival_time"] for row in matrix.timed_trace(distinct_pool, 1.5, 27, 10)])

    def test_future_rates_and_pending_case_can_change_without_repeating_native(self):
        plan = dict(model="8B", rates=[.02, .05, .1], engine_args=dict(max_total_tokens=32768),
                    cases=[dict(name="native", policy="sglang-native"), dict(name="plas", interval=8)])
        row = dict(condition="native", rate=.02, repeat=0, plan=plan, versions={"sglang": "pinned"})
        revised = copy.deepcopy(plan)
        revised["rates"] = [.005, .01, .02]
        revised["cases"][1]["interval"] = 64
        self.assertIs(matrix.completed_trials([row], revised, row["versions"])[("native", .02, 0)], row)
        for changed in ["resources", "completed_case", "removed_rate"]:
            invalid = copy.deepcopy(revised)
            if changed == "resources":
                invalid["engine_args"]["max_total_tokens"] = 65536
            elif changed == "completed_case":
                invalid["cases"][0]["policy"] = "fcfs"
            else:
                invalid["rates"].remove(.02)
            with self.assertRaises(ValueError):
                matrix.completed_trials([row], invalid, row["versions"])

    def test_completed_arm_is_keyed_separately_from_pending_pair(self):
        row = dict(condition="native", rate=.016, repeat=0, plan={"model": "8B"}, versions={"sglang": "pinned"})
        complete = matrix.completed_trials([row], row["plan"], row["versions"])
        self.assertIs(complete[("native", .016, 0)], row)
        self.assertNotIn(("plas", .016, 0), complete)
        for plan, versions in [({"model": "32B"}, row["versions"]), (row["plan"], {"sglang": "changed"})]:
            with self.assertRaises(ValueError):
                matrix.completed_trials([row], plan, versions)
        with self.assertRaises(ValueError):
            matrix.completed_trials([row, row], row["plan"], row["versions"])

    def test_failure_leaves_other_arms_runnable_and_resumes_only_the_failed_arm(self):
        with TemporaryDirectory() as directory:
            plan_path = Path(directory, "plan.json")
            output = Path(directory, "result.json")
            plan_path.write_text(json.dumps(dict(model="8B", workload="sharegpt.json", device="3",
                rates=[.02], engine_args={}, cases=[dict(name="first", policy="plas"),
                                                  dict(name="second", policy="fcfs")])))
            argv = ["--plan", str(plan_path), "--output", str(output), "--resume", "--keep-going"]
            row = dict(response_seconds_per_output_token={"program": .12}, output_tokens_per_second=12,
                       program_latency_seconds={"mean": 1})
            with (patch.object(matrix, "version", return_value="pinned"),
                  patch.object(matrix, "load_workloads", return_value=[{"program_id": "program"}]),
                  patch.object(matrix.subprocess, "check_output", return_value=""),
                  patch.object(matrix, "trial", side_effect=[RuntimeError("host KV exhausted"), row.copy()])):
                self.assertEqual(matrix.main(argv), 1)
            self.assertEqual([x["condition"] for x in json.loads(output.read_text())], ["second"])
            errors = json.loads(output.with_name("result-failures.json").read_text())
            self.assertEqual([(e["condition"], e["error"]) for e in errors],
                             [("first", "RuntimeError: host KV exhausted")])
            with (patch.object(matrix, "version", return_value="pinned"),
                  patch.object(matrix, "load_workloads", return_value=[{"program_id": "program"}]),
                  patch.object(matrix, "trial", return_value=row.copy()) as retry):
                self.assertEqual(matrix.main(argv), 0)
            retry.assert_called_once()
            self.assertEqual(retry.call_args.args[4], "plas")
            self.assertEqual({x["condition"] for x in json.loads(output.read_text())}, {"first", "second"})


if __name__ == "__main__":
    unittest.main()
