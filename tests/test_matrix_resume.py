import importlib.util
import copy
import json
import sys
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
try:
    spec = importlib.util.spec_from_file_location("matrix", Path(__file__).parents[1] / "scripts/run_sglang_matrix.py")
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
finally:
    sys.path.pop(0)


class MatrixResumeTests(unittest.TestCase):
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
