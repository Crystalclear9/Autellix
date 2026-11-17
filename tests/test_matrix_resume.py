import importlib.util
import copy
import sys
from pathlib import Path
import unittest

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


if __name__ == "__main__":
    unittest.main()
