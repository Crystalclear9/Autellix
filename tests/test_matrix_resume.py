import importlib.util
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
