import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("comparison_summary", Path(__file__).parents[1]/"scripts/summarize_sglang_comparison.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ComparisonSummaryTests(unittest.TestCase):
    def rows(self):
        common = dict(rate=1, repeat=0, trace_sha256="trace", completion_tokens=10,
                      output_hashes={"p/c":"output"})
        return [dict(common,arm=arm,program_latency_seconds={"mean":lat,"p95":lat},
                     paper_sequential_token_latency={"mean":lat/10},output_tokens_per_second=10/lat)
                for arm,lat in [("sglang-native",10),("atlas",12)]]

    def test_negative_benefit_is_preserved(self):
        result=module.summarize(self.rows(),1)["pairs"][0]
        self.assertAlmostEqual(result["mean_latency_reduction_pct"],-20)
        self.assertTrue(result["matched"])

    def test_missing_arm_or_repeat_is_rejected(self):
        with self.assertRaises(ValueError): module.summarize([],1)
        with self.assertRaises(ValueError): module.summarize(self.rows()[:1],1)
        with self.assertRaises(ValueError): module.summarize(self.rows(),3)
        rows=self.rows()
        for row in rows: row["repeat"]=2
        with self.assertRaises(ValueError): module.summarize(rows,1)

    def test_different_generated_text_invalidates_pair(self):
        rows=self.rows(); rows[1]["output_hashes"]={"p/c":"different"}
        self.assertFalse(module.summarize(rows,1)["aggregate"][0]["all_outputs_match"])

    def test_different_resource_budget_invalidates_pair(self):
        rows=self.rows()
        rows[0]["effective_engine_args"]={"max_total_tokens":4096,"autellix_native_baseline":True}
        rows[1]["effective_engine_args"]={"max_total_tokens":8192,"autellix_native_baseline":False}
        self.assertFalse(module.summarize(rows,1)["pairs"][0]["matched"])
