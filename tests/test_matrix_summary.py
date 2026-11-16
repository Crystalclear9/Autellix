import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "matrix_summary", Path(__file__).parents[1] / "scripts/summarize_sglang_matrix.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MatrixSummaryTests(unittest.TestCase):
    def records(self):
        native = dict(condition="native", arm="sglang-native", rate=.02, repeat=0,
                      trace_sha256="trace", model="model", versions={"torch": "fixed"},
                      effective_engine_args={"max_total_tokens":8192, "autellix_native_baseline":True},
                      output_hashes={"p/c":"answer"}, programs={"p":10}, completion_tokens=20,
                      calls=[dict(workload_program="p", workload_call="c", prompt_tokens=30, completion_tokens=20)],
                      program_latency_seconds={"mean":10, "p95":10},
                      paper_sequential_token_latency={"mean":.5, "p95":.5},
                      output_tokens_per_second=2, elapsed_seconds=10, arrival_horizon_seconds=0,
                      drain_seconds=10, completions_during_arrivals=0, first_submission_lag_seconds={"p":.01})
        candidate = copy.deepcopy(native)
        candidate.update(condition="plas", arm="plas",
                         program_latency_seconds={"mean":12, "p95":12})
        candidate["effective_engine_args"]["autellix_native_baseline"] = False
        return [native, candidate]

    def test_text_variation_separate_from_identical_inputs_and_lengths(self):
        records = self.records()
        records[1]["output_hashes"]["p/c"] = "different"
        result = module.summarize(records)["rows"][1]
        self.assertEqual(result["matching_texts"], 0)
        self.assertTrue(result["same_trace"] and result["same_call_lengths"])
        self.assertTrue(result["same_engine_args"])
        self.assertAlmostEqual(result["mean_latency_change_pct"], 20)

    def test_different_lengths_and_resources_remain_visible(self):
        records = self.records()
        records[1]["calls"][0]["completion_tokens"] = 19
        records[1]["effective_engine_args"]["max_total_tokens"] = 16384
        result = module.summarize(records)["rows"][1]
        self.assertFalse(result["same_call_lengths"])
        self.assertFalse(result["same_engine_args"])

    def test_same_policy_can_have_multiple_named_conditions(self):
        records = self.records()
        records.append(copy.deepcopy(records[1]))
        records[-1]["condition"] = "plas-other-quantum"
        self.assertEqual(len(module.summarize(records)["rows"]), 3)
