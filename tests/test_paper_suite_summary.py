import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("paper_summary", Path(__file__).parents[1] / "scripts/summarize_paper_suite.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class PaperSuiteSummary(unittest.TestCase):
    def records(self):
        ids = [str(i) for i in range(4)]
        native = dict(plan={"workload_kind": "lats"}, condition="native", arm="sglang-native", rate=.1, seed=42,
            device_groups=["1"], arrival_times={p: int(p)*10 for p in ids},
            program_completion_times={p: int(p)*10+10 for p in ids}, arrival_horizon_seconds=30,
            programs={p: 10 for p in ids}, program_token_latencies={p: .8 for p in ids},
            response_seconds_per_output_token={p: 1 for p in ids}, completion_tokens=40,
            calls=[dict(workload_program=p, workload_call="0", engine_id=0, prompt_tokens=2, completion_tokens=10) for p in ids],
            elapsed_seconds=40, drain_seconds=10, completions_during_arrivals=3,
            first_submission_lag_seconds={p: 0 for p in ids}, output_tokens_per_second=1,
            trace_sha256="matching", effective_engine_args={"autellix_native_baseline": True},
            model="Llama", versions={"sglang": "0.4.9.post6"}, output_hashes={p: "output" for p in ids})
        candidate = copy.deepcopy(native)
        candidate.update(condition="atlas", arm="atlas", programs={p: 12 for p in ids},
                         program_token_latencies={p: .6 for p in ids}, output_tokens_per_second=.8)
        candidate["effective_engine_args"]["autellix_native_baseline"] = False
        return [native, candidate]

    def test_dag_metric_and_negative_throughput_remain_separate(self):
        result = summary.summarize(self.records())
        pair = result["comparisons"][0]
        self.assertAlmostEqual(pair["mean_response_change_pct"], 20)
        self.assertAlmostEqual(pair["mean_paper_latency_change_pct"], -25)
        self.assertAlmostEqual(pair["throughput_change_pct"], -20)
        self.assertEqual(pair["faster_programs"], 0)
        self.assertEqual(result["trials"][0]["backlog_checkpoints"][-1]["unfinished_programs"], 1)

    def test_different_work_is_not_a_pair(self):
        rows = self.records()
        rows[1]["calls"][0]["completion_tokens"] = 11
        with self.assertRaises(ValueError):
            summary.summarize(rows)

    def test_same_label_and_devices_can_be_reused_for_a_larger_model(self):
        small, large = self.records(), self.records()
        for row in large:
            row["model"] = "Llama-70B"
        result = summary.summarize(small + large)
        self.assertEqual({row["model"] for row in result["comparisons"]}, {"Llama", "Llama-70B"})

    def test_candidate_host_swap_budget_does_not_change_native_gpu_pairing(self):
        rows = self.records()
        rows[0]["effective_engine_args"]["autellix_swap_space"] = 8
        rows[1]["effective_engine_args"]["autellix_swap_space"] = 24
        result = summary.summarize(rows)
        self.assertTrue(result["comparisons"][0]["pairing"]["same_options"])
        self.assertEqual(result["trials"][1]["engine_args"]["autellix_swap_space"], 24)
        rows[1]["effective_engine_args"]["max_total_tokens"] = 32768
        with self.assertRaises(ValueError):
            summary.summarize(rows)

    def test_labeled_dag_replay_uses_critical_path_and_records_backend_cache_count(self):
        rows = self.records()
        for row in rows:
            row["plan"].update(workload_kind="lats-h20", workload_family="lats")
            for call in row["calls"]:
                call["cached_prompt_tokens"] = 1
        result = summary.summarize(rows)
        self.assertAlmostEqual(result["comparisons"][0]["mean_paper_latency_change_pct"], -25)
        self.assertEqual(result["trials"][0]["sglang_cached_tokens_per_prompt_token"], .5)
        for call in rows[1]["calls"]:
            call["sglang_cached_tokens"] = 3
        result = summary.summarize(rows)
        self.assertEqual(result["trials"][1]["sglang_cached_tokens"], 12)
        self.assertEqual(result["trials"][1]["sglang_cached_tokens_per_prompt_token"], 1.5)

    def test_upstream_default_is_listed_against_the_matched_native_baseline(self):
        native, candidate = self.records()
        for row in (native, candidate):
            row["effective_engine_args"].update(attention_backend="triton", sampling_backend="pytorch")
        upstream = copy.deepcopy(native)
        upstream["condition"] = "native-default"
        upstream["effective_engine_args"].update(autellix_native_defaults=True, attention_backend=None,
                                                 sampling_backend=None)
        result = summary.summarize([upstream, native, candidate])
        first = result["comparisons"][0]
        self.assertEqual(first["baseline_condition"], "native")
        self.assertTrue(first["upstream_default_optimizations"])
        self.assertAlmostEqual(result["trials"][0]["completed_programs_per_second"], .1)
        self.assertIn("p99_paper_latency_change_pct", result["comparisons"][1])


if __name__ == "__main__":
    unittest.main()
