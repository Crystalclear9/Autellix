import unittest
from autellix.experiments.workloads import make_synthetic_workload
from autellix.experiments.datasets import workload_analysis


class SyntheticStatisticsTests(unittest.TestCase):
    def test_synthetic_means_follow_paper_table(self):
        # Fixed large samples detect lognormal median/mean confusion and
        # truncation bias in call counts; they do not assume exact random draws.
        for kind, calls, inputs, outputs in [("sharegpt", 6.66, 256, 277),
                                             ("bfcl", 10.75, 735.06, 34.14),
                                             ("lats", 159.7, 467.2, 72.6)]:
            count = 1200 if kind != "lats" else 300
            stats = workload_analysis(make_synthetic_workload(kind, seed=19, num_programs=count))
            with self.subTest(kind=kind):
                self.assertAlmostEqual(stats["mean_calls_per_program"], calls, delta=calls*.06)
                self.assertAlmostEqual(stats["mean_prefill_tokens"], inputs, delta=inputs*.05)
                self.assertAlmostEqual(stats["mean_decode_tokens"], outputs, delta=outputs*.05)
