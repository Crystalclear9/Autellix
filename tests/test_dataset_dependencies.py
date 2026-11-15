import unittest
from pathlib import Path
from autellix.experiments.datasets import programs_from_records
from autellix.experiments.workloads import make_paper_workload


class DatasetDependencyTests(unittest.TestCase):
    def test_explicit_empty_parents_preserve_parallel_roots(self):
        program = programs_from_records([{"program_id": "p", "calls": [
            {"call_id": "a", "parents": [], "model_time": 1},
            {"call_id": "b", "parents": [], "model_time": 1},
            {"call_id": "c", "parents": ["a", "b"], "model_time": 1},
        ]}])[0]
        self.assertEqual([c.parents for c in program.calls], [(), (), ("a", "b")])

    def test_dataset_sweep_changes_arrivals_and_preserves_sampled_programs(self):
        path = str(Path(__file__).parent / "fixtures/tiny_workload.jsonl")
        slow = make_paper_workload("sharegpt", seed=17, num_programs=12, arrival_rate=.1, dataset_path=path)
        fast = make_paper_workload("sharegpt", seed=17, num_programs=12, arrival_rate=1., dataset_path=path)
        self.assertEqual(len(slow), 12)
        self.assertEqual([p.program_id for p in slow], [p.program_id for p in fast])
        self.assertGreater(slow[-1].arrival_time, fast[-1].arrival_time)
        self.assertEqual(len({p.program_id for p in slow}), 12)
        for program in slow:
            self.assertTrue(all(c.program_id == program.program_id for c in program.calls))
