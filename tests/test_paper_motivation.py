import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("paper_motivation", Path(__file__).parents[1] / "scripts/analyze_paper_motivation.py")
motivation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motivation)


def test_call_concurrency_samples_outstanding_calls():
    calls = [dict(submitted_seconds=0, completed_seconds=5),
             dict(submitted_seconds=4, completed_seconds=11)]
    assert [p["active_calls"] for p in motivation.call_concurrency(calls, 12, interval=2)] == [1, 1, 2, 1, 1, 1, 0]


def test_prefix_counts_distinguish_same_program_and_prior_programs():
    programs = [dict(calls=[dict(input_ids=[1, 2, 3, 4]), dict(input_ids=[1, 2, 5, 6])]),
                dict(calls=[dict(input_ids=[1, 2, 5, 6])])]
    rows = {row["scope"]: row for row in motivation.prefix_opportunities(programs, block_size=2)}
    assert rows["within"]["mean_reusable_input_fraction"] == .5
    assert rows["across"]["mean_reusable_input_fraction"] == 1.
