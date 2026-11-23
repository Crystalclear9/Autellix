import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("paper_motivation", Path(__file__).parents[1] / "scripts/analyze_paper_motivation.py")
motivation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motivation)


def test_call_concurrency_samples_outstanding_calls():
    calls = [dict(submitted_seconds=0, completed_seconds=5),
             dict(submitted_seconds=4, completed_seconds=11)]
    assert [p["active_calls"] for p in motivation.call_concurrency(calls, 12, interval=2)] == [1, 1, 2, 1, 1, 1, 0]
    assert motivation.call_concurrency(calls, 12, interval=2, start=4) == [
        dict(seconds=4, active_calls=2), dict(seconds=6, active_calls=1),
        dict(seconds=8, active_calls=1), dict(seconds=10, active_calls=1), dict(seconds=12, active_calls=0)]


def test_steady_state_counts_completions_and_clips_outstanding_intervals():
    row = dict(calls=[dict(submitted_seconds=0, completed_seconds=5),
                      dict(submitted_seconds=2, completed_seconds=7),
                      dict(submitted_seconds=6, completed_seconds=11),
                      dict(submitted_seconds=8, completed_seconds=12)],
               program_completion_times={"a": 5, "b": 11})

    assert motivation.steady_state(row, 5, 11) == dict(
        start_seconds=5, end_seconds=11, call_completions=2, program_completions=1,
        completed_calls_per_second=2 / 6, completed_programs_per_second=1 / 6,
        mean_outstanding_calls=10 / 6)


def test_prefix_counts_distinguish_same_program_and_prior_programs():
    programs = [dict(calls=[dict(input_ids=[1, 2, 3, 4]), dict(input_ids=[1, 2, 5, 6])]),
                dict(calls=[dict(input_ids=[1, 2, 5, 6])])]
    rows = {row["scope"]: row for row in motivation.prefix_opportunities(programs, block_size=2)}
    assert rows["within"]["mean_reusable_input_fraction"] == .5
    assert rows["across"]["mean_reusable_input_fraction"] == 1.
