import subprocess
import sys
import unittest

from autellix import Simulator, make_figure2_workload
from autellix.baselines import make_baseline
from autellix.experiments import ExperimentRunner, plot_records, write_records
from autellix.load_balancer import LocalityAwareLoadBalancer
from autellix.models import CallSpec, EngineState, ProcessEntry, ProgramSpec
from autellix.service import AutellixService
from autellix.schedulers import make_scheduler


class Figure2Tests(unittest.TestCase):
    def test_plas_reduces_total_wait_on_figure2(self):
        programs = make_figure2_workload()
        fcfs = Simulator(programs, scheduler="fcfs", batch_size=2).run()
        mlfq = Simulator(programs, scheduler="mlfq", batch_size=2).run()
        plas = Simulator(programs, scheduler="plas", batch_size=2).run()

        self.assertLess(plas.total_wait_time, fcfs.total_wait_time)
        self.assertLess(plas.total_wait_time, mlfq.total_wait_time)

    def test_plas_assigns_later_long_program_calls_to_lower_queue(self):
        result = Simulator(make_figure2_workload(), scheduler="plas", batch_size=2).run()
        a2 = result.calls[("A", "A2")]
        c2 = result.calls[("C", "C2")]

        self.assertGreaterEqual(a2.max_queue_index, 1)
        self.assertEqual(c2.service_priority, 1)


class SchedulerTests(unittest.TestCase):
    def test_srpt_prioritizes_short_remaining_work(self):
        programs = [
            ProgramSpec("L", (CallSpec("l", "L", model_time=100),)),
            ProgramSpec("S", (CallSpec("s", "S", model_time=1),)),
        ]
        scheduler = make_scheduler("srpt")
        result = Simulator(programs, scheduler=scheduler, batch_size=1).run()
        self.assertLess(result.calls[("S", "s")].finish_time, result.calls[("L", "l")].finish_time)

    def test_round_robin_rotates_calls(self):
        programs = [
            ProgramSpec("A", (CallSpec("a", "A", model_time=3),)),
            ProgramSpec("B", (CallSpec("b", "B", model_time=3),)),
        ]
        result = Simulator(programs, scheduler="round-robin", batch_size=1).run()
        order = [(row["program_id"], row["time"]) for row in result.gantt[:4]]
        self.assertEqual([program for program, _ in order], ["A", "B", "A", "B"])

    def test_schedule_interval_delays_new_batch(self):
        programs = [
            ProgramSpec("A", (CallSpec("a", "A", model_time=1),)),
            ProgramSpec("B", (CallSpec("b", "B", model_time=1),)),
        ]
        result = Simulator(programs, scheduler="fcfs", batch_size=1, schedule_interval=3).run()
        self.assertEqual(result.calls[("B", "b")].start_time, 3)

    def test_overprovision_prefetch_fills_slots_between_schedule_ticks(self):
        programs = [
            ProgramSpec("A", (CallSpec("a", "A", model_time=1),)),
            ProgramSpec("B", (CallSpec("b", "B", model_time=1),)),
            ProgramSpec("C", (CallSpec("c", "C", model_time=1),)),
        ]
        result = Simulator(
            programs,
            scheduler="fcfs",
            batch_size=1,
            schedule_interval=10,
            overprovision=2,
        ).run()

        self.assertEqual(result.calls[("B", "b")].start_time, 1)
        self.assertEqual(result.calls[("C", "c")].start_time, 2)
        self.assertEqual(result.prefetched_calls, 2)
        per_tick = {}
        for row in result.gantt:
            per_tick.setdefault(row["time"], 0)
            per_tick[row["time"]] += 1
        self.assertTrue(all(count <= 1 for count in per_tick.values()))

    def test_queue_binning_and_demotions(self):
        program = ProgramSpec(
            "P",
            (
                CallSpec("c1", "P", model_time=5, decode_tokens=5),
                CallSpec("c2", "P", model_time=1, decode_tokens=1, parents=("c1",)),
            ),
        )
        scheduler = make_scheduler(
            "plas",
            priority_boundaries=(0, 2, 4, float("inf")),
            queue_quanta=(1, 1, 1),
        )
        result = Simulator([program], scheduler=scheduler, batch_size=1).run()

        c1 = result.calls[("P", "c1")]
        c2 = result.calls[("P", "c2")]
        self.assertGreaterEqual(c1.max_queue_index, 2)
        self.assertEqual(c2.queue_index, 2)
        self.assertEqual(c2.service_priority, 5)

    def test_anti_starvation_promotes_waiting_call(self):
        long = ProgramSpec(
            "L",
            (CallSpec("l1", "L", model_time=10, decode_tokens=10),),
        )
        short = ProgramSpec(
            "S",
            (CallSpec("s1", "S", model_time=1, decode_tokens=1, release_delay=1),),
        )
        scheduler = make_scheduler(
            "plas",
            priority_boundaries=(0, 1, float("inf")),
            queue_quanta=(1, 100),
            anti_starvation_beta=1.0,
        )
        result = Simulator([long, short], scheduler=scheduler, batch_size=1).run()
        self.assertEqual(result.calls[("S", "s1")].queue_index, 0)

    def test_program_level_anti_starvation_promotes_across_engines(self):
        program = ProgramSpec(
            "P",
            (
                CallSpec("root", "P", model_time=1),
                CallSpec("left", "P", model_time=2, parents=("root",)),
                CallSpec("right", "P", model_time=2, parents=("root",)),
            ),
        )
        blocker_a = ProgramSpec(
            "A",
            (CallSpec("a", "A", model_time=10, release_delay=1),),
        )
        blocker_b = ProgramSpec(
            "B",
            (CallSpec("b", "B", model_time=10, release_delay=1),),
        )
        scheduler = make_scheduler(
            "atlas",
            priority_boundaries=(0, 1, float("inf")),
            queue_quanta=(1, 100),
            anti_starvation_beta=1.0,
        )
        result = Simulator(
            [program, blocker_a, blocker_b],
            scheduler=scheduler,
            load_balancer="round-robin",
            num_engines=2,
            batch_size=1,
        ).run()

        self.assertEqual(result.calls[("P", "left")].queue_index, 0)
        self.assertEqual(result.calls[("P", "right")].queue_index, 0)

    def test_fcfs_runs_without_preempting(self):
        program = ProgramSpec(
            "P",
            (CallSpec("c1", "P", model_time=5, decode_tokens=5),),
        )
        result = Simulator([program], scheduler="fcfs", batch_size=1).run()
        self.assertEqual(result.calls[("P", "c1")].max_queue_index, 0)
        self.assertEqual(result.program_metrics["P"].response_time, 5)


class AtlasTests(unittest.TestCase):
    def test_atlas_releases_parallel_children_and_tracks_critical_path(self):
        program = ProgramSpec(
            "P",
            (
                CallSpec("root", "P", model_time=2, decode_tokens=2),
                CallSpec("left", "P", model_time=4, decode_tokens=4, parents=("root",)),
                CallSpec("right", "P", model_time=1, decode_tokens=1, parents=("root",)),
                CallSpec("join", "P", model_time=3, decode_tokens=3, parents=("left", "right")),
            ),
        )
        result = Simulator([program], scheduler="atlas", batch_size=2).run()

        left = result.calls[("P", "left")]
        right = result.calls[("P", "right")]
        join = result.calls[("P", "join")]
        self.assertEqual(left.ready_time, right.ready_time)
        self.assertEqual(join.ready_time, max(left.finish_time, right.finish_time))
        self.assertEqual(result.process_table["P"].service_time, 9)

    def test_atlas_uses_per_call_critical_path_priority(self):
        program = ProgramSpec(
            "P",
            (
                CallSpec("root", "P", model_time=2),
                CallSpec("left", "P", model_time=4, parents=("root",)),
                CallSpec("right", "P", model_time=1, parents=("root",)),
                CallSpec("join", "P", model_time=3, parents=("left", "right")),
            ),
        )
        result = Simulator([program], scheduler="atlas", batch_size=2).run()

        self.assertEqual(result.calls[("P", "root")].critical_path_service, 0)
        self.assertEqual(result.calls[("P", "left")].service_priority, 2)
        self.assertEqual(result.calls[("P", "right")].service_priority, 2)
        self.assertEqual(result.calls[("P", "join")].critical_path_service, 6)
        self.assertEqual(result.calls[("P", "join")].service_priority, 6)

    def test_dag_token_latency_uses_critical_path_response(self):
        program = ProgramSpec(
            "P",
            (
                CallSpec("root", "P", model_time=1, decode_tokens=1),
                CallSpec(
                    "left",
                    "P",
                    model_time=1,
                    decode_tokens=1,
                    parents=("root",),
                    release_delay=5,
                ),
                CallSpec(
                    "right",
                    "P",
                    model_time=1,
                    decode_tokens=1,
                    parents=("root",),
                ),
            ),
        )
        result = Simulator([program], scheduler="atlas", batch_size=1).run()
        metric = result.program_metrics["P"]

        self.assertEqual(metric.response_time, 7)
        self.assertEqual(metric.critical_path_response_time, 2)
        self.assertEqual(metric.token_latency, metric.critical_path_token_latency)
        self.assertNotEqual(metric.token_latency, metric.response_time / metric.generated_tokens)


class BaselineTests(unittest.TestCase):
    def test_autellix_baseline_selects_plas_for_sequential_and_atlas_for_dag(self):
        sequential = make_figure2_workload()
        dag = [
            ProgramSpec(
                "P",
                (
                    CallSpec("root", "P", model_time=1),
                    CallSpec("left", "P", model_time=1, parents=("root",)),
                    CallSpec("right", "P", model_time=1, parents=("root",)),
                ),
            )
        ]

        self.assertEqual(make_baseline("autellix", programs=sequential).scheduler.name, "plas")
        self.assertEqual(make_baseline("autellix", programs=dag).scheduler.name, "atlas")


class LoadBalancerTests(unittest.TestCase):
    def test_short_requests_use_least_used_and_long_requests_pin(self):
        balancer = LocalityAwareLoadBalancer(token_threshold=2048)
        engines = [
            EngineState(0, batch_size=1, queue_count=1),
            EngineState(1, batch_size=1, queue_count=1),
        ]
        process_table = {"P": ProcessEntry("P", arrival_time=0)}
        engines[0].queues[0].append(
            # Deliberately only to increase workload; this dummy state is never run.
            Simulator(
                [ProgramSpec("D", (CallSpec("d", "D", model_time=1),))],
                scheduler="fcfs",
                batch_size=1,
            ).calls[("D", "d")]
        )

        short = CallSpec("s", "P", model_time=1, prefill_tokens=100, decode_tokens=100)
        self.assertEqual(balancer.assign(short, engines, process_table).engine_id, 1)

        long1 = CallSpec("l1", "P", model_time=1, prefill_tokens=3000, decode_tokens=10)
        long2 = CallSpec("l2", "P", model_time=1, prefill_tokens=4000, decode_tokens=10)
        first_engine = balancer.assign(long1, engines, process_table).engine_id
        self.assertEqual(balancer.assign(long2, engines, process_table).engine_id, first_engine)


class RobustnessTests(unittest.TestCase):
    def test_invalid_programs_fail_early(self):
        with self.assertRaises(ValueError):
            ProgramSpec("E", ())
        with self.assertRaises(ValueError):
            ProgramSpec(
                "C",
                (
                    CallSpec("a", "C", model_time=1, parents=("b",)),
                    CallSpec("b", "C", model_time=1, parents=("a",)),
                ),
            )


class ServiceTests(unittest.TestCase):
    def test_dynamic_session_matches_static_trace(self):
        service = AutellixService(scheduler="plas", batch_size=1)
        session = service.start_session("P")
        service.submit_call(session.session_id, "c1", model_time=2)
        service.submit_call(session.session_id, "c2", model_time=1, parents=("c1",))
        program = service.complete_session(session.session_id)
        dynamic = service.run()
        static = Simulator([program], scheduler="plas", batch_size=1).run()
        self.assertEqual(dynamic.makespan, static.makespan)

    def test_online_submission_updates_live_process_table_and_drains_session(self):
        service = AutellixService(scheduler="plas", batch_size=1)
        session = service.start_session("P")
        root = service.submit_call(session.session_id, "root", model_time=2)

        partial = service.tick()
        self.assertIsNotNone(partial)
        self.assertIn("root", partial.process_table["P"].active_call_ids)

        child = service.submit_call(
            session.session_id,
            "child",
            model_time=1,
            parents=("root",),
        )
        dynamic = service.drain()
        static = Simulator(
            [
                ProgramSpec(
                    "P",
                    (
                        root,
                        child,
                    ),
                )
            ],
            scheduler="plas",
            batch_size=1,
        ).run()

        self.assertEqual(dynamic.makespan, static.makespan)
        self.assertEqual(service.sessions, {})


class ExecutionModelTests(unittest.TestCase):
    def test_cache_model_reduces_later_same_program_prefill(self):
        program = ProgramSpec(
            "P",
            (
                CallSpec("c1", "P", model_time=1, prefill_tokens=4000, decode_tokens=64),
                CallSpec("c2", "P", model_time=1, prefill_tokens=4000, decode_tokens=64, parents=("c1",)),
            ),
        )
        result = Simulator([program], scheduler="plas", batch_size=1, execution_model="autellix").run()
        self.assertGreater(result.calls[("P", "c1")].prefill_time, result.calls[("P", "c2")].prefill_time)
        self.assertGreater(result.calls[("P", "c2")].cache_hit_rate, result.calls[("P", "c1")].cache_hit_rate)


class ExperimentTests(unittest.TestCase):
    def test_experiment_sweep_and_plot_smoke(self):
        runner = ExperimentRunner(seed=0, batch_size=4)
        records = runner.sweep(
            workload="sharegpt",
            baseline_names=["vllm", "autellix"],
            arrival_rates=[0.2],
            engines=1,
            num_programs=2,
        )
        self.assertTrue(records)
        self.assertIn("p95_token_latency", records[0])
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            json_path, csv_path = write_records(records[:3], tmp)
            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())
            figures = plot_records(json_path, tmp)
            self.assertTrue(figures[0].exists())

    def test_multi_engine_ablation_changes_only_load_balancer(self):
        runner = ExperimentRunner(seed=0, batch_size=4)
        records = runner.sweep(
            workload="sharegpt",
            baseline_names=["autellix"],
            arrival_rates=[0.4],
            engines=2,
            num_programs=2,
            load_balancers=["round-robin", "least-used", "autellix"],
        )

        self.assertEqual({row["scheduler_policy"] for row in records}, {"plas"})
        self.assertEqual(
            {row["load_balancer_policy"] for row in records},
            {"round-robin", "least-used", "autellix"},
        )


class CLITests(unittest.TestCase):
    def test_run_and_compare_commands(self):
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "autellix.cli",
                "run",
                "--workload",
                "figure2",
                "--policy",
                "plas",
                "--batch-size",
                "2",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("policy=plas", run.stdout)
        compare = subprocess.run(
            [
                sys.executable,
                "-m",
                "autellix.cli",
                "compare",
                "--workload",
                "figure2",
                "--policies",
                "fcfs,mlfq,plas",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("policy", compare.stdout)
        self.assertIn("plas", compare.stdout)


if __name__ == "__main__":
    unittest.main()
