# Autellix Scheduling Simulator

Autellix is a CPU-only, deterministic simulator for the paper
`2502.13965v1.pdf`, "Autellix: An Efficient Serving Engine for LLM Agents as
General Programs".

The repository models LLM calls as timed jobs and reproduces the paper's
program-aware scheduling, locality-aware routing, synthetic workloads, and
experiment structure. It does not run real models, patch vLLM, launch a CUDA
kernel, or require GPU hardware.

## Quick Start

```powershell
python -m unittest discover -s tests
python -m autellix.cli compare --workload figure2 --policies fcfs,mlfq,plas
python -m autellix.cli paper-preset --preset workload-analysis --dataset tests\fixtures\tiny_workload.jsonl --programs 2
```

## Project Layout

```text
autellix/
  core/          Scheduling simulator, models, execution costs, load balancing
  frontend/      Stateful service, OpenAI-style client, async engine facade
  experiments/   Baselines, workloads, dataset importers, paper-style presets
  *.py           Backward-compatible wrappers for old import paths
cuda/            Standalone batched swap benchmark scaffold
integrations/    Optional vLLM integration scaffold
tests/           Unit tests and tiny dataset fixtures
outputs/         Example generated experiment output
```

Preferred imports use the new subpackages:

```python
from autellix.core import Simulator
from autellix.frontend import AutellixClient
from autellix.experiments import ExperimentRunner
```

Older imports such as `from autellix.simulator import Simulator` remain
supported through compatibility wrappers.

## Implemented Surface

- Process table with service time, waiting time, engine assignment, active
  calls, thread metadata, and arrival/completion timestamps.
- Schedulers: `fcfs`, `round-robin`, `mlfq`, `plas`, `atlas`, and simulator-only
  `srpt`.
- Baselines: `vllm`, `vllm-opt`, `mlfq`, and `autellix`.
- Autellix load balancer: short requests use least-used routing; long requests
  are pinned to a program engine for locality.
- Multi-step scheduling with overprovisioned prefetch slots.
- Stateful frontend and OpenAI-style simulated chat API.
- JSON/JSONL/CSV workload importers and paper-style experiment presets.

## CLI

Run the Figure 2 workload:

```powershell
python -m autellix.cli run --workload figure2 --policy plas --batch-size 2
python -m autellix.cli compare --workload figure2 --policies fcfs,mlfq,plas
```

Run synthetic workload sweeps:

```powershell
python -m autellix.cli run --workload mixed --policy atlas --engines 4 --seed 0
python -m autellix.cli sweep --workload mixed --policies vllm,vllm-opt,mlfq,autellix --arrival-rates 0.1,0.2,0.4
python -m autellix.cli paper-suite --quick --output outputs/quick
python -m autellix.cli plot --input outputs/quick/results.json --output outputs/quick/figures
```

Run paper-style presets:

```powershell
python -m autellix.cli paper-preset --preset workload-analysis --dataset tests\fixtures\tiny_workload.jsonl
python -m autellix.cli paper-preset --preset timing-breakdown --workload sharegpt --programs 4
python -m autellix.cli paper-preset --preset latency-throughput --output outputs/latency_throughput
```

Available presets are `workload-analysis`, `latency-throughput`,
`load-balancer`, `offline-makespan`, and `timing-breakdown`.

## Python API

Core simulator:

```python
from autellix.core import Simulator
from autellix.experiments import make_figure2_workload

programs = make_figure2_workload()
result = Simulator(programs, scheduler="plas", batch_size=2).run()
print(result.summary())
```

Stateful frontend:

```python
from autellix.frontend import AutellixClient

client = AutellixClient(scheduler="atlas", batch_size=2)
with client.session("program-1", drain_on_exit=True) as session:
    client.chat.completions.create(
        model="simulated-model",
        session_id=session.session_id,
        messages=[{"role": "user", "content": "Start"}],
        call_id="root",
        thread_id="main",
        framework_metadata={"framework": "langgraph"},
    )

print(client.service.last_result.process_table["program-1"].thread_metadata)
```

Async engine facade:

```python
from autellix.frontend import AsyncMultiLLMEngine

engine = AsyncMultiLLMEngine(
    scheduler="plas",
    load_balancer="autellix",
    num_engines=2,
    batch_size=1,
)

future = engine.submit_call(
    "program-1",
    "call-1",
    model_time=2,
    prefill_tokens=4096,
    decode_tokens=128,
)

engine.drain()
print(future.done())
print(future.result().metrics)
```

`AsyncMultiLLMEngine(process_mode=True)` starts a lightweight worker process and
mirrors submit/step/drain commands through multiprocessing primitives. This is
still a simulator scaffold, not real vLLM engine parallelism.

## Datasets

Use `load_programs_from_file()` for tiny JSON, JSONL, or CSV traces. Common
fields are `program_id`, `call_id`, `parent_id`, `parents`, `prefill_tokens`,
`decode_tokens`, `model_time`, `arrival_time`, and `thread_id`.

```python
from autellix.experiments import load_programs_from_file, workload_analysis

programs = load_programs_from_file("tests/fixtures/tiny_workload.jsonl")
print(workload_analysis(programs))
```

## Tunable Defaults

The paper does not publish exact numeric values for queue boundaries, time
quanta, or beta. This simulator uses:

- priority boundaries: `0,2,4,8,16,32,64,inf`
- queue quanta: `1,2,4,8,16,32,64`
- anti-starvation beta: `8.0`
- locality token threshold: `2048`
- schedule interval: `1`
- Autellix baseline overprovision: `1`

Override them from the CLI:

```powershell
python -m autellix.cli run --policy plas --boundaries 0,4,16,inf --quanta 1,4,16 --beta 6
```

## Metrics

`SimulationResult.summary()` and JSON output include:

- `scheduler_policy` / `policy`
- `load_balancer_policy` / `load_balancer`
- `prefetched_calls`
- `critical_path_response_time`
- `critical_path_token_latency`
- aggregate wait, execution, prefill, decode, swap, and scheduler time

For fork/join DAG programs, token latency follows the paper footnote:
critical-path response time divided by total generated tokens across all
threads.

## Tests

```powershell
python -m unittest discover -s tests
```

The test suite covers Figure 2 behavior, PLAS/ATLAS scheduling, queue demotion,
anti-starvation, cache-aware execution, dynamic sessions, async engine futures,
dataset importers, paper presets, CLI smoke checks, and optional vLLM scaffold
imports.

## Research Scaffolds

- `integrations/vllm/` contains import-safe vLLM v0.6.1 adapter notes and
  request metadata helpers. It does not import vLLM unless explicitly asked.
- `cuda/` contains a standalone batched swap benchmark scaffold. The current
  benchmark is CPU-only and documents where a real CUDA memcpy benchmark should
  replace it.

## Boundaries

This repository is not a full paper artifact. It does not:

- patch or run vLLM,
- execute real LLM inference,
- implement the CUDA/C++ KV swap kernel,
- reproduce A100/Falcon/LLaMA throughput numbers,
- provide production-grade multi-process serving.

It is intended as a compact, testable implementation of the scheduling,
routing, API, and experiment ideas at simulation level.
