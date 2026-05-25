# Autellix Scheduling Simulator

This repository contains a CPU-only reproduction of the simulation-level
content from `2502.13965v1.pdf`, "Autellix: An Efficient Serving Engine for
LLM Agents as General Programs".

It does not create a virtual environment, download models, run vLLM, or
reproduce the CUDA swap kernel. Instead, it models LLM calls as timed jobs with
prefill/decode/cache/swap approximations and reproduces the paper's scheduling,
routing, and experiment structure in a deterministic simulator.

## What Is Implemented

- Program process table with cumulative service time, waiting time, engine
  assignment, active calls, thread metadata, and arrival/completion timestamps.
- Stateful frontend-style API with `AutellixService.start_session()`,
  `submit_call()`, `complete_session()`, `tick()`, `run_until_idle()`, and
  `drain()`. Submitted calls receive concrete online submit times and are
  admitted into a live simulator/process table.
- Schedulers:
  - `fcfs`: first-come first-served baseline.
  - `round-robin`: preemptive round-robin baseline.
  - `mlfq`: preemptive multilevel feedback queue baseline.
  - `plas`: Program-Level Attained Service for single-threaded programs.
  - `atlas`: Adaptive Thread-Level Attained Service for DAG-style programs,
    using each call's observed critical-path attained service:
    `max(parent.critical_path_service + parent.model_time)`.
  - `srpt`: clairvoyant simulator-only baseline for comparison.
- Baselines:
  - `vllm`: FCFS plus no prefix caching.
  - `vllm-opt`: FCFS plus prefix caching and unbatched swap costs.
  - `mlfq`: MLFQ plus prefix caching and unbatched swap costs.
  - `autellix`: PLAS for single-chain programs, ATLAS for fork/join DAG
    programs, locality-aware balancing, multi-step scheduling, and batched swap
    costs.
- Autellix load balancer:
  - requests with `total_tokens <= 2048` go to the least-used engine.
  - longer requests are pinned to their program's engine for locality.
- Multi-step scheduling:
  - the scheduler runs every `N` decode steps.
  - it may select `batch_size + overprovision` candidates.
  - at most `batch_size` calls run at once; extra candidates sit in an engine
    prefetch queue and immediately fill slots that open before the next
    scheduler tick.
- Workloads:
  - `figure2`: toy workload from Figure 2.
  - `sharegpt`, `bfcl`, `lats`, `mixed`: seeded synthetic workloads inspired by
    the paper's Section 6 distributions.

## Run

```powershell
python -m autellix.cli run --workload figure2 --policy plas --batch-size 2
python -m autellix.cli compare --workload figure2 --policies fcfs,mlfq,plas
python -m autellix.cli run --workload mixed --policy atlas --engines 4 --seed 0
python -m autellix.cli sweep --workload mixed --policies vllm,vllm-opt,mlfq,autellix --arrival-rates 0.1,0.2,0.4
python -m autellix.cli paper-suite --quick --output outputs/quick
python -m autellix.cli plot --input outputs/quick/results.json --output outputs/quick/figures
```

For JSON output:

```powershell
python -m autellix.cli run --workload figure2 --policy plas --json
```

## Tunable Defaults

The paper does not publish exact numeric values for queue boundaries, time
quanta, or beta. This reproduction uses:

- priority boundaries: `0,2,4,8,16,32,64,inf`
- queue quanta: `1,2,4,8,16,32,64`
- anti-starvation beta: `8.0`
- locality token threshold: `2048`
- schedule interval: `1`
- Autellix baseline overprovision: `1`

These can be overridden from the CLI:

```powershell
python -m autellix.cli run --policy plas --boundaries 0,4,16,inf --quanta 1,4,16 --beta 6
```

## Test

```powershell
python -m unittest discover -s tests
```

The tests cover the Figure 2 workload, queue binning and demotion,
anti-starvation promotion, FCFS non-preemption, ATLAS DAG dependency handling
and per-call critical-path priority, SRPT, round-robin, multi-step prefetch,
dynamic online sessions, cache-aware execution, locality-aware load balancing,
load-balancer-only multi-engine ablation semantics, experiment output,
plotting, and CLI smoke checks.

## Metrics

`SimulationResult.summary()` and JSON output include both legacy names and
paper-oriented labels:

- `scheduler_policy` / `policy`
- `load_balancer_policy` / `load_balancer`
- `prefetched_calls`
- `critical_path_response_time`
- `critical_path_token_latency`

For single-chain programs, token latency remains `response_time / generated
tokens`. For fork/join DAG programs, token latency follows the paper footnote:
critical-path response time divided by total generated tokens across all
threads.

## Simulation Boundaries

The cache and swap model is a deterministic approximation for reproducing
paper-level trends and ablations. Prefix cache hit rates, recomputation costs,
swap overhead, and batched swap savings are configurable simulation costs, not
measurements from vLLM internals or a CUDA kernel implementation.

## Python API

```python
from autellix import Simulator, make_figure2_workload

programs = make_figure2_workload()
result = Simulator(programs, scheduler="plas", num_engines=1, batch_size=2).run()
print(result.summary())
```

For paper-style baselines and load-balancer ablations:

```python
from autellix.baselines import make_baseline
from autellix.experiments import ExperimentRunner
from autellix.workloads import make_paper_workload

programs = make_paper_workload("lats", seed=0, num_programs=4)
baseline = make_baseline("autellix", programs=programs)
print(baseline.scheduler.name)  # atlas for fork/join DAG workloads

runner = ExperimentRunner(seed=0)
records = runner.sweep(
    workload="sharegpt",
    baseline_names=["autellix"],
    arrival_rates=[0.4],
    engines=2,
    num_programs=4,
    load_balancers=["round-robin", "least-used", "autellix"],
)
```

## Stateful API Prototype

The repository also includes an in-process prototype of the paper's stateful
frontend and multi-engine meta-engine. It is still a deterministic simulator:
it does not call a real model, vLLM, CUDA, or multiprocessing IPC.

OpenAI-style chat facade:

```python
from autellix import AutellixClient

client = AutellixClient(scheduler="plas", num_engines=2, batch_size=1)
session = client.service.start_session("program-1")

response = client.chat.completions.create(
    model="simulated-model",
    session_id=session.session_id,
    messages=[{"role": "user", "content": "Plan a tool call"}],
    call_id="root",
    max_tokens=64,
)

result = client.service.drain()
print(response.usage)
print(result.calls[("program-1", "root")].engine_id)
```

Async multi-engine facade:

```python
from autellix import AsyncMultiLLMEngine

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

`AutellixService.end_session()` is a semantic alias for
`complete_session()`. `drain()` marks completed sessions done and removes them
from the live service registry, matching the frontend session lifecycle in the
paper at simulation level.
