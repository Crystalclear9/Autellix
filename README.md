# Autellix

Program-aware LLM inference based on "Autellix: An Efficient Serving Engine for
LLM Agents as General Programs" (`2502.13965v1.pdf`).

The repository contains **real GPU inference backends** and a separate CPU
simulator. Use `autellix.runtime` for real inference. The original
`AutellixClient`, `AsyncMultiLLMEngine`, and simulation CLI remain simulation
APIs for backward compatibility.

## Real inference: Linux / Ubuntu WSL

Use separate Python 3.10 environments: vLLM **0.6.1** and SGLang **0.4.9.post6**
have different Torch dependencies. The integrations validate backend versions
and modify the scheduler in their own processes, without editing site-packages.

```bash
# In Ubuntu, from this repository. uv must be installed.
sudo apt-get update && sudo apt-get install -y build-essential python3-dev
bash scripts/setup_backend.sh vllm
source ~/autellix-envs/vllm/bin/activate

autellix-serve --backend vllm --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --policy atlas --devices 0 --state-dir /tmp/autellix-server \
  --engine-args '{"max_model_len":512,"max_num_seqs":4,"gpu_memory_utilization":0.35,"swap_space":0.25}'
```

For SGLang, install with `bash scripts/setup_backend.sh sglang`, activate that
environment, and use:

```bash
autellix-serve --backend sglang --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --policy atlas --devices 0 \
  --engine-args '{"context_length":512,"max_running_requests":4,"mem_fraction_static":0.35}'
```

The server provides `/v1/chat/completions` (including SSE streaming), `/v1/models`,
`/sessions`, `/requests/{id}`, and `/health`. Explicit sessions carry program
history between calls; calls without a session use a temporary one-call program.
The checkpoint must supply a chat template for chat requests. Raw prompts or
token IDs are supported by the Python runtime.

```python
from autellix.runtime import InferenceClient

client = InferenceClient("http://127.0.0.1:8000")
with client.session() as sid:
    answer = client.chat(
        [{"role": "user", "content": "What is the capital of France?"}],
        session_id=sid, temperature=0, max_tokens=32,
    )
    print(answer["choices"][0]["message"]["content"])
```

For direct Python use, see `examples/real_inference.py`. `InferenceEngine.submit`
returns a concurrent future completed only by the selected worker's actual
result; cancellation and worker failures propagate through IPC. Create the
engine under `if __name__ == "__main__"` when using multiprocessing.

### Backend behavior

- Online PLAS/ATLAS, MLFQ, and FCFS; measured model service in seconds, queue
  demotion, FIFO ordering, and per-call anti-starvation.
- Transactional program statistics shared between replica processes. Sessions
  close after their admitted calls finish or are cancelled.
- Real tokenizer-based short-request balancing and long-request engine affinity.
- vLLM: native block allocation, GPU/CPU KV swap, batched CUDA packing and pinned
  host transfers, and native multi-step execution (`--schedule-interval N`).
- SGLang: scheduler-process hooks, native cache retraction/recomputation, and
  optional resident radix-prefix reserves (`--overprovision K`).
- vLLM `--overprovision K` retains up to K displaced requests' allocated GPU KV
  between scheduling decisions, subject to memory availability.
- Each replica uses one GPU; use `--devices 0,1` for multiple replicas. Sharing a
  GPU is possible with explicit per-replica memory budgets. TP/PP, speculative
  decoding, and chunked prefill are outside the pinned adapters' supported mode.

SGLang policy refresh every N steps and vLLM native multi-step execution are
different mechanisms. Resident reserves reuse KV; they do not promise arbitrary
mid-window insertion into vLLM's cached CUDA batch. These implementations do not
claim the paper's A100 throughput numbers or an exact performance reproduction.

### Validation

```bash
# CPU policy / IPC / protocol checks; GPU tests explicitly skip without opt-in.
python -m unittest discover -s tests -v

# Real generation, preemption, resumed-output equality, session inheritance.
AUTELLIX_GPU_BACKEND=vllm AUTELLIX_TEST_MODEL=/path/to/model \
  AUTELLIX_TEST_CUDA_SWAP=1 \
  python -m unittest discover -s tests -p test_gpu_runtime.py -v

# Repeat in the SGLang environment with AUTELLIX_GPU_BACKEND=sglang.
# Native vLLM multi-step validation: add AUTELLIX_TEST_STEPS=3.
python cuda/batched_swap_benchmark.py --blocks 128 --layers 4
```

`scripts/http_smoke.py` starts an actual local server and checks real generation,
streaming, and session cleanup. Traces in `state_dir/replica-*.jsonl` contain
admissions, batches, measured execution, promotions/demotions, and preemptions.
Store the shared SQLite state on a local Linux filesystem, not a network drive.

Use `AUTELLIX_TEST_REPLICAS=1` to additionally test two actual model processes
and cancellation on one GPU with separate memory budgets. Use
`AUTELLIX_TEST_RESERVE=1` to verify resident KV reuse. These are separate checks
from the fake-worker IPC unit tests.

### Measured program workloads

`autellix.runtime.benchmark` executes real sequential/fork/join programs and
records measured program latency, output throughput, per-call TTFT, and service
statistics. Supply JSON calls with text prompts and parent IDs; the example
workload is a functional smoke workload, not a paper dataset.

```bash
python -m autellix.runtime.benchmark --backend vllm \
  --model HuggingFaceTB/SmolLM2-135M-Instruct --workload examples/real_workload.json \
  --policies fcfs,mlfq,plas,atlas \
  --engine-args '{"max_model_len":512,"max_num_seqs":2,"gpu_memory_utilization":0.35,"swap_space":0.25}' \
  --output outputs/real/results.json
```

`requirements/*-py310-linux.txt` pin the complete separately tested runtime
environments. The setup script uses these locks, not floating backend extras.

For one-command GPU, HTTP, and four-policy DAG validation, activate the relevant
backend environment and run `bash scripts/validate_backend.sh vllm /path/to/model`
(or `sglang`). Use a small chat model fitting the 512-token, 35% GPU-memory smoke
configuration. `AUTELLIX_TEST_STEPS=3`, `AUTELLIX_TEST_RESERVE=1`, and
`AUTELLIX_TEST_REPLICAS=1` enable the additional GPU integration cases described
above. `AUTELLIX_PYTHON` can select an environment without activating it.

## Simulator quick start

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
  runtime/       Real inference coordinator, policies, HTTP API, KV transfers
  *.py           Backward-compatible wrappers for old import paths
cuda/            Real GPU batched swap benchmark
integrations/    Pinned vLLM and SGLang internal scheduler integrations
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

## Simulation boundaries

The simulation APIs and `autellix.cli` experiments still model token execution,
cache hit rates, and engine workload. Their `vllm` / `vllm-opt` baseline labels
refer to cost models, not real backend measurements. Use the runtime and GPU
tests above for actual execution. Online `atlas` follows algorithm 1; the
explicit-parent equation (2) variant is available as `atlas-dag`.
