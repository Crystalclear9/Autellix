# Autellix

An **unofficial, independent reproduction attempt** of
["Autellix: An Efficient Serving Engine for LLM Agents as General Programs"
(arXiv:2502.13965v1)](https://arxiv.org/abs/2502.13965v1).
This is not the authors' implementation and is not affiliated with the paper's
authors. The paper is the method reference, not a source of executable code.

The repository contains **real GPU inference backends** and a separate CPU
simulator. Use `autellix.runtime` for real inference. The original
`AutellixClient`, `AsyncMultiLLMEngine`, and simulation CLI remain simulation
APIs for backward compatibility.

The goal is an independent implementation of the paper's methods. Correctness
checks use a small local model; reproducing the paper's large-model/A100
performance figures is not a requirement for using or validating this project.

### Reproduction scope

The real runtime implements online PLAS/ATLAS service inheritance, the shared
process table, FIFO priority queues with quantum demotion and anti-starvation,
preemption with KV transfers, scheduling windows with resident request reserves,
and the paper's 2048-token load-balancing/affinity rule. Online ATLAS follows
Algorithm 1: calls inherit the longest observed program service path; completion
updates it with the maximum of the previous value and inherited service plus
the call's measured execution time. It does not require a user-supplied DAG.

The default `--implementation paper` uses global-priority admission and an
execution plan lasting N decode steps. vLLM's native prefill/decode/swap queue
preferences do not choose the batch. Between boundaries, the executor advances
the existing plan and fills vacancies from prepared GPU reserves; it does not
run native batch selection again. Per-token tensor, output and KV bookkeeping
still runs: multi-step scheduling does not mean token execution stops needing
metadata updates. SGLang mixes admitted prefills and decodes and bypasses native
prefill selection during decode-only continuation.

`--implementation compat` explicitly restores the former backend-mediated
scheduler. Both modes retain real inference. Default N=8 and reserve count=1
enable both optimizations; these values, queue boundaries, quanta and starvation
thresholds are configurable project choices because the paper does not publish
all numerical settings. KV packing uses PyTorch CUDA operations and pinned host
memory. No identical CUDA kernels or CPU overhead are claimed. SGLang support
is an extension of the paper's vLLM-based design.

Small-model tests have exercised both pinned backends, resumed-output equality,
real host KV transfers, mid-window refill, routing, cancellation and HTTP
streaming/session cleanup. Multi-GPU tensor parallelism remains unverified on
hardware. Paper-scale datasets, reported speedups and exact implementation
equivalence have not been reproduced or established.

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
`/sessions`, `/requests/{id}`, and `/health`. The Python client automatically
creates and reuses a program session, including across threads, and annotates
each call with a unique call ID and thread ID. Use it as a context manager to
close the session on normal exit or an application exception; interpreter-exit
cleanup is best effort. Raw HTTP calls without a session remain one-call programs.
`GET /sessions` exposes shared arrival/completion timestamps, engine placement,
and active-call waiting/service statistics. Explicit `client.session()` scopes
remain available and automatically annotate calls within the scope.
Closed clients reject calls even with an explicit session ID. After a process
fork, create a new client; the child cannot reuse or close the parent's sessions.
If session cleanup fails during an application exception, the application
exception remains primary and the cleanup error is retained as its cause.
The checkpoint must supply a chat template for chat requests. Raw prompts or
token IDs are supported by the Python runtime.

```python
from autellix.runtime import InferenceClient

with InferenceClient("http://127.0.0.1:8000") as client:
    answer = client.chat(
        [{"role": "user", "content": "What is the capital of France?"}],
        temperature=0, max_tokens=32,
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
- vLLM: global-priority batch construction, native block ownership, GPU/CPU KV
  swap, batched CUDA packing, and multi-step plans (`--schedule-interval N`).
- SGLang: synchronous packed GPU/CPU KV swap for MHA token pools (page size 1),
  and resident radix-prefix reserves. `autellix_swap_space` in engine arguments
  limits pinned host KV storage in GiB (default 1); requests retain their
  generated tokens and restore computed prefixes on admission.
- `--schedule-interval N --overprovision K` freezes policy order for N decode
  steps and prepares up to K additional requests on the GPU. A completed active
  request is immediately replaced by a prepared reserve; traces record `refill`.
  Reserve preparation performs real prefill (and its first sampled token) or
  swap-in. Decode capacity stays at the requested batch size; reserve preparation
  can temporarily prefill up to batch-size + K requests. Memory pressure releases
  reserves rather than blocking the active cohort indefinitely.
- In default paper mode, vLLM reserves future KV slots when preparing a window
  and supports mixed prefill/decode batches without native phase preference.
  `--implementation compat --overprovision 0` on one GPU retains the old native
  cached multi-step execution; with reserves, compat uses the former wrapper.
- Use `--devices 0,1` for independent single-GPU replicas. vLLM tensor-parallel
  replicas use `--device-groups '0,1;2,3'` and
  `--engine-args '{"tensor_parallel_size":2,...}'`; worker hooks run on every
  rank. The TP path has an opt-in two-GPU test and has not been hardware-validated
  on the one-GPU development machine. SGLang remains one GPU per replica. PP and
  speculation are outside the adapters' supported mode.

### Validation

```bash
# CPU policy / IPC / protocol checks; GPU tests explicitly skip without opt-in.
python -m unittest discover -s tests -v

# Real generation, preemption, resumed-output equality, session inheritance.
AUTELLIX_GPU_BACKEND=vllm AUTELLIX_TEST_MODEL=/path/to/model \
  AUTELLIX_TEST_CUDA_SWAP=1 \
  python -m unittest discover -s tests -p test_gpu_runtime.py -v

# Repeat in the SGLang environment with AUTELLIX_GPU_BACKEND=sglang.
# Paper multi-step validation: add AUTELLIX_TEST_STEPS=3.
# For the former adapter, also set AUTELLIX_TEST_IMPLEMENTATION=compat.
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
Set both `AUTELLIX_TEST_STEPS=3` and `AUTELLIX_TEST_RESERVE=1` to assert actual
mid-window refill, and `AUTELLIX_TEST_TP=1` for the optional two-GPU vLLM test.

### Measured program workloads

`autellix.runtime.benchmark` executes real sequential/fork/join programs and
records measured program latency, output throughput, per-call TTFT, and service
statistics. Supply JSON calls with text prompts and parent IDs; the example
workload is a functional smoke workload, not a paper dataset.

For an optional arrival-rate sweep, add `--arrival-rates 1,2,4 --programs 100
--seed 42`. The same sampled program trace is reused for every policy. JSONL is
also accepted. Multiple files after `--workload` create a mixed workload, sampled
equally by dataset and then by program. Inputs must contain real text prompts;
token-length-only simulation traces are rejected. `arrival_time` and `think_time`
are seconds. Set a call's `append_parent_outputs` to false for recorded prompts
that already include their original history.

Results include program response time, DAG critical-path response time divided
by total generated tokens across all threads (mean/P95/P99), per-program values,
arrival times, configuration, dependency versions, and a trace hash. Critical-path
time is the longest dependency path's measured call latencies plus external
`think_time`; measured call latency includes tokenization and queueing. Zero-output
programs have null per-token latency rather than dividing by zero.

Native vLLM baselines are opt-in with `--policies vllm,vllm-opt,vllm-opt-multistep,mlfq,plas,atlas`.
They retain the original scheduling order and original KV transfer implementation;
Autellix observes their execution only to collect comparable metrics. In pinned
vLLM 0.6.1, chunked prefill and native multi-step are mutually exclusive:
`vllm-opt` enables prefix caching and chunked prefill, while
`vllm-opt-multistep` enables prefix caching and native multi-step (8 steps by
default, configurable with `--native-opt-steps`). These separate supported
profiles are not advertised as the paper's combined optimized baseline.

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
quanta, or beta. For real inference, `autellix.runtime.PolicyConfig` uses seconds:
boundaries `0,.02,.04,.08,.16,.32,.64,inf`, quanta
`.01,.02,.04,.08,.16,.32,.64`, beta `8`, schedule interval `8`, one reserve,
and `implementation="paper"` by default. Pass a custom `PolicyConfig` to
`InferenceEngine` to tune these. The server and real benchmark expose
`--implementation paper|compat`, `--schedule-interval` and `--overprovision`.

The separate simulator uses:

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
dataset importers, paper presets, CLI smoke checks, and backend adapter imports.
Runtime regressions additionally cover process-table consistency,
automatic client sessions, concurrent cancellation, admission-time duplicate
request IDs, scheduling windows and measured DAG workload metrics. Opt-in GPU
tests exercise the real integrations described above.

## Simulation boundaries

The simulation APIs and `autellix.cli` experiments still model token execution,
cache hit rates, and engine workload. Their `vllm` / `vllm-opt` baseline labels
refer to cost models, not real backend measurements. Use the runtime and GPU
tests above for actual execution. Online `atlas` follows algorithm 1; the
explicit-parent equation (2) variant is available as `atlas-dag`.

The simulator routes by input token count, without using future output length.
Scheduling windows start when work is available; quantum expiry and per-call
anti-starvation are evaluated at window boundaries, preserving current queue
FIFO order. Running calls compete with waiting calls at those boundaries, and
prepared reserves can fill vacancies inside a window. Swap costs apply only
when a resident call actually leaves the selected cohort. Window length is
measured in simulated model ticks, whereas the real runtime counts decode
steps. Simulated locality benefits require completed work on the target engine;
assigning a new request there alone does not create a cache hit.
