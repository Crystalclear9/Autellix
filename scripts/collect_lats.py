"""Collect real LATS calls with a local LLM and HotpotQA's supplied documents."""
import argparse
import hashlib
import importlib.util
import json
import logging
import random
import re
import shutil
import string
import sys
import time
import types
from pathlib import Path

REVISION = "853d81614607dd27433faf17c7b0a7d660f95d22"


def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


class DocumentEnv:
    """Search/lookup/finish over a question's unmodified HotpotQA documents."""
    def __init__(self, rows):
        self.rows = rows

    def reset(self, idx):
        self.row = self.rows[idx]
        self.pages = dict(zip(self.row["context"]["title"], self.row["context"]["sentences"]))
        self.page, self.lookup_key, self.lookup_index = [], None, 0
        return "Question: " + self.row["question"]

    def step(self, action):
        kind, _, argument = action.partition("[")
        argument = argument.rstrip("]")
        if kind == "finish":
            em = float(normalize(argument) == normalize(self.row["answer"]))
            return "Episode finished.", em, True, {"em": em, "answer": argument}
        if kind == "search":
            exact = next((t for t in self.pages if normalize(t) == normalize(argument)), None)
            if exact is None:
                matches = sorted(self.pages, key=lambda t: -len(set(normalize(t).split()) &
                                                                set(normalize(argument).split())))[:5]
                return "Could not find that page. Similar titles: " + str(matches), 0, False, {}
            self.page, self.lookup_key, self.lookup_index = self.pages[exact], None, 0
            return " ".join(self.page[:5]), 0, False, {}
        if kind == "lookup":
            if self.lookup_key != argument:
                self.lookup_key, self.lookup_index = argument, 0
            matches = [s for s in self.page if argument.lower() in s.lower()]
            index = self.lookup_index
            self.lookup_index += 1
            result = f"(Result {index+1}/{len(matches)}) {matches[index]}" if index < len(matches) else "No more results."
            return result, 0, False, {}
        return "Invalid action. Use Search[title], Lookup[keyword], or Finish[answer].", 0, False, {}


class SearchBudget(Exception):
    pass


class SearchExhausted(Exception):
    pass


class Recorder:
    def __init__(self, engine, tokenizer, max_calls):
        self.engine, self.tokenizer, self.max_calls = engine, tokenizer, max_calls

    def begin(self):
        self.session = self.engine.start_session()
        self.calls, self.parents = [], []
        self.last_finished = time.monotonic()

    def gpt(self, prompt, model=None, temperature=1., max_tokens=100, n=1, stop=None):
        return self.generate_many([prompt], temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)[0]

    def generate_many(self, prompts, *, temperature=1., max_tokens=100, n=1, stop=None):
        if not prompts:
            return []
        if len(self.calls) + len(prompts) * n > self.max_calls:
            raise SearchBudget()
        inputs = [self.tokenizer.apply_chat_template([dict(role="user", content=prompt)],
                                                     tokenize=True, add_generation_prompt=True)
                  for prompt in prompts]
        think_time = time.monotonic() - self.last_finished
        group, futures = [], []
        sampling = dict(temperature=temperature, max_tokens=max_tokens)
        if stop:
            sampling["stop"] = stop
        for prompt, ids in zip(prompts, inputs):
            for _ in range(n):
                cid = str(len(self.calls) + len(group))
                futures.append(self.engine.submit(self.session, input_ids=ids, call_id=cid, sampling=sampling))
                group.append(dict(call_id=cid, prompt=prompt, input_ids=ids, input_tokens=len(ids),
                                  parents=list(self.parents), think_time=think_time, append_parent_outputs=False))
        outputs = []
        for call, future in zip(group, futures):
            result = future.result(1800)
            count = result["completion_tokens"]
            if count == 0:
                raise RuntimeError("LATS collection returned an empty token sequence")
            call.update(max_tokens=count, reference_output_tokens=count,
                        reference_output=result["text"], sampling={"ignore_eos": True})
            outputs.append(result["text"])
        self.calls.extend(group)
        self.parents = [c["call_id"] for c in group]
        print(f"CAPTURE calls={len(self.calls)} group_outputs={len(outputs)}", flush=True)
        self.last_finished = time.monotonic()
        return [outputs[start:start+n] for start in range(0, len(outputs), n)]


def parallel_values(lats, recorder, task, question, candidates, samples, cache_value=True):
    trajectories = lats.get_unique_trajectories(lats.failed_trajectories)
    seen, prompt_slots, pending = set(), {}, []
    values = [0] * len(candidates)
    for index, candidate in enumerate(candidates):
        if candidate in seen:
            continue
        seen.add(candidate)
        prompt = task.value_prompt_wrap(question, candidate, trajectories, lats.reflection_map)
        if cache_value and prompt in task.value_cache:
            values[index] = task.value_cache[prompt]
            continue
        slot = prompt_slots.setdefault(prompt, len(pending)) if cache_value else len(pending)
        if slot == len(pending):
            pending.append((prompt, []))
        pending[slot][1].append(index)
    outputs = recorder.generate_many([prompt for prompt, _ in pending], n=samples)
    for (prompt, indices), responses in zip(pending, outputs):
        value = task.value_outputs_unwrap(responses)
        logging.info("VALUE PROMPT: %s\nVALUE OUTPUTS: %s\nVALUES: %s", prompt, responses, value)
        if cache_value:
            task.value_cache[prompt] = value
        for index in indices:
            values[index] = value
    return values


def load_official(source, recorder, env, tokenizer, *, parallel_evaluations=False):
    sys.path.insert(0, str(source))
    models = types.ModuleType("models")
    models.gpt = recorder.gpt
    sys.modules["models"] = models
    wiki = types.ModuleType("wikienv")
    wiki.WikiEnv = lambda: env
    sys.modules["wikienv"] = wiki
    wrappers = types.ModuleType("wrappers")
    wrappers.HotPotQAWrapper = lambda instance, **kwargs: instance
    wrappers.LoggingWrapper = lambda instance: instance
    sys.modules["wrappers"] = wrappers
    # Keep the original prompts and tree search. Use the serving tokenizer for
    # prompt-length selection instead of downloading an unrelated GPT-2 model.
    spec = importlib.util.spec_from_file_location("hotpotqa", source / "hotpotqa.py")
    task_module = importlib.util.module_from_spec(spec)
    task_module.__dict__["_serving_tokenizer"] = tokenizer
    text = (source / "hotpotqa.py").read_text().replace(
        'GPT2Tokenizer.from_pretrained("gpt2")', "_serving_tokenizer")
    exec(compile(text, str(source / "hotpotqa.py"), "exec"), task_module.__dict__)
    import lats
    if parallel_evaluations:
        from functools import partial
        lats.get_values = partial(parallel_values, lats, recorder)
    original_select = lats.select_node
    def select_node(node):
        selected = original_select(node)
        if selected is None:
            raise SearchExhausted()
        return selected
    lats.select_node = select_node
    return lats, task_module.HotPotQATask


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--backend", choices=("sglang", "vllm"), default="sglang")
    p.add_argument("--chat-template", type=Path, help="local Jinja template for base-model tokenizers")
    p.add_argument("--device", default="2")
    p.add_argument("--output", required=True)
    p.add_argument("--programs", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-calls", type=int, default=400)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--parallel-evaluations", action="store_true",
                   help="evaluate independent candidates in each search stage as one parallel call group")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.iterations < 1:
        p.error("--iterations must be positive")
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from autellix.runtime import InferenceEngine, ReplicaConfig, PolicyConfig
    rows = pq.read_table(args.data).to_pylist()
    indices = random.Random(args.seed).sample(range(len(rows)), args.programs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    template_sha256 = None
    if args.chat_template:
        template = args.chat_template.read_bytes()
        tokenizer.chat_template = template.decode("utf-8")
        template_sha256 = hashlib.sha256(template).hexdigest()
    source, out = Path(args.source).resolve(), Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    programs, outcomes = [], []
    if args.resume and out.exists():
        programs = json.loads(out.read_text())
        saved = json.loads(out.with_suffix(".manifest.json").read_text())
        if (saved.get("backend", "sglang") != args.backend or saved["search"]["iterations"] != args.iterations
                or saved["search"]["max_calls"] != args.max_calls or saved["seed"] != args.seed
                or saved["search"].get("parallel_evaluations", False) != args.parallel_evaluations
                or saved.get("chat_template_sha256") != template_sha256 or saved["model"] != args.model
                or saved["indices"] != indices):
            p.error("resume requires the same model, backend, template, search configuration, and question sample")
        outcomes = saved["outcomes"]
    completed = {p["program_id"] for p in programs}
    if completed:
        print(f"RESUME programs={len(completed)}", flush=True)
    if args.backend == "vllm":
        options = dict(autellix_mode="vllm-opt", observe_native=False, dtype="bfloat16",
                       max_model_len=32768, max_num_seqs=8, max_num_batched_tokens=2048,
                       num_gpu_blocks_override=4096, enforce_eager=False, seed=args.seed,
                       completion_only=True, preemption_mode=None)
    else:
        options = dict(autellix_legacy_gpu=True, autellix_native_baseline=True, dtype="float16",
                       context_length=32768, max_running_requests=4, max_total_tokens=32768,
                       max_prefill_tokens=32768, mem_fraction_static=.9, random_seed=args.seed)
    with InferenceEngine([ReplicaConfig(args.backend, args.model, args.device, options)],
                         policy=PolicyConfig(policy="fcfs", overprovision=0,
                                             schedule_interval=1 if args.backend == "vllm" else 8),
                         scheduler_trace=args.backend == "sglang") as engine:
        recorder = Recorder(engine, tokenizer, args.max_calls)
        lats, task_class = load_official(source, recorder, DocumentEnv(rows), tokenizer,
                                         parallel_evaluations=args.parallel_evaluations)
        search_log = engine.state_dir / "lats-search.log"
        if args.resume and out.with_suffix(".search.log").exists():
            shutil.copy2(out.with_suffix(".search.log"), search_log)
        logging.basicConfig(filename=search_log, level=logging.INFO, force=True,
                            format="%(asctime)s - %(levelname)s - %(message)s")
        config = types.SimpleNamespace(backend="local-llama", temperature=1., n_generate_sample=5,
                                       n_evaluate_sample=1, prompt_sample="cot", log=str(search_log))
        for index in indices:
            if "hotpot-" + rows[index]["id"] in completed:
                continue
            random.seed(args.seed + index)
            recorder.begin()
            lats.gpt = recorder.gpt
            status, reward = "completed", None
            try:
                result = lats.lats_search(config, task_class(), index, iterations=args.iterations, to_print=False)
                reward = result[3]
            except SearchBudget:
                status = f"{args.max_calls}-call search budget"
            except SearchExhausted:
                status = "tree exhausted"
            finally:
                engine.end_session(recorder.session)
            item = dict(program_id="hotpot-" + rows[index]["id"], dataset="lats", calls=recorder.calls)
            programs.append(item)
            outcomes.append(dict(id=item["program_id"], index=index, status=status, reward=reward,
                                 calls=len(recorder.calls), output_tokens=sum(c["max_tokens"] for c in recorder.calls)))
            out.write_text(json.dumps(programs, ensure_ascii=False, indent=2))
            for handler in logging.getLogger().handlers:
                handler.flush()
            shutil.copy2(search_log, out.with_suffix(".search.log"))
            print(json.dumps(outcomes[-1]), flush=True)
            manifest = dict(source="https://github.com/andyz245/LanguageAgentTreeSearch", revision=REVISION,
                            model=args.model, backend=args.backend, chat_template_sha256=template_sha256,
                            seed=args.seed, indices=indices, outcomes=outcomes,
                            search=dict(iterations=args.iterations, expansion_width=5, evaluation_samples=1,
                                        temperature=1., max_output_tokens=100, max_calls=args.max_calls,
                                        parallel_evaluations=args.parallel_evaluations),
                            environment="HotpotQA supplied distractor documents; local Search/Lookup/Finish",
                            adaptations=["local Llama replaces hosted GPT",
                                         "serving tokenizer selects original long/short prompt templates",
                                         "Search/Lookup use the question's supplied documents",
                                         "exhausted tree terminates instead of repeatedly selecting None"],
                            source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                           for f in source.glob("*.py")},
                            data_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in
                                         (sorted(Path(args.data).glob("*.parquet")) if Path(args.data).is_dir() else [Path(args.data)])},
                            replay_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                            protocol="Actual local-model tree search capture; each n-sample group runs concurrently; fixed prompts, generated lengths and group barriers in replay.")
            out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
