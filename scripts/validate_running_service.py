"""Exercise an already-running HTTP service and save its deployment checks."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

from autellix.runtime.client import InferenceClient
from autellix.runtime.telemetry import GPUMonitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    def get(path):
        with urlopen(args.url + path, timeout=30) as response:
            return json.load(response)

    def generate(index):
        with InferenceClient(args.url, timeout=900, program_id=f"deployment-{index}") as client:
            return client.chat([dict(role="user", content="Name the capital of France. /no_think")],
                               temperature=0, max_tokens=32)

    started = time.monotonic()
    with GPUMonitor() as monitor:
        health = get("/health")
        models = get("/v1/models")
        assert models["data"][0]["id"] == args.model
        with ThreadPoolExecutor(max_workers=6) as pool:
            concurrent = list(pool.map(generate, range(6)))
        assert all(row["usage"]["completion_tokens"] > 0 for row in concurrent)
        assert all(row["choices"][0]["message"]["content"].strip() for row in concurrent)

        messages = [dict(role="user", content="What is 2 plus 3? /no_think")]
        with InferenceClient(args.url, timeout=900, program_id="deployment-history") as client:
            first = client.chat(messages, temperature=0, max_tokens=32)
            second = client.chat(messages, temperature=0, max_tokens=32)
            assert second["autellix"]["metrics"]["inherited"] > 0
            chunks = list(client.chat(messages, temperature=0, max_tokens=32, stream=True))
            streamed = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
            assert streamed == first["choices"][0]["message"]["content"]
            assert chunks[-1]["choices"][0]["finish_reason"] in {"stop", "length"}

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        def long_generate(index):
            sentence = ["The river flows through the green valley. ",
                        "The mountain rises above the quiet village. "][index]
            text = tokenizer.decode(tokenizer.encode(sentence * 600, add_special_tokens=False)[:3900])
            with InferenceClient(args.url, timeout=900, program_id=f"deployment-long-prefill-{index}") as client:
                return client.chat([dict(role="user", content=text + "\nSummarize in one sentence. /no_think")],
                                   temperature=0, max_tokens=32)
        with ThreadPoolExecutor(max_workers=2) as pool:
            long_prompts = list(pool.map(long_generate, range(2)))
        assert all(3800 <= row["usage"]["prompt_tokens"] < 4096 for row in long_prompts)
        assert all(row["usage"]["completion_tokens"] > 0 for row in long_prompts)
        sessions = get("/sessions")
        assert sessions == {}
        final_health = get("/health")
        assert all(load == 0 for load in final_health["loads"])

    result = dict(health=health, models=models, concurrent=concurrent, history=[first, second],
                  streamed_text=streamed, streaming_matches=True, long_prompts=long_prompts,
                  sessions_empty=True, final_health=final_health,
                  elapsed_seconds=time.monotonic()-started, gpu_samples=monitor.samples,
                  gpu_monitor_errors=monitor.errors)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(concurrent_completed=len(concurrent), streaming_matches=True,
                          inherited=second["autellix"]["metrics"]["inherited"],
                          long_prompt_usage=[row["usage"] for row in long_prompts], sessions_empty=True)))


if __name__ == "__main__":
    main()
