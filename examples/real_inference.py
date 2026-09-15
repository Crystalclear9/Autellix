"""Run in a pinned backend environment: python examples/real_inference.py --help."""
import argparse
import json

from autellix.runtime import InferenceEngine, ReplicaConfig, PolicyConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("vllm", "sglang"), default="vllm")
    parser.add_argument("--engine-args", default="{}")
    args = parser.parse_args()
    replicas = [ReplicaConfig(args.backend, args.model, engine_args=json.loads(args.engine_args))]
    with InferenceEngine(replicas, policy=PolicyConfig(policy="atlas")) as engine:
        pid = engine.start_session()
        try:
            first = engine.submit(pid, messages=[{"role": "user", "content": "Name a European capital."}],
                                  sampling={"temperature": 0, "max_tokens": 32}).result(180)
            second = engine.submit(pid, messages=[
                {"role": "user", "content": "Name a European capital."},
                {"role": "assistant", "content": first["text"]},
                {"role": "user", "content": "Which country is it in?"}],
                sampling={"temperature": 0, "max_tokens": 32}).result(180)
            print(json.dumps([first, second], ensure_ascii=False, indent=2))
        finally:
            engine.end_session(pid)


if __name__ == "__main__":
    main()
