"""Measure cached-prefix SDPA on the actual GPU, with numerical comparison."""
import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch
    from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
    from integrations.sglang.attention import prefix_attention

    torch.manual_seed(42)
    rows = []
    for total, count in ((512, 1), (2048, 1), (2048, 64), (2048, 2048)):
        q = torch.randn(count, 4, 64, device="cuda")
        k, v = (torch.randn(total, 1, 64, device="cuda") for _ in range(2))
        inputs = (q, torch.empty_like(q), k, v, torch.arange(total, device="cuda")[None],
                  torch.tensor([0], device="cuda"), torch.tensor([total], device="cuda"),
                  torch.tensor([total-count], device="cuda"), torch.tensor([count], device="cuda"))
        outputs, timings = {}, {}
        for name, operation in (("padded", TorchNativeAttnBackend._run_sdpa_forward_extend),
                                ("suffix", prefix_attention)):
            samples = []
            for index in range(12):
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = operation(None, *inputs, scaling=.125, enable_gqa=True, causal=True)
                torch.cuda.synchronize()
                if index >= 2:
                    samples.append(time.perf_counter() - start)
            outputs[name] = result.clone()
            timings[name] = statistics.median(samples)
        torch.testing.assert_close(outputs["suffix"], outputs["padded"], atol=1e-5, rtol=1e-4)
        rows.append(dict(total_tokens=total, query_tokens=count, seconds=timings,
                         speedup=timings["padded"]/timings["suffix"],
                         max_absolute_error=(outputs["suffix"]-outputs["padded"]).abs().max().item()))
    result = dict(device=torch.cuda.get_device_name(), dtype="float32", cases=rows)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
