"""Compare Pascal dense MLP values and peak memory with token chunking."""
import argparse
import json
from pathlib import Path

import torch
from integrations.sglang.legacy_gpu import token_chunked_forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(2026)
    device, dtype = "cuda", torch.float16
    up = torch.randn(28672, 4096, device=device, dtype=dtype) * .02
    down = torch.randn(4096, 14336, device=device, dtype=dtype) * .02
    x = torch.randn(16000, 4096, device=device, dtype=dtype)

    def forward(self, inputs):
        gate, values = torch.nn.functional.linear(inputs, up).chunk(2, dim=-1)
        return torch.nn.functional.linear(torch.nn.functional.silu(gate) * values, down)

    results = {}
    with torch.inference_mode():
        for name, fn in [("full", forward), ("chunked", token_chunked_forward(forward, 512))]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()
            output = fn(None, x)
            torch.cuda.synchronize()
            results[name] = dict(temporary_peak_bytes=torch.cuda.max_memory_allocated() - baseline)
            sample = output[::31].clone()
            del output
            if name == "full":
                expected = sample
            else:
                torch.testing.assert_close(sample, expected, rtol=.02, atol=.02)
                results["max_absolute_error"] = float((sample - expected).abs().max())
    results.update(tokens=len(x), hidden_size=4096, intermediate_size=14336,
                   chunk_size=512, gpu=torch.cuda.get_device_name(), dtype=str(dtype))
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
