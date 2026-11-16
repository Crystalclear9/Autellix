"""Plot recorded GPU resources and paired per-program response times."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--prefix", default="three-gpu")
    args = parser.parse_args()
    rows = json.loads(Path(args.input).read_text())
    if args.conditions:
        rows = [row for row in rows if row["condition"] in args.conditions]
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(rows), figsize=(6*len(rows), 6), squeeze=False)
    colors = ["#2563eb", "#d97706", "#15803d"]
    for column, row in enumerate(rows):
        by_gpu = defaultdict(list)
        for sample in row["gpu_samples"]:
            for gpu in sample["gpus"]:
                by_gpu[gpu["index"]].append((sample["seconds"]/60, gpu["used_mib"]/1024,
                                              gpu["utilization"]))
        for index, samples in by_gpu.items():
            x, memory, utilization = zip(*samples)
            axes[0, column].plot(x, memory, color=colors[index % 3], label=f"GPU {index}")
            axes[1, column].plot(x, utilization, color=colors[index % 3], linewidth=.8, alpha=.8)
        axes[0, column].set_title(row["condition"])
        axes[0, column].set_ylabel("GPU memory used (GiB)")
        axes[0, column].legend(loc="lower right", ncol=3)
        axes[0, column].set_ylim(0, 25)
        axes[1, column].set_ylim(0, 105)
        axes[1, column].set_ylabel("nvidia-smi GPU utilization (%)")
        axes[1, column].set_xlabel("Elapsed time (minutes)")
        for axis in axes[:, column]:
            axis.grid(alpha=.2)
    model = Path(rows[0]["model"]).name
    pp = rows[0]["effective_engine_args"].get("pp_size", 1)
    layout = f"pipeline parallelism = {pp}" if pp > 1 else f"replicas = {len(rows[0]['device_groups'])}"
    fig.suptitle(f"{model} / {layout}")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(destination / f"{args.prefix}-usage.{suffix}", dpi=160)
    plt.close(fig)

    baseline = next((row for row in rows if row["arm"] == "sglang-native"), rows[0])
    counts = defaultdict(int)
    for call in baseline["calls"]:
        counts[call["workload_program"]] += call["completion_tokens"]
    programs = sorted(counts, key=counts.get)
    fig, axis = plt.subplots(figsize=(11, 4))
    width = .8/len(rows)
    for index, row in enumerate(rows):
        x = [i + (index-(len(rows)-1)/2)*width for i in range(len(programs))]
        axis.bar(x, [row["programs"][pid]/60 for pid in programs], width=width, label=row["condition"])
    axis.set_xticks(range(len(programs)), [str(counts[pid]) for pid in programs],
                   rotation=45 if len(programs) > 12 else 0)
    axis.set_xlabel("Total output tokens per program (sorted)")
    axis.set_ylabel("Program response time (minutes)")
    axis.set_title(f"Same {len(programs)} complete ShareGPT programs, paired arrival trace")
    axis.legend()
    axis.grid(axis="y", alpha=.2)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(destination / f"{args.prefix}-programs.{suffix}", dpi=160)


if __name__ == "__main__":
    main()
