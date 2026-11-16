"""Plot program latency against offered program rate from matrix summaries."""
import argparse
import json
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary")
    parser.add_argument("--output", required=True, help="output path without suffix")
    parser.add_argument("--title", default="ShareGPT whole conversations / Llama-3.1-8B-Instruct / P40")
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    rows = json.loads(Path(args.summary).read_text(encoding="utf-8"))["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for condition in dict.fromkeys(row["condition"] for row in rows):
        selected = [row for row in rows if row["condition"] == condition]
        rates = sorted({row["rate"] for row in selected})
        for ax, field, title in zip(axes,
                                   ["mean_seconds_per_output_token", "p95_seconds_per_output_token"],
                                   ["Mean program token latency", "P95 program token latency"]):
            values = [mean(row[field] for row in selected if row["rate"] == rate) for rate in rates]
            ax.plot(rates, values, marker="o", label=condition)
            ax.set(xscale="log", xlabel="Offered programs / second", ylabel="Program response seconds / output token",
                   title=title, xticks=sorted({row["rate"] for row in rows}))
            ax.set_xticklabels([str(rate) for rate in sorted({row["rate"] for row in rows})])
            ax.grid(alpha=.25)
    axes[0].legend(frameon=False)
    fig.suptitle(args.title, fontsize=12)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
