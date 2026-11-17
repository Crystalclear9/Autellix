"""Plot measured paper-workload latency curves and paired changes."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    result = json.loads(Path(args.summary).read_text())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"native": "#62717D", "plas": "#007F73", "atlas": "#007F73"}

    def save(fig, name):
        for suffix in ("png", "svg"):
            path = output / f"{name}.{suffix}"
            fig.savefig(path, dpi=180, facecolor="white")
            if suffix == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    for ax, workload in zip(axes.flat, ("sharegpt", "bfcl", "lats", "mixed")):
        selected = [t for t in result["trials"] if t["workload"] == workload]
        if not selected:
            ax.set(title=workload.upper())
            ax.text(.5, .5, "Pending completed pairs", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        for condition in dict.fromkeys(t["condition"] for t in selected):
            rows = sorted((t for t in selected if t["condition"] == condition), key=lambda t: t["rate"])
            x = [t["rate"] for t in rows]
            color = colors[condition]
            ax.plot(x, [t["paper_token_latency"]["mean"] for t in rows], "o-", color=color, label=condition + " mean")
            ax.plot(x, [t["paper_token_latency"]["p95"] for t in rows], "x--", color=color, alpha=.65, label=condition + " P95")
        ax.set(title=workload.upper(), xlabel="Offered programs / second", ylabel="Program token latency (s/token)")
        ax.grid(alpha=.18)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Llama-3.1-8B-Instruct / P40 / paired on the same GPU", fontsize=13)
    save(fig, "paper-workload-curves")

    rows = result["comparisons"]
    fig, axes = plt.subplots(1, 3, figsize=(14, max(5, len(rows)*.45)), layout="constrained", sharey=True)
    labels = [f"{r['workload']} {r['rate']:g} / {r['condition']}" for r in rows]
    for ax, field, title in zip(axes,
            ("mean_response_change_pct", "mean_paper_latency_change_pct", "throughput_change_pct"),
            ("Mean response change (%)\nLower is better", "Mean token latency change (%)\nLower is better", "Output throughput change (%)\nHigher is better")):
        values = [r[field] for r in rows]
        good = [v >= 0 if field == "throughput_change_pct" else v <= 0 for v in values]
        ax.barh(range(len(rows)), values, color=["#007F73" if g else "#B55D42" for g in good])
        ax.axvline(0, color="#38434C", linewidth=.8)
        ax.set(title=title, yticks=range(len(rows)), yticklabels=labels)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.18)
        for i, value in enumerate(values):
            ax.annotate(f"{value:+.1f}", (value, i), xytext=(3 if value >= 0 else -3, 0),
                        textcoords="offset points", va="center", ha="left" if value >= 0 else "right", fontsize=8)
        ax.margins(x=.23)
    save(fig, "paper-workload-changes")


if __name__ == "__main__":
    main()
