"""Plot measured paper-workload latency curves and paired changes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", help="model and GPU description for the curves")
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    result = json.loads(Path(args.summary).read_text())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"native": "#62717D", "plas": "#007F73", "atlas": "#007F73", "native-default": "#65529C"}

    def save(fig, name):
        for suffix in ("png", "svg"):
            path = output / f"{name}.{suffix}"
            fig.savefig(path, dpi=180, facecolor="white")
            if suffix == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
        plt.close(fig)

    trials = [t for t in result["trials"] if t["rate"] is not None
              and (not t.get("paper_figures") or set(t["paper_figures"]) & {12, 13})]
    workloads = ("sharegpt", "bfcl", "lats", "mixed")
    models = sorted({Path(t["model"]).name for t in trials})
    conditions = list(dict.fromkeys(t["condition"] for t in trials))

    def curve(ax, workload, metric):
        rows = [t for t in trials if t.get("workload_family", t["workload"]) == workload]
        for index, condition in enumerate(conditions):
            results = defaultdict(list)
            for trial in rows:
                value = trial["paper_token_latency"][metric]
                if trial["condition"] == condition and value is not None:
                    results[trial["rate"]].append(value)
            if results:
                rates = sorted(results)
                ax.plot(rates, [mean(results[rate]) for rate in rates], "o-",
                        color=colors.get(condition, f"C{index}"), label=condition)
        ax.set(title=f"{workload.upper()} · {metric.upper()}", xlabel="Offered programs / second",
               ylabel="Program token latency (s/token)")
        ax.grid(alpha=.18)
        if ax.lines:
            ax.legend(frameon=False, fontsize=8)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    for ax, workload in zip(axes.flat, workloads):
        curve(ax, workload, "mean")
    fig.suptitle(args.title or " / ".join(models), fontsize=13)
    save(fig, "paper-workload-curves")

    fig, axes = plt.subplots(4, 2, figsize=(11, 13), layout="constrained")
    for (p95, p99), workload in zip(axes, workloads):
        curve(p95, workload, "p95")
        curve(p99, workload, "p99")
    fig.suptitle(args.title or " / ".join(models), fontsize=13)
    save(fig, "paper-workload-tail-curves")

    rows = [row for row in result["comparisons"] if row["rate"] is not None]
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
