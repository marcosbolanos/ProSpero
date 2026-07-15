from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from prospero.plotting_style import COLORS, set_prospero_style

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "rl_vs_og" / "prosst_vocab_ablation_k128"
TASKS = ["AAV", "LGK"]

RESTRICTED = {}
UNRESTRICTED = {}


@dataclass(frozen=True)
class Method:
    label: str
    color: str
    marker: str
    root: Path


def load_seed(path: Path) -> dict[int, float]:
    try:
        with path.open("rb") as handle:
            data = pickle.load(handle)
    except Exception as exc:
        print(f"skip unreadable {path}: {exc}")
        return {}
    return {
        int(k): float(v["Best score"])
        for k, v in data.items()
        if isinstance(k, int) and isinstance(v, dict) and "Best score" in v
    }


def aggregate(root: Path):
    rows = [load_seed(p) for p in sorted(root.glob("seed_*.pkl"))]
    xs = np.arange(1, 11)
    means, sems, counts = [], [], []
    for it in xs:
        vals = np.array([row[it] for row in rows if it in row], dtype=float)
        counts.append(int(len(vals)))
        if len(vals) == 0:
            means.append(np.nan)
            sems.append(np.nan)
        else:
            means.append(float(vals.mean()))
            sems.append(float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0)
    return xs, np.array(means), np.array(sems), counts


def parse_args():
    parser = argparse.ArgumentParser(description="Plot restricted vs unrestricted ProSST vocabulary ablation.")
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--restricted-root", default=None, help="Root containing TASK/seed_*.pkl for restricted runs.")
    parser.add_argument("--unrestricted-root", default=None, help="Root containing TASK/seed_*.pkl for unrestricted runs.")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--budget", type=int, default=128)
    return parser.parse_args()


def main():
    global OUT, TASKS
    args = parse_args()
    OUT = Path(args.output_dir)
    TASKS = list(args.tasks)
    if args.restricted_root is not None:
        base = Path(args.restricted_root)
        for task in TASKS:
            RESTRICTED[task] = base / task
    if args.unrestricted_root is not None:
        base = Path(args.unrestricted_root)
        for task in TASKS:
            UNRESTRICTED[task] = base / task
    missing = [task for task in TASKS if task not in RESTRICTED or task not in UNRESTRICTED]
    if missing:
        raise ValueError(
            "Explicit --restricted-root and --unrestricted-root are required; "
            f"missing task roots for: {', '.join(missing)}"
        )
    set_prospero_style()
    plt.rcParams.update({
        "axes.titlesize": 17,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
    })
    methods_by_task = {
        task: [
            Method("Restricted amino acid vocabulary", COLORS["prosst"], "s", RESTRICTED[task]),
            Method("Unrestricted amino acid vocabulary", "#D55E00", "^", UNRESTRICTED[task]),
        ]
        for task in TASKS
    }

    n_tasks = len(TASKS)
    ncols = min(4, n_tasks)
    nrows = int(math.ceil(n_tasks / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.8 * ncols, 4.7 * nrows),
        sharex=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()
    legend_seen = set()
    summary = []
    for ax, task in zip(axes_flat, TASKS):
        for method in methods_by_task[task]:
            x, y, e, counts = aggregate(method.root)
            valid = np.isfinite(y)
            label = method.label if method.label not in legend_seen else "_nolegend_"
            legend_seen.add(method.label)
            ax.plot(
                x[valid], y[valid],
                color=method.color,
                marker=method.marker,
                linewidth=2.8,
                markersize=7,
                label=label,
            )
            ax.fill_between(x[valid], y[valid] - e[valid], y[valid] + e[valid], color=method.color, alpha=0.12, linewidth=0)
            summary.append(f"{task} {method.label}: counts={counts} round10={y[-1]:.6g} sem={e[-1]:.3g}")
        ax.set_title(task, loc="left", pad=6)
        ax.set_xlim(1, 10)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Optimization round")
        ax.set_ylabel("Mean max fitness")
        ax.grid(True, axis="y")
    for ax in axes_flat[n_tasks:]:
        ax.axis("off")

    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle(f"K={args.budget} ProSST vocabulary restriction ablation", fontsize=20, fontweight=600, y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"prosst_restricted_vs_unrestricted_vocab_k{args.budget}_mean_max.png"
    pdf = OUT / f"prosst_restricted_vs_unrestricted_vocab_k{args.budget}_mean_max.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    summary_path = OUT / "plot_summary.txt"
    summary_path.write_text("Written:\n" + str(png) + "\n" + str(pdf) + "\n\nSummary:\n" + "\n".join(summary) + "\n")
    print(png)
    print(pdf)
    print(summary_path)
    print("\n".join(summary))


if __name__ == "__main__":
    main()
