from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from prospero.plotting_style import COLORS, set_prospero_style

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "outputs/0423_epistasis/epistasis_additivity_all_tasks.json"
OUT = ROOT / "outputs/rl_vs_og/epistasis_additivity"

TASK_ORDER = ["AAV", "LGK", "GFP", "Pab1", "AMIE", "E4B", "TEM", "UBE2I"]


def load_results():
    payload = json.loads(SOURCE.read_text())
    by_task = {row["task"]: row for row in payload["results"]}
    ordered = [by_task[t] for t in TASK_ORDER if t in by_task]
    return ordered


def robust_limits(x, y):
    vals = np.concatenate([x, y])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return -1, 1
    lo, hi = np.quantile(vals, [0.005, 0.995])
    lo = min(lo, float(np.min(vals))) if len(vals) < 30 else lo
    hi = max(hi, float(np.max(vals))) if len(vals) < 30 else hi
    if lo == hi:
        lo -= 1
        hi += 1
    pad = 0.055 * (hi - lo)
    return float(lo - pad), float(hi + pad)


def corr(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def plot_scatter(results):
    set_prospero_style()
    plt.rcParams.update({
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
    })
    n_tasks = len(results)
    n_cols = 3
    n_rows = int(math.ceil(n_tasks / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16.5, 4.25 * n_rows), sharex=False, sharey=False)
    axes_flat = np.atleast_1d(axes).reshape(-1)

    for idx, task_result in enumerate(results):
        ax = axes_flat[idx]
        task = task_result["task"]
        rows = task_result["double_mutants"]
        x = np.asarray([row["additive_delta_fitness"] for row in rows], dtype=float)
        y = np.asarray([row["delta_fitness_double"] for row in rows], dtype=float)
        lim_lo, lim_hi = robust_limits(x, y)
        ax.scatter(x, y, s=13, alpha=0.9, color="#2F5D8A", edgecolors="none", rasterized=True)
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], linestyle="--", linewidth=1.35, color=COLORS["ink"], alpha=0.8)
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_title(task, loc="left", pad=6)
        ax.set_xlabel(r"Additive fitness change, $\Delta_i + \Delta_j$")
        ax.set_ylabel(r"Observed double-mutant change, $\Delta_{ij}$")
        ax.grid(True, axis="both")
        ax.text(
            0.98,
            0.04,
            f"r={corr(x, y):.2f}\nn={len(x):,}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10.5,
            color=COLORS["muted"],
        )

    for idx in range(n_tasks, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle("Oracle additivity of sampled double mutants", fontsize=22, fontweight=600, y=1.012)
    fig.text(
        0.5,
        0.985,
        r"Each point compares the observed oracle fitness change of a double mutant to the sum of its two single-mutant effects.",
        ha="center",
        va="top",
        fontsize=13,
        color=COLORS["muted"],
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "epistasis_additivity_oracle_scatter.png"
    pdf = OUT / "epistasis_additivity_oracle_scatter.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def plot_histograms(results):
    set_prospero_style()
    plt.rcParams.update({
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })
    n_tasks = len(results)
    n_cols = 3
    n_rows = int(math.ceil(n_tasks / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16.5, 4.0 * n_rows), sharex=False, sharey=False)
    axes_flat = np.atleast_1d(axes).reshape(-1)

    for idx, task_result in enumerate(results):
        ax = axes_flat[idx]
        task = task_result["task"]
        rows = task_result["double_mutants"]
        e = np.asarray([row["epistasis_e_ij"] for row in rows], dtype=float)
        q_lo, q_hi = np.quantile(e, [0.01, 0.99])
        if q_lo == q_hi:
            q_lo, q_hi = float(np.min(e)), float(np.max(e))
        pad = 0.06 * max(1e-12, q_hi - q_lo)
        x_lo, x_hi = float(q_lo - pad), float(q_hi + pad)
        clipped = int(np.sum((e < x_lo) | (e > x_hi)))
        hist_vals = e[(e >= x_lo) & (e <= x_hi)]
        ax.hist(hist_vals, bins=35, color="#2F5D8A", alpha=0.92)
        ax.axvline(0, linestyle="--", linewidth=1.25, color=COLORS["ink"], alpha=0.8)
        ax.set_xlim(x_lo, x_hi)
        ax.set_title(task, loc="left", pad=6)
        ax.set_xlabel(r"Epistasis, $e_{ij} = \Delta_{ij} - (\Delta_i + \Delta_j)$")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y")
        note = f"mean={np.mean(e):.2g}\nsd={np.std(e):.2g}"
        if clipped:
            note += f"\n{clipped} clipped"
        ax.text(
            0.98,
            0.84,
            note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10.5,
            color=COLORS["muted"],
        )

    for idx in range(n_tasks, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle("Oracle epistasis distributions", fontsize=22, fontweight=600, y=1.012)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "epistasis_distributions_oracle_histograms.png"
    pdf = OUT / "epistasis_distributions_oracle_histograms.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot epistasis/additivity oracle results.")
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--output-dir", default=str(OUT))
    return parser.parse_args()


def main():
    global SOURCE, OUT
    args = parse_args()
    SOURCE = Path(args.source)
    OUT = Path(args.output_dir)
    results = load_results()
    written = []
    written.extend(plot_scatter(results))
    written.extend(plot_histograms(results))
    summary = OUT / "plot_summary.txt"
    summary.write_text("Source: " + str(SOURCE) + "\nWritten:\n" + "\n".join(str(p) for p in written) + "\n")
    print(f"Wrote {len(written)} plot files")
    print(summary)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
