from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from prospero.plotting_style import COLORS, set_prospero_style


DEFAULT_TASKS = ("AAV", "LGK")


@dataclass(frozen=True)
class Method:
    label: str
    color: str
    marker: str
    root: Path


def load_seed(path: Path) -> dict[int, float]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    return {
        int(round_index): float(entry["Best score"])
        for round_index, entry in data.items()
        if isinstance(round_index, int)
        and isinstance(entry, dict)
        and "Best score" in entry
    }


def aggregate(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    rows = [load_seed(path) for path in sorted(root.glob("seed_*.pkl"))]
    rounds = np.arange(1, 11)
    means, standard_errors, counts = [], [], []
    for round_index in rounds:
        values = np.asarray(
            [row[int(round_index)] for row in rows if int(round_index) in row],
            dtype=float,
        )
        counts.append(len(values))
        means.append(float(values.mean()) if len(values) else np.nan)
        standard_errors.append(
            float(values.std(ddof=1) / math.sqrt(len(values)))
            if len(values) > 1
            else 0.0
            if len(values)
            else np.nan
        )
    return rounds, np.asarray(means), np.asarray(standard_errors), counts


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot restricted and unrestricted decoding vocabularies."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--restricted-root", type=Path, required=True)
    parser.add_argument("--unrestricted-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--budget", type=int, default=128)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    set_prospero_style()
    plt.rcParams.update(
        {
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 14,
        }
    )
    tasks = list(args.tasks)
    columns = min(4, len(tasks))
    rows = math.ceil(len(tasks) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(5.8 * columns, 4.7 * rows), sharex=True
    )
    axes = np.atleast_1d(axes).ravel()
    legend_labels: set[str] = set()
    summary = []

    for axis, task in zip(axes, tasks):
        methods = (
            Method(
                "Restricted amino acid vocabulary",
                COLORS["prosst"],
                "s",
                args.restricted_root / task,
            ),
            Method(
                "Unrestricted amino acid vocabulary",
                "#D55E00",
                "^",
                args.unrestricted_root / task,
            ),
        )
        for method in methods:
            rounds_, means, errors, counts = aggregate(method.root)
            valid = np.isfinite(means)
            label = method.label if method.label not in legend_labels else "_nolegend_"
            legend_labels.add(method.label)
            axis.plot(
                rounds_[valid],
                means[valid],
                color=method.color,
                marker=method.marker,
                linewidth=2.8,
                markersize=7,
                label=label,
            )
            axis.fill_between(
                rounds_[valid],
                means[valid] - errors[valid],
                means[valid] + errors[valid],
                color=method.color,
                alpha=0.12,
                linewidth=0,
            )
            summary.append(
                f"{task} {method.label}: counts={counts} "
                f"round10={means[-1]:.6g} sem={errors[-1]:.3g}"
            )
        axis.set_title(task, loc="left", pad=6)
        axis.set(xlim=(1, 10), xlabel="Optimization round", ylabel="Mean max fitness")
        axis.set_xticks(range(1, 11))
        axis.grid(True, axis="y")
    for axis in axes[len(tasks) :]:
        axis.axis("off")

    figure.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"restricted_vs_unrestricted_vocab_k{args.budget}_mean_max"
    paths = [args.output_dir / f"{stem}.{suffix}" for suffix in ("png", "pdf", "svg")]
    for path in paths:
        figure.savefig(
            path, dpi=320 if path.suffix == ".png" else None, bbox_inches="tight"
        )
    plt.close(figure)
    (args.output_dir / "plot_summary.txt").write_text(
        "Written:\n"
        + "\n".join(map(str, paths))
        + "\n\nSummary:\n"
        + "\n".join(summary)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
