from __future__ import annotations

import argparse
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from prospero.dataset import RegressionDataset
from prospero.experiments_config import WT_SEQUENCES
from prospero.plotting_style import COLORS, set_prospero_style


DEFAULT_TASKS = ("AAV", "LGK", "GFP", "Pab1", "AMIE", "E4B", "TEM", "UBE2I")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot selected candidates' oracle-fitness distributions by round."
    )
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--method-label", default="0shotProt")
    parser.add_argument("--output-name")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--bins", type=int, default=32)
    return parser


def normalize_sequence(sequence: Any) -> str:
    if isinstance(sequence, str):
        return sequence
    return "".join(map(str, sequence))


def initial_sequence_scores(task: str) -> list[tuple[str, float]]:
    dataset = RegressionDataset(task)
    return [
        (normalize_sequence(sequence), float(score))
        for sequences, scores in (
            (dataset.train, dataset.train_scores),
            (dataset.valid, dataset.valid_scores),
        )
        for sequence, score in zip(sequences, scores)
    ]


def load_explicit_run_rounds(
    seed_paths: list[Path], task: str
) -> tuple[dict[int, list[float]], dict[int, float]]:
    """Load queried fitnesses and reconstruct the online incumbent by round."""
    scores_by_round: defaultdict[int, dict[str, float]] = defaultdict(dict)
    starting_scores_by_round: defaultdict[int, list[float]] = defaultdict(list)
    wild_type = WT_SEQUENCES[task]
    offline_scores = dict(initial_sequence_scores(task))
    if wild_type not in offline_scores:
        raise ValueError(f"WT sequence for {task} is absent from the dataset.")

    for seed_path in seed_paths:
        with seed_path.open("rb") as handle:
            results = pickle.load(handle)
        observed = [(wild_type, offline_scores[wild_type])]
        starting_score = offline_scores[wild_type]
        for round_index in sorted(key for key in results if isinstance(key, int)):
            starting_scores_by_round[round_index].append(starting_score)
            entry = results[round_index]
            sequences = entry.get("Iter sequences") or []
            scores = entry.get("Iter scores") or []
            for sequence, score in zip(sequences, scores):
                sequence_score = float(score)
                sequence = str(sequence)
                scores_by_round[round_index][sequence] = sequence_score
                observed.append((sequence, sequence_score))
            starting_score = max(score for _, score in observed)

    round_scores = {
        round_index: list(sequence_scores.values())
        for round_index, sequence_scores in scores_by_round.items()
    }
    start_scores = {
        round_index: float(np.mean(scores))
        for round_index, scores in starting_scores_by_round.items()
    }
    return round_scores, start_scores


def plot_rounds(
    task: str,
    method_label: str,
    round_scores: dict[int, list[float]],
    start_scores: dict[int, float],
    output_directory: Path,
    bins: int,
    output_name: str | None = None,
) -> Path | None:
    all_scores = [score for scores in round_scores.values() for score in scores]
    if not all_scores:
        return None
    all_values = all_scores + list(start_scores.values())
    lower, upper = float(np.min(all_values)), float(np.max(all_values))
    if lower == upper:
        lower, upper = lower - 0.5, upper + 0.5
    padding = 0.04 * (upper - lower)
    bin_edges = np.linspace(lower - padding, upper + padding, bins + 1)

    set_prospero_style()
    plt.rcParams.update(
        {
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 9,
        }
    )
    rounds = sorted(round_scores)
    figure, axes = plt.subplots(
        len(rounds), 1, figsize=(9.0, 1.55 * len(rounds)), sharex=True
    )
    axes = np.atleast_1d(axes)
    for round_index, axis in zip(rounds, axes):
        scores = round_scores[round_index]
        axis.hist(
            scores,
            bins=bin_edges,
            color=COLORS["prosst"],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.45,
        )
        axis.axvline(
            start_scores[round_index],
            color="#1F7A8C",
            linewidth=1.25,
            zorder=5,
        )
        axis.set_ylabel(
            f"R{round_index}", rotation=0, ha="right", va="center", labelpad=22
        )
        axis.grid(axis="y", alpha=0.25)
        axis.text(
            0.985,
            0.74,
            f"n={len(scores)}",
            ha="right",
            transform=axis.transAxes,
            fontsize=9,
            color=COLORS["muted"],
        )
    axes[-1].set_xlabel("Oracle fitness")
    figure.text(0.017, 0.5, "Optimization round", rotation=90, va="center", fontsize=12)
    figure.subplots_adjust(left=0.12, right=0.985, top=0.985, bottom=0.06, hspace=0.20)

    output_directory.mkdir(parents=True, exist_ok=True)
    if output_name is None:
        safe_label = re.sub(r"[^A-Za-z0-9]+", "_", method_label).strip("_")
        output_name = f"{task}_{safe_label}_selected_fitness_histograms.png"
    output_path = output_directory / output_name
    figure.savefig(output_path, dpi=300)
    figure.savefig(output_path.with_suffix(".pdf"))
    figure.savefig(output_path.with_suffix(".svg"))
    plt.close(figure)
    return output_path


def seed_paths(run_directory: Path, task: str, seeds: list[int] | None) -> list[Path]:
    task_directory = run_directory / task
    root = task_directory if task_directory.is_dir() else run_directory
    paths = sorted(root.glob("seed_*.pkl"))
    if seeds is None:
        return paths
    selected = {f"seed_{seed}.pkl" for seed in seeds}
    return [path for path in paths if path.name in selected]


def main() -> None:
    args = get_parser().parse_args()
    written = []
    for task in args.tasks:
        paths = seed_paths(args.run_directory, task, args.seeds)
        if not paths:
            print(f"No seed results found for {task} in {args.run_directory}")
            continue
        round_scores, start_scores = load_explicit_run_rounds(paths, task)
        output_path = plot_rounds(
            task,
            args.method_label,
            round_scores,
            start_scores,
            args.output_directory,
            args.bins,
            args.output_name,
        )
        if output_path is not None:
            written.append(output_path)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "histogram_summary.txt").write_text(
        "Written plots:\n" + "\n".join(map(str, written)) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(written)} plots")


if __name__ == "__main__":
    main()
