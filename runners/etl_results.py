from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np


SEED_FILENAME_RE = re.compile(r"seed_(\d+)\.pkl$")


def extract_data(seed_runs: list[dict], n_iter: int) -> dict:
    iter_runs = [run[n_iter] for run in seed_runs if n_iter in run]
    if not iter_runs:
        raise ValueError(f"No seed data found for iteration {n_iter}")

    sequences = [run["Sequences"] for run in iter_runs]
    max_scores = [run["Best score"] for run in iter_runs]
    diversity = [run["Diversity"] for run in iter_runs]
    novelty = [run["WT Novelty"] for run in iter_runs]
    mean_scores = [run["Performance"] for run in iter_runs]
    median_scores = [run["Median performance"] for run in iter_runs]

    return {
        "Sequences": np.concatenate(sequences).tolist(),
        "Mean max score": np.mean(max_scores).round(3),
        "Std max score": np.std(max_scores).round(3),
        "Mean performance": np.mean(mean_scores).round(3),
        "Std performance": np.std(mean_scores).round(3),
        "Mean diversity": np.mean(diversity).round(3),
        "Std diversity": np.std(diversity).round(3),
        "Mean novelty": np.mean(novelty).round(3),
        "Std novelty": np.std(novelty).round(3),
        "Mean median performance": np.mean(median_scores).round(3),
        "Std median performance": np.std(median_scores).round(3),
    }


def load_seed_data(task_path: Path) -> list[dict]:
    seed_paths = []
    for path in task_path.iterdir():
        match = SEED_FILENAME_RE.match(path.name)
        if match is not None:
            seed_paths.append((int(match.group(1)), path))

    seed_runs = []
    for _, path in sorted(seed_paths):
        with path.open("rb") as handle:
            seed_runs.append(pickle.load(handle))
    return seed_runs


def write_transformed_results(results_dirpath: Path, task: str, n_iters: int) -> Path:
    task_path = results_dirpath / task
    seed_runs = load_seed_data(task_path)
    if not seed_runs:
        raise FileNotFoundError(f"No seed_*.pkl files found in {task_path}")

    transformed = {n_iter: extract_data(seed_runs, n_iter) for n_iter in range(1, n_iters + 1)}
    output_path = task_path / "transformed_results.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(transformed, handle)
    return output_path


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dirpath", type=Path, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--n_iters", type=int, default=10)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    output_path = write_transformed_results(args.results_dirpath, args.task, args.n_iters)
    print(output_path)


if __name__ == "__main__":
    main()
