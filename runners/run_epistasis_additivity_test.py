from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from prospero.experiments_config import WT_SEQUENCES
from prospero.landscapes import get_landscape
from prospero.utils import set_seed

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class SingleMutation:
    position: int
    wt_residue: str
    mutant_residue: str
    mutant_sequence: str
    oracle_fitness: float
    delta_fitness: float
    effect_bin: str


@dataclass(frozen=True)
class DoubleMutationSample:
    pair_type: str
    first: SingleMutation
    second: SingleMutation
    double_mutant_sequence: str
    oracle_fitness_double: float
    delta_fitness_double: float
    additive_delta_fitness: float
    epistasis_e_ij: float


def _fitness(oracle, task: str, sequences: list[str]) -> np.ndarray:
    if task.startswith("D_SHIFT"):
        values = oracle.get_fitness(sequences)
    else:
        values = oracle.get_fitness(np.asarray(sequences))
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _batched_oracle_scores(
    oracle,
    task: str,
    sequences: list[str],
    batch_size: int,
) -> np.ndarray:
    if not sequences:
        return np.empty((0,), dtype=np.float32)
    outputs: list[np.ndarray] = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        outputs.append(_fitness(oracle, task, batch))
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def _enumerate_single_mutants(wt_sequence: str) -> tuple[list[str], list[tuple[int, str, str]]]:
    sequences: list[str] = []
    metadata: list[tuple[int, str, str]] = []
    for idx, wt_residue in enumerate(wt_sequence):
        for aa in AA_ALPHABET:
            if aa == wt_residue:
                continue
            seq_chars = list(wt_sequence)
            seq_chars[idx] = aa
            sequences.append("".join(seq_chars))
            metadata.append((idx, wt_residue, aa))
    return sequences, metadata


def _assign_bins(delta_fitness: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    q1 = float(np.quantile(delta_fitness, 1.0 / 3.0))
    q2 = float(np.quantile(delta_fitness, 2.0 / 3.0))
    # bottom third -> bad, middle third -> neutral, top third -> good
    bins = np.empty(delta_fitness.shape[0], dtype=object)
    bins[delta_fitness <= q1] = "bad"
    bins[(delta_fitness > q1) & (delta_fitness <= q2)] = "neutral"
    bins[delta_fitness > q2] = "good"
    return bins, {"q33": q1, "q67": q2}


def _build_single_mutation_records(
    wt_sequence: str,
    sequences: list[str],
    metadata: list[tuple[int, str, str]],
    oracle_scores: np.ndarray,
    wt_oracle_fitness: float,
    bins: np.ndarray,
) -> list[SingleMutation]:
    out: list[SingleMutation] = []
    for i, seq in enumerate(sequences):
        pos, wt_residue, mutant_residue = metadata[i]
        score = float(oracle_scores[i])
        out.append(
            SingleMutation(
                position=int(pos),
                wt_residue=wt_residue,
                mutant_residue=mutant_residue,
                mutant_sequence=seq,
                oracle_fitness=score,
                delta_fitness=score - wt_oracle_fitness,
                effect_bin=str(bins[i]),
            )
        )
    return out


def _double_sequence(wt_sequence: str, first: SingleMutation, second: SingleMutation) -> str:
    seq = list(wt_sequence)
    seq[first.position] = first.mutant_residue
    seq[second.position] = second.mutant_residue
    return "".join(seq)


def _sample_pair_indices(
    rng: random.Random,
    left: list[int],
    right: list[int],
    n: int,
    require_distinct_positions: bool,
    singles: list[SingleMutation],
    same_group: bool,
) -> tuple[list[tuple[int, int]], bool]:
    sampled: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    max_attempts = max(10_000, n * 500)
    attempts = 0
    replacement_used = False

    while len(sampled) < n and attempts < max_attempts:
        attempts += 1
        i = rng.choice(left)
        j = rng.choice(right)
        if same_group and i == j:
            continue
        if require_distinct_positions and singles[i].position == singles[j].position:
            continue
        key = (i, j) if not same_group else tuple(sorted((i, j)))
        if key in seen:
            continue
        seen.add(key)
        sampled.append((i, j))

    # Fallback: sample with replacement if unique pool is too small.
    while len(sampled) < n:
        i = rng.choice(left)
        j = rng.choice(right)
        if same_group and i == j:
            continue
        if require_distinct_positions and singles[i].position == singles[j].position:
            continue
        sampled.append((i, j))
        replacement_used = True

    return sampled, replacement_used


def _sample_double_mutations(
    rng: random.Random,
    wt_sequence: str,
    singles: list[SingleMutation],
    n_per_group: int,
) -> tuple[list[DoubleMutationSample], dict[str, object]]:
    index_by_bin = {"bad": [], "neutral": [], "good": []}
    for idx, mutation in enumerate(singles):
        index_by_bin[mutation.effect_bin].append(idx)

    pair_plan = [
        ("bad_x_bad", "bad", "bad", True),
        ("bad_x_good", "bad", "good", False),
        ("neutral_x_neutral", "neutral", "neutral", True),
        ("good_x_good", "good", "good", True),
    ]

    samples: list[DoubleMutationSample] = []
    sampling_meta: dict[str, object] = {"replacement_used": {}}
    doubles_to_score: list[str] = []
    pair_index_metadata: list[tuple[str, int, int]] = []

    for pair_name, left_bin, right_bin, same_group in pair_plan:
        pairs, replacement_used = _sample_pair_indices(
            rng=rng,
            left=index_by_bin[left_bin],
            right=index_by_bin[right_bin],
            n=n_per_group,
            require_distinct_positions=True,
            singles=singles,
            same_group=same_group,
        )
        sampling_meta["replacement_used"][pair_name] = bool(replacement_used)
        for i, j in pairs:
            first = singles[i]
            second = singles[j]
            doubles_to_score.append(_double_sequence(wt_sequence, first, second))
            pair_index_metadata.append((pair_name, i, j))

    sampling_meta["n_pairs_total"] = len(doubles_to_score)
    return samples, {
        "index_by_bin": index_by_bin,
        "pair_index_metadata": pair_index_metadata,
        "doubles_to_score": doubles_to_score,
        "sampling_meta": sampling_meta,
    }


def _materialize_double_samples(
    singles: list[SingleMutation],
    pair_index_metadata: list[tuple[str, int, int]],
    double_sequences: list[str],
    double_scores: np.ndarray,
    wt_oracle_fitness: float,
) -> list[DoubleMutationSample]:
    out: list[DoubleMutationSample] = []
    for row_idx, (pair_name, i, j) in enumerate(pair_index_metadata):
        first = singles[i]
        second = singles[j]
        observed = float(double_scores[row_idx] - wt_oracle_fitness)
        additive = float(first.delta_fitness + second.delta_fitness)
        out.append(
            DoubleMutationSample(
                pair_type=pair_name,
                first=first,
                second=second,
                double_mutant_sequence=double_sequences[row_idx],
                oracle_fitness_double=float(double_scores[row_idx]),
                delta_fitness_double=observed,
                additive_delta_fitness=additive,
                epistasis_e_ij=float(observed - additive),
            )
        )
    return out


def _run_task(args, task: str, rng: random.Random) -> dict[str, object]:
    task_start = time.perf_counter()
    wt_sequence = WT_SEQUENCES[task]
    oracle = get_landscape(task)
    wt_oracle_fitness = float(_fitness(oracle, task, [wt_sequence])[0])

    single_sequences, single_metadata = _enumerate_single_mutants(wt_sequence)
    single_scores = _batched_oracle_scores(
        oracle=oracle,
        task=task,
        sequences=single_sequences,
        batch_size=args.oracle_batch_size,
    )
    single_delta = single_scores - wt_oracle_fitness
    bins, quantiles = _assign_bins(single_delta)
    single_records = _build_single_mutation_records(
        wt_sequence=wt_sequence,
        sequences=single_sequences,
        metadata=single_metadata,
        oracle_scores=single_scores,
        wt_oracle_fitness=wt_oracle_fitness,
        bins=bins,
    )

    _, prep = _sample_double_mutations(
        rng=rng,
        wt_sequence=wt_sequence,
        singles=single_records,
        n_per_group=args.samples_per_pair_type,
    )
    double_sequences = prep["doubles_to_score"]
    pair_index_metadata = prep["pair_index_metadata"]
    double_scores = _batched_oracle_scores(
        oracle=oracle,
        task=task,
        sequences=double_sequences,
        batch_size=args.oracle_batch_size,
    )
    double_records = _materialize_double_samples(
        singles=single_records,
        pair_index_metadata=pair_index_metadata,
        double_sequences=double_sequences,
        double_scores=double_scores,
        wt_oracle_fitness=wt_oracle_fitness,
    )

    # Release model memory before moving to the next task.
    del oracle
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    pair_type_counts: dict[str, int] = {}
    for record in double_records:
        pair_type_counts[record.pair_type] = pair_type_counts.get(record.pair_type, 0) + 1

    epistasis_values = np.asarray([r.epistasis_e_ij for r in double_records], dtype=np.float32)
    additive_values = np.asarray(
        [r.additive_delta_fitness for r in double_records], dtype=np.float32
    )
    observed_values = np.asarray(
        [r.delta_fitness_double for r in double_records], dtype=np.float32
    )

    return {
        "task": task,
        "wt_sequence_length": len(wt_sequence),
        "wt_oracle_fitness": wt_oracle_fitness,
        "num_single_mutants": len(single_records),
        "single_mutant_quantiles": quantiles,
        "single_mutant_bin_counts": {
            "bad": int(sum(r.effect_bin == "bad" for r in single_records)),
            "neutral": int(sum(r.effect_bin == "neutral" for r in single_records)),
            "good": int(sum(r.effect_bin == "good" for r in single_records)),
        },
        "sampling": {
            "samples_per_pair_type": args.samples_per_pair_type,
            "pair_type_counts": pair_type_counts,
            "replacement_used": prep["sampling_meta"]["replacement_used"],
        },
        "double_mutant_summary": {
            "epistasis_mean": float(np.mean(epistasis_values)),
            "epistasis_std": float(np.std(epistasis_values)),
            "epistasis_min": float(np.min(epistasis_values)),
            "epistasis_max": float(np.max(epistasis_values)),
            "pearson_observed_vs_additive": float(
                np.corrcoef(observed_values, additive_values)[0, 1]
            ),
        },
        "double_mutants": [asdict(record) for record in double_records],
        "timings_seconds": {
            "task_total_seconds": time.perf_counter() - task_start,
        },
    }


def _plot_scatter_all_tasks(results: list[dict[str, object]], output_path: Path) -> None:
    n_tasks = len(results)
    n_cols = 3
    n_rows = int(math.ceil(n_tasks / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 3.8 * n_rows),
        sharex=False,
        sharey=False,
    )
    axes_flat = np.atleast_1d(axes).reshape(-1)

    for idx, task_result in enumerate(results):
        ax = axes_flat[idx]
        task = task_result["task"]
        rows = task_result["double_mutants"]
        x = np.asarray([row["additive_delta_fitness"] for row in rows], dtype=np.float32)
        y = np.asarray([row["delta_fitness_double"] for row in rows], dtype=np.float32)

        all_values = np.concatenate([x, y])
        lo = float(np.min(all_values))
        hi = float(np.max(all_values))
        pad = 0.02 * max(1e-6, hi - lo)
        line_lo = lo - pad
        line_hi = hi + pad

        ax.scatter(x, y, s=10, alpha=0.35, color="#2f5d8a")
        ax.plot(
            [line_lo, line_hi],
            [line_lo, line_hi],
            linestyle="--",
            linewidth=1.2,
            color="black",
        )
        ax.set_title(task)
        ax.set_xlabel(r"Additive: $\Delta_i + \Delta_j$")
        ax.set_ylabel(r"Observed: $\Delta_{ij}$")
        ax.grid(alpha=0.2)

    for idx in range(n_tasks, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(r"Epistasis Additivity per Task ($\Delta_{ij}$ vs $\Delta_i + \Delta_j$)", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.985])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_histograms_all_tasks(results: list[dict[str, object]], output_path: Path) -> None:
    n_tasks = len(results)
    n_cols = 3
    n_rows = int(math.ceil(n_tasks / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 3.6 * n_rows),
        sharex=False,
        sharey=False,
    )
    axes_flat = np.atleast_1d(axes).reshape(-1)

    for idx, task_result in enumerate(results):
        ax = axes_flat[idx]
        task = task_result["task"]
        rows = task_result["double_mutants"]
        e_vals = np.asarray([row["epistasis_e_ij"] for row in rows], dtype=np.float32)
        ax.hist(e_vals, bins=35, alpha=0.8, color="#2f5d8a")
        ax.axvline(0.0, linestyle="--", linewidth=1.0, color="black")
        ax.set_title(task)
        ax.set_xlabel(r"$e_{ij} = \Delta_{ij} - (\Delta_i + \Delta_j)$")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.2)

    for idx in range(n_tasks, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle("Epistasis Distributions per Task", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.985])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Epistasis additivity benchmark: enumerate all single mutants, bin by "
            "single-mutant effect quantiles, sample structured double mutants, "
            "score with oracle, and produce JSON + plots."
        )
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=list(WT_SEQUENCES.keys()),
        help="Tasks to evaluate. Defaults to all tasks with WT sequences.",
    )
    parser.add_argument(
        "--samples-per-pair-type",
        type=int,
        default=250,
        help="Number of sampled double mutants per pair class.",
    )
    parser.add_argument(
        "--oracle-batch-size",
        type=int,
        default=128,
        help="Batch size for oracle scoring.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducible pair sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/0423_epistasis",
        help="Directory for JSON outputs and plots.",
    )
    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed, True)
    rng = random.Random(args.seed)

    run_start = time.perf_counter()
    run_started_at_utc = datetime.now(timezone.utc).isoformat()

    task_results: list[dict[str, object]] = []
    for task in args.tasks:
        if task not in WT_SEQUENCES:
            raise ValueError(f"Unknown task {task!r}. Available keys: {sorted(WT_SEQUENCES)}")
        print(f"[epistasis] start task={task}")
        task_results.append(_run_task(args, task, rng))
        print(f"[epistasis] done task={task}")

    scatter_path = out_dir / "epistasis_scatter_all_tasks.png"
    hist_path = out_dir / "epistasis_histograms_all_tasks.png"
    _plot_scatter_all_tasks(task_results, scatter_path)
    _plot_histograms_all_tasks(task_results, hist_path)

    run_finished_at_utc = datetime.now(timezone.utc).isoformat()
    payload = {
        "run_started_at_utc": run_started_at_utc,
        "run_finished_at_utc": run_finished_at_utc,
        "seed": args.seed,
        "tasks": args.tasks,
        "config": {
            "samples_per_pair_type": args.samples_per_pair_type,
            "oracle_batch_size": args.oracle_batch_size,
            "pair_types": [
                "bad_x_bad",
                "bad_x_good",
                "neutral_x_neutral",
                "good_x_good",
            ],
            "single_mutant_bins": {
                "bad": "bottom third by single-mutant delta_fitness",
                "neutral": "middle third by single-mutant delta_fitness",
                "good": "top third by single-mutant delta_fitness",
            },
        },
        "plots": {
            "scatter_delta_ij_vs_additive": str(scatter_path),
            "histograms_epistasis_e_ij": str(hist_path),
        },
        "results": task_results,
        "total_runtime_seconds": time.perf_counter() - run_start,
    }

    output_json = out_dir / "epistasis_additivity_all_tasks.json"
    output_json.write_text(json.dumps(payload, indent=2))
    print(f"saved: {output_json}")
    print(f"saved: {scatter_path}")
    print(f"saved: {hist_path}")


if __name__ == "__main__":
    main()
