import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from prospero.dataset import RegressionDataset
from prospero.experiments_config import WT_SEQUENCES
from prospero.plotting_style import COLORS, set_prospero_style


TASKS = ["AAV", "LGK", "GFP", "Pab1", "AMIE", "E4B", "TEM", "UBE2I"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot vertically stacked per-round oracle-fitness histograms for "
            "zero-shot EvoDiff runs. Duplicate sequences are collapsed per "
            "task/method/budget/round."
        )
    )
    parser.add_argument("--output_dir", default="outputs/evodiff_zero_shot_alignment_20260602/fitness_histograms")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--strategy", default="seed_grow")
    parser.add_argument("--budgets", nargs="+", default=None, help="Query budgets, e.g. 8 64 128. Defaults to all discovered.")
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--run_dir", default=None, help="Explicit run root containing TASK/seed_*.pkl files.")
    parser.add_argument("--method_label", default="run", help="Label used when --run_dir is provided.")
    parser.add_argument("--output_name", default=None, help="Explicit output filename for --run_dir mode.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Optional seed filter for --run_dir mode.")
    return parser.parse_args()


def task_slug(task):
    return task.lower()


def newest(paths):
    paths = [p for p in paths if p.exists()]
    return sorted(paths)[-1] if paths else None


def discover_roots(task):
    slug = task_slug(task)
    output_root = Path("outputs")
    no_ft = newest(list(output_root.glob(f"{slug}_zero_shot_fixed_mask_k4*")))

    ft_candidates = [
        path
        for path in output_root.glob(f"{slug}_zero_shot_evodiff_ft_k4_variable_k*")
        if "kl0p1" not in path.name and "pretrain" not in path.name
    ]
    ft = newest(ft_candidates)
    return {"no_ft": no_ft, "ft_kl2": ft}


def discover_budget_dirs(root, strategy, task):
    if root is None:
        return {}

    budget_dirs = {}
    task_dir = root / strategy / task
    if task_dir.exists():
        budget_dirs[read_query_budget(task_dir) or "default"] = task_dir

    for path in sorted(root.glob(f"n_samples_*/{strategy}/{task}")):
        match = re.search(r"n_samples_(\d+)", str(path))
        if match:
            budget_dirs[match.group(1)] = path
    return budget_dirs


def read_query_budget(task_dir):
    metadata_paths = sorted(task_dir.glob("seed_*.zero_shot_metadata.json"))
    if not metadata_paths:
        return None
    with metadata_paths[0].open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    n_queries = metadata.get("n_queries")
    return str(n_queries) if n_queries is not None else None


def load_round_scores(seed_paths):
    round_to_seq_score = defaultdict(dict)
    for seed_path in seed_paths:
        with seed_path.open("rb") as handle:
            results = pickle.load(handle)
        for round_idx in range(1, 11):
            if round_idx not in results:
                continue
            entry = results[round_idx]
            seqs = entry.get("Iter sequences") or []
            scores = entry.get("Iter scores") or []
            for seq, score in zip(seqs, scores):
                round_to_seq_score[round_idx][str(seq)] = float(score)
    return {round_idx: list(seq_scores.values()) for round_idx, seq_scores in round_to_seq_score.items()}


def normalize_sequence(seq):
    if isinstance(seq, str):
        return seq
    return "".join(map(str, seq))


def initial_sequence_scores(task):
    dataset = RegressionDataset(task)
    seq_scores = []
    for seq, score in zip(dataset.train, dataset.train_scores):
        seq_scores.append((normalize_sequence(seq), float(score)))
    for seq, score in zip(dataset.valid, dataset.valid_scores):
        seq_scores.append((normalize_sequence(seq), float(score)))
    return seq_scores


def load_explicit_run_rounds(seed_paths, task):
    """Load selected oracle-query scores and reconstruct each round's starting fitness."""
    round_to_seq_score = defaultdict(dict)
    round_to_start_scores = defaultdict(list)
    wt = WT_SEQUENCES[task]
    initial_scores = initial_sequence_scores(task)

    for seed_path in seed_paths:
        with seed_path.open("rb") as handle:
            results = pickle.load(handle)

        known = list(initial_scores)
        score_by_seq = {}
        for seq, score in known:
            if seq not in score_by_seq or score > score_by_seq[seq]:
                score_by_seq[seq] = score

        wt_score = score_by_seq.get(wt)
        if wt_score is None:
            raise ValueError(f"WT sequence for {task} was not found in the initial dataset.")
        starting_score = wt_score

        for round_idx in range(1, 11):
            if round_idx not in results:
                continue
            round_to_start_scores[round_idx].append(float(starting_score))
            entry = results[round_idx]
            seqs = entry.get("Iter sequences") or []
            scores = entry.get("Iter scores") or []
            for seq, score in zip(seqs, scores):
                seq = str(seq)
                score = float(score)
                round_to_seq_score[round_idx][seq] = score
                known.append((seq, score))
                if seq not in score_by_seq or score > score_by_seq[seq]:
                    score_by_seq[seq] = score
            if known:
                _, starting_score = max(known, key=lambda item: item[1])

    round_scores = {round_idx: list(seq_scores.values()) for round_idx, seq_scores in round_to_seq_score.items()}
    start_scores = {
        round_idx: float(np.mean(scores))
        for round_idx, scores in round_to_start_scores.items()
        if scores
    }
    return round_scores, start_scores


def plot_explicit_run(task, method_label, round_scores, start_scores, output_dir, bins, output_name=None):
    set_prospero_style()
    all_scores = [score for scores in round_scores.values() for score in scores]
    all_starts = list(start_scores.values())
    if not all_scores:
        return None

    score_min = float(np.min(all_scores + all_starts))
    score_max = float(np.max(all_scores + all_starts))
    if score_min == score_max:
        score_min -= 0.5
        score_max += 0.5
    pad = 0.04 * (score_max - score_min)
    bin_edges = np.linspace(score_min - pad, score_max + pad, bins + 1)

    plt.rcParams.update(
        {
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(10, 1, figsize=(9.0, 15.5), sharex=True)
    hist_color = COLORS["prosst"]
    start_color = "#1F7A8C"

    for round_idx, ax in enumerate(axes, start=1):
        scores = round_scores.get(round_idx, [])
        ax.hist(scores, bins=bin_edges, color=hist_color, alpha=0.82, edgecolor="white", linewidth=0.45)
        if round_idx in start_scores:
            ax.axvline(start_scores[round_idx], color=start_color, linewidth=1.25, alpha=0.95, zorder=5)
        ax.set_ylabel(f"R{round_idx}", rotation=0, ha="right", va="center", labelpad=22)
        ax.grid(axis="y", alpha=0.25)
        ax.text(
            0.985,
            0.74,
            f"n={len(scores)}",
            ha="right",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color=COLORS["muted"],
        )
    axes[-1].set_xlabel("Oracle fitness")
    fig.suptitle(
        f"{task} selected-candidate fitness distributions",
        x=0.12,
        y=0.985,
        ha="left",
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.12,
        0.965,
        f"{method_label}; blue bar = round starting-sequence fitness",
        ha="left",
        va="top",
        fontsize=10,
        color=COLORS["muted"],
    )
    fig.text(0.017, 0.5, "Optimization round", rotation=90, va="center", ha="center", fontsize=12)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.925, bottom=0.06, hspace=0.20)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_name is None:
        output_name = f"{task}_{method_label}_selected_fitness_histograms.png"
    out_path = output_dir / output_name
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)
    return out_path


def plot_task_budget(task, budget, method_to_scores, output_dir, bins):
    all_scores = [
        score
        for round_scores in method_to_scores.values()
        for scores in round_scores.values()
        for score in scores
    ]
    if not all_scores:
        return None

    score_min = float(np.min(all_scores))
    score_max = float(np.max(all_scores))
    if score_min == score_max:
        score_min -= 0.5
        score_max += 0.5
    bin_edges = np.linspace(score_min, score_max, bins + 1)

    methods = list(method_to_scores)
    fig, axes = plt.subplots(
        10,
        len(methods),
        figsize=(5.2 * len(methods), 18),
        sharex=True,
        squeeze=False,
    )
    colors = {"no_ft": "#395C6B", "ft_kl2": "#D9822B"}

    for col, method in enumerate(methods):
        axes[0][col].set_title(method.replace("_", " "))
        for round_idx in range(1, 11):
            ax = axes[round_idx - 1][col]
            scores = method_to_scores[method].get(round_idx, [])
            ax.hist(scores, bins=bin_edges, color=colors.get(method, "#666666"), alpha=0.85)
            ax.set_ylabel(f"R{round_idx}\ncount")
            ax.grid(axis="y", alpha=0.25)
            ax.text(
                0.98,
                0.78,
                f"n={len(scores)}",
                ha="right",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
    for ax in axes[-1]:
        ax.set_xlabel("oracle fitness")

    fig.suptitle(f"{task} zero-shot generated fitness distributions, seed_grow K=4 KL=2, query budget={budget}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    output_dir.mkdir(parents=True, exist_ok=True)
    method_suffix = "_".join(methods)
    out_path = output_dir / f"{task}_n{budget}_{method_suffix}_seed_grow_k4_kl2_histograms.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_task_budget_method(task, budget, method, round_scores, output_dir, bins):
    return plot_task_budget(
        task,
        budget,
        {method: round_scores},
        output_dir / "by_method",
        bins,
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        written = []
        for task in args.tasks:
            task_dir = run_dir / task
            seed_paths = sorted(task_dir.glob("seed_*.pkl")) if task_dir.exists() else sorted(run_dir.glob("seed_*.pkl"))
            if args.seeds is not None:
                wanted = {f"seed_{seed}.pkl" for seed in args.seeds}
                seed_paths = [path for path in seed_paths if path.name in wanted]
            if not seed_paths:
                print(f"No seed pickles found for {task} in {run_dir}")
                continue
            round_scores, start_scores = load_explicit_run_rounds(seed_paths, task)
            out_path = plot_explicit_run(
                task=task,
                method_label=args.method_label,
                round_scores=round_scores,
                start_scores=start_scores,
                output_dir=output_dir,
                bins=args.bins,
                output_name=args.output_name,
            )
            if out_path is not None:
                written.append(out_path)
        summary_path = output_dir / "histogram_summary.txt"
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            "Written plots:\n" + "\n".join(map(str, written)) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {len(written)} plots")
        print(f"Summary: {summary_path}")
        return

    written = []
    missing = []
    missing_method_budgets = []
    requested_budgets = set(args.budgets) if args.budgets else None

    for task in args.tasks:
        roots = discover_roots(task)
        method_budget_dirs = {
            method: discover_budget_dirs(root, args.strategy, task)
            for method, root in roots.items()
        }
        budgets = sorted(
            set().union(*(dirs.keys() for dirs in method_budget_dirs.values())),
            key=lambda value: int(value) if value.isdigit() else -1,
        )
        if requested_budgets is not None:
            budgets = [budget for budget in budgets if budget in requested_budgets]
        if not budgets:
            missing.append({"task": task, "reason": "no matching seed_grow output directories"})
            continue

        for budget in budgets:
            method_to_scores = {}
            for method, budget_dirs in method_budget_dirs.items():
                run_dir = budget_dirs.get(budget)
                if run_dir is None:
                    if method == "no_ft":
                        missing_method_budgets.append({"task": task, "budget": budget, "method": method, "reason": "missing directory"})
                    continue
                seed_paths = sorted(run_dir.glob("seed_*.pkl"))
                if not seed_paths:
                    missing_method_budgets.append({"task": task, "budget": budget, "method": method, "reason": "missing seed pickles"})
                    continue
                method_to_scores[method] = load_round_scores(seed_paths)
            if not method_to_scores:
                continue
            out_path = plot_task_budget(task, budget, method_to_scores, output_dir, args.bins)
            if out_path is not None:
                written.append(str(out_path))
            for method, round_scores in method_to_scores.items():
                out_path = plot_task_budget_method(task, budget, method, round_scores, output_dir, args.bins)
                if out_path is not None:
                    written.append(str(out_path))

    summary_path = output_dir / "histogram_summary.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Written plots:\n")
        for path in written:
            handle.write(f"{path}\n")
        handle.write("\nMissing:\n")
        for item in missing:
            handle.write(f"{item}\n")
        handle.write("\nMissing method/budget combinations:\n")
        for item in missing_method_budgets:
            handle.write(f"{item}\n")

    print(f"Wrote {len(written)} plots")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
