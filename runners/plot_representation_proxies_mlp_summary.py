from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it with `uv add matplotlib`."
    ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot cross-task mean Spearman by budget for MLP configs and per-task "
            "curves for the best low-n config."
        )
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        required=True,
        help="Path to benchmark_results.json from an MLP representation-proxy run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots and summary tables. Defaults to <results_dir>/mlp_summary_plots.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="spearman",
        help="Metric to plot and aggregate (default: spearman).",
    )
    parser.add_argument(
        "--low-n-max-budget",
        type=int,
        default=64,
        help="Largest budget included in low-n selection (default: 64).",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def _collect_config_budget_values(summary: dict, metric: str) -> dict[str, dict[int, list[float]]]:
    config_budget_values: dict[str, dict[int, list[float]]] = {}
    for task_payload in summary["tasks"].values():
        for config_name, config_payload in task_payload["configs"].items():
            budget_map = config_budget_values.setdefault(config_name, {})
            for entry in config_payload["budgets"]:
                budget = int(entry["budget"])
                value = float(entry["metrics"][metric]["mean"])
                budget_map.setdefault(budget, []).append(value)
    return config_budget_values


def _collect_per_task_curve(summary: dict, config_name: str, metric: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    per_task: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for task_name, task_payload in sorted(summary["tasks"].items()):
        if config_name not in task_payload["configs"]:
            continue
        entries = task_payload["configs"][config_name]["budgets"]
        budgets = np.array([int(entry["budget"]) for entry in entries], dtype=float)
        values = np.array([float(entry["metrics"][metric]["mean"]) for entry in entries], dtype=float)
        per_task[task_name] = (budgets, values)
    return per_task


def _write_means_table(config_budget_values: dict[str, dict[int, list[float]]], out_csv: Path) -> None:
    rows: list[dict[str, object]] = []
    for config_name in sorted(config_budget_values):
        for budget in sorted(config_budget_values[config_name]):
            vals = np.array(config_budget_values[config_name][budget], dtype=float)
            rows.append(
                {
                    "config": config_name,
                    "budget": budget,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n_tasks": int(len(vals)),
                }
            )
    with open(out_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config", "budget", "mean", "std", "n_tasks"])
        writer.writeheader()
        writer.writerows(rows)


def _select_best_low_n_config(
    config_budget_values: dict[str, dict[int, list[float]]],
    low_n_max_budget: int,
) -> tuple[str, float, list[int]]:
    best_config = None
    best_score = None
    best_budgets: list[int] = []

    for config_name, budget_map in sorted(config_budget_values.items()):
        low_budgets = sorted([budget for budget in budget_map if budget <= low_n_max_budget])
        if not low_budgets:
            continue
        per_budget_means = [float(np.mean(budget_map[budget])) for budget in low_budgets]
        low_n_score = float(np.mean(per_budget_means))
        if best_score is None or low_n_score > best_score:
            best_config = config_name
            best_score = low_n_score
            best_budgets = low_budgets

    if best_config is None or best_score is None:
        raise RuntimeError(
            f"No config had any budget <= low_n_max_budget ({low_n_max_budget})."
        )
    return best_config, best_score, best_budgets


def _plot_cross_task_means(
    config_budget_values: dict[str, dict[int, list[float]]],
    metric: str,
    out_png: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for idx, config_name in enumerate(sorted(config_budget_values)):
        budgets = sorted(config_budget_values[config_name])
        means = np.array(
            [np.mean(config_budget_values[config_name][budget]) for budget in budgets],
            dtype=float,
        )
        stds = np.array(
            [np.std(config_budget_values[config_name][budget]) for budget in budgets],
            dtype=float,
        )
        color = cmap(idx % 10)
        ax.plot(budgets, means, label=config_name, linewidth=2, marker="o", color=color)
        ax.fill_between(budgets, means - stds, means + stds, alpha=0.15, color=color)

    ax.set_title(f"Cross-task mean {metric} by train budget")
    ax.set_xlabel("Train budget")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _plot_best_config_per_task(
    per_task_curve: dict[str, tuple[np.ndarray, np.ndarray]],
    best_config: str,
    metric: str,
    out_png: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    cmap = plt.get_cmap("tab20")
    for idx, task_name in enumerate(sorted(per_task_curve)):
        budgets, values = per_task_curve[task_name]
        ax.plot(
            budgets,
            values,
            label=task_name,
            linewidth=2,
            marker="o",
            color=cmap(idx % 20),
        )

    ax.set_title(f"Per-task {metric} vs budget ({best_config})")
    ax.set_xlabel("Train budget")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncols=2)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    with open(args.results_json, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    output_dir = args.output_dir or (args.results_json.parent / "mlp_summary_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_budget_values = _collect_config_budget_values(summary, args.metric)
    means_csv = output_dir / "mean_spearman_by_budget.csv"
    _write_means_table(config_budget_values, means_csv)

    best_config, low_n_score, low_n_budgets = _select_best_low_n_config(
        config_budget_values=config_budget_values,
        low_n_max_budget=args.low_n_max_budget,
    )
    per_task_curve = _collect_per_task_curve(summary, best_config, args.metric)

    cross_task_png = output_dir / "cross_task_mean_spearman_by_budget.png"
    per_task_png = output_dir / "best_low_n_variant_per_task_curves.png"
    _plot_cross_task_means(config_budget_values, args.metric, cross_task_png, dpi=args.dpi)
    _plot_best_config_per_task(per_task_curve, best_config, args.metric, per_task_png, dpi=args.dpi)

    payload = {
        "metric": args.metric,
        "low_n_max_budget": args.low_n_max_budget,
        "best_low_n_config": best_config,
        "best_low_n_score": low_n_score,
        "low_n_budgets_used": low_n_budgets,
        "outputs": {
            "means_csv": str(means_csv),
            "cross_task_plot": str(cross_task_png),
            "per_task_plot": str(per_task_png),
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
