from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
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
            "Analyze flatten_mlp architecture selections and plot mean/per-task "
            "Spearman curves from benchmark_records.csv."
        )
    )
    parser.add_argument("--records-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric", type=str, default="spearman")
    parser.add_argument("--config-name", type=str, default="flatten_mlp")
    parser.add_argument(
        "--config-prefix",
        type=str,
        default=None,
        help="If set, include rows whose config starts with this prefix (overrides --config-name exact match).",
    )
    parser.add_argument("--low-n-max-budget", type=int, default=64)
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of best architectures to include in per-task curves.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def _canon_hparam(raw: str) -> str:
    value = ast.literal_eval(raw)
    if isinstance(value, tuple):
        return str(tuple(int(x) for x in value))
    if isinstance(value, list):
        return str(tuple(int(x) for x in value))
    return str(value)


def _load_rows(
    path: Path,
    config_name: str,
    config_prefix: str | None,
    metric: str,
) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if config_prefix is not None:
                if not row["config"].startswith(config_prefix):
                    continue
            elif row["config"] != config_name:
                continue
            rows.append(
                {
                    "task": row["task"],
                    "budget": int(row["budget"]),
                    "seed": int(row["seed"]),
                    "hparam": _canon_hparam(row["selected_hyperparameter"]),
                    "metric": float(row[metric]),
                }
            )
    if not rows:
        raise RuntimeError(f"No rows found for config={config_name!r} in {path}.")
    return rows


def _aggregate_means(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_hparam_budget: dict[tuple[str, int], list[float]] = defaultdict(list)
    by_hparam: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_hparam_budget[(row["hparam"], row["budget"])].append(row["metric"])
        by_hparam[row["hparam"]].append(row["metric"])

    budget_rows: list[dict] = []
    for (hparam, budget), values in sorted(by_hparam_budget.items(), key=lambda x: (x[0][0], x[0][1])):
        arr = np.array(values, dtype=float)
        budget_rows.append(
            {
                "hparam": hparam,
                "budget": budget,
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "n_runs": int(len(arr)),
            }
        )

    overall_rows: list[dict] = []
    for hparam, values in sorted(by_hparam.items()):
        arr = np.array(values, dtype=float)
        overall_rows.append(
            {
                "hparam": hparam,
                "overall_mean": float(np.mean(arr)),
                "overall_std": float(np.std(arr)),
                "n_runs": int(len(arr)),
            }
        )
    overall_rows.sort(key=lambda x: x["overall_mean"], reverse=True)

    return budget_rows, overall_rows


def _low_n_ranking(rows: list[dict], low_n_max_budget: int) -> list[dict]:
    by_hparam: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["budget"] <= low_n_max_budget:
            by_hparam[row["hparam"]].append(row["metric"])

    ranked: list[dict] = []
    for hparam, values in sorted(by_hparam.items()):
        arr = np.array(values, dtype=float)
        ranked.append(
            {
                "hparam": hparam,
                "low_n_mean": float(np.mean(arr)),
                "low_n_std": float(np.std(arr)),
                "n_runs": int(len(arr)),
            }
        )
    ranked.sort(key=lambda x: x["low_n_mean"], reverse=True)
    return ranked


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_mean_by_budget(budget_rows: list[dict], metric: str, out_png: Path, dpi: int) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in budget_rows:
        grouped[row["hparam"]].append(row)

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    cmap = plt.get_cmap("tab20")
    for idx, hparam in enumerate(sorted(grouped)):
        rows = sorted(grouped[hparam], key=lambda x: x["budget"])
        budgets = np.array([row["budget"] for row in rows], dtype=float)
        means = np.array([row["mean"] for row in rows], dtype=float)
        stds = np.array([row["std"] for row in rows], dtype=float)
        color = cmap(idx % 20)
        ax.plot(budgets, means, marker="o", linewidth=2, label=hparam, color=color)
        ax.fill_between(budgets, means - stds, means + stds, color=color, alpha=0.14)

    ax.set_title(f"flatten_mlp: mean {metric} by budget and selected architecture")
    ax.set_xlabel("Train budget")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncols=2)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _plot_per_task_curves_for_topk(
    rows: list[dict],
    top_hparams: list[str],
    metric: str,
    out_dir: Path,
    dpi: int,
) -> list[str]:
    by_hparam_task_budget: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        by_hparam_task_budget[(row["hparam"], row["task"], row["budget"])].append(row["metric"])

    task_names = sorted({row["task"] for row in rows})
    written: list[str] = []
    for hparam in top_hparams:
        fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
        cmap = plt.get_cmap("tab20")
        for idx, task in enumerate(task_names):
            points = []
            for budget in sorted({row["budget"] for row in rows}):
                vals = by_hparam_task_budget.get((hparam, task, budget))
                if vals:
                    points.append((budget, float(np.mean(vals))))
            if not points:
                continue
            x = np.array([p[0] for p in points], dtype=float)
            y = np.array([p[1] for p in points], dtype=float)
            ax.plot(x, y, marker="o", linewidth=2, label=task, color=cmap(idx % 20))

        ax.set_title(f"flatten_mlp selected architecture {hparam}: per-task {metric} vs budget")
        ax.set_xlabel("Train budget")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", ncols=2)

        filename = f"per_task_curves_hparam_{hparam.replace(' ', '').replace(',', '-').replace('(', '').replace(')', '')}.png"
        out_png = out_dir / filename
        fig.savefig(out_png, dpi=dpi)
        plt.close(fig)
        written.append(str(out_png))
    return written


def _plot_task_subplots_all_hparams(
    rows: list[dict],
    metric: str,
    out_png: Path,
    dpi: int,
) -> None:
    task_names = sorted({row["task"] for row in rows})
    hparams = sorted({row["hparam"] for row in rows})
    budgets_all = sorted({row["budget"] for row in rows})
    min_budget = min(budgets_all)
    max_budget = max(budgets_all)

    by_task_hparam_budget: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        by_task_hparam_budget[(row["task"], row["hparam"], row["budget"])].append(row["metric"])

    n_tasks = len(task_names)
    ncols = 3
    nrows = int(np.ceil(n_tasks / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 3.2 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    cmap = plt.get_cmap("tab20")
    colors = {h: cmap(i % 20) for i, h in enumerate(hparams)}

    for idx, task in enumerate(task_names):
        ax = axes[idx // ncols][idx % ncols]
        for hparam in hparams:
            x_vals: list[float] = []
            y_vals: list[float] = []
            y_stds: list[float] = []
            for budget in budgets_all:
                vals = by_task_hparam_budget.get((task, hparam, budget))
                if not vals:
                    continue
                arr = np.array(vals, dtype=float)
                x_vals.append(float(budget))
                y_vals.append(float(np.mean(arr)))
                y_stds.append(float(np.std(arr)))
            if not x_vals:
                continue
            x = np.array(x_vals, dtype=float)
            y = np.array(y_vals, dtype=float)
            y_std = np.array(y_stds, dtype=float)
            ax.plot(x, y, linewidth=1.8, marker="o", markersize=3, color=colors[hparam], label=hparam)
            ax.fill_between(x, y - y_std, y + y_std, color=colors[hparam], alpha=0.10)

        ax.set_title(task)
        ax.set_xlabel("Train budget")
        ax.set_ylabel(metric)
        ax.set_xlim(min_budget, max_budget)
        ax.set_xticks(budgets_all)
        ax.grid(alpha=0.25)

    for idx in range(n_tasks, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncols=min(4, len(labels)),
        )
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(
        args.records_csv,
        config_name=args.config_name,
        config_prefix=args.config_prefix,
        metric=args.metric,
    )
    budget_rows, overall_rows = _aggregate_means(rows)
    low_n_rows = _low_n_ranking(rows, low_n_max_budget=args.low_n_max_budget)

    _write_csv(
        args.output_dir / "flatten_mlp_hparam_mean_by_budget.csv",
        ["hparam", "budget", "mean", "std", "n_runs"],
        budget_rows,
    )
    _write_csv(
        args.output_dir / "flatten_mlp_hparam_overall_ranking.csv",
        ["hparam", "overall_mean", "overall_std", "n_runs"],
        overall_rows,
    )
    _write_csv(
        args.output_dir / "flatten_mlp_hparam_low_n_ranking.csv",
        ["hparam", "low_n_mean", "low_n_std", "n_runs"],
        low_n_rows,
    )

    mean_plot = args.output_dir / "flatten_mlp_hparam_mean_by_budget.png"
    _plot_mean_by_budget(budget_rows, args.metric, mean_plot, dpi=args.dpi)

    task_subplots_plot = args.output_dir / "flatten_mlp_task_subplots_all_hparams.png"
    _plot_task_subplots_all_hparams(
        rows,
        metric=args.metric,
        out_png=task_subplots_plot,
        dpi=args.dpi,
    )

    top_hparams = [row["hparam"] for row in low_n_rows[: max(1, args.top_k)]]
    per_task_plots = _plot_per_task_curves_for_topk(
        rows,
        top_hparams=top_hparams,
        metric=args.metric,
        out_dir=args.output_dir,
        dpi=args.dpi,
    )

    payload = {
        "note": (
            "Analysis uses benchmark_records rows where a given architecture was "
            "selected by inner tuning for flatten_mlp."
        ),
        "records_csv": str(args.records_csv),
        "config_name": args.config_name,
        "metric": args.metric,
        "low_n_max_budget": args.low_n_max_budget,
        "top_hparams_by_low_n_mean": top_hparams,
        "outputs": {
            "mean_by_budget_csv": str(args.output_dir / "flatten_mlp_hparam_mean_by_budget.csv"),
            "overall_ranking_csv": str(args.output_dir / "flatten_mlp_hparam_overall_ranking.csv"),
            "low_n_ranking_csv": str(args.output_dir / "flatten_mlp_hparam_low_n_ranking.csv"),
            "mean_by_budget_plot": str(mean_plot),
            "task_subplots_all_hparams_plot": str(task_subplots_plot),
            "per_task_plots": per_task_plots,
        },
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
