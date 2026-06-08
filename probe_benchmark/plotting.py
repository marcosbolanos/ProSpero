from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it with `uv add matplotlib`."
    ) from exc


PLOT_METRICS = ("spearman", "r2", "top_decile_recall")


def _task_colors(names: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {name: cmap(i % 10) for i, name in enumerate(names)}


def _sorted_config_names(summary: dict) -> list[str]:
    names = set()
    for task_payload in summary["tasks"].values():
        names.update(task_payload["configs"].keys())
    return sorted(names)


def _collect_global_matrix(summary: dict, metric_name: str) -> tuple[list[str], list[str], np.ndarray]:
    task_names = sorted(summary["tasks"])
    config_names = _sorted_config_names(summary)
    matrix = np.full((len(task_names), len(config_names)), np.nan, dtype=float)

    for task_idx, task_name in enumerate(task_names):
        task_payload = summary["tasks"][task_name]
        for config_idx, config_name in enumerate(config_names):
            config_payload = task_payload["configs"].get(config_name)
            if config_payload is None:
                continue
            budget_entries = config_payload["budgets"]
            best_budget = max(budget_entries, key=lambda entry: entry["budget"])
            matrix[task_idx, config_idx] = float(best_budget["metrics"][metric_name]["mean"])

    return task_names, config_names, matrix


def save_plots(summary: dict, output_dir: str | Path, dpi: int = 200) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    save_task_metric_curves(summary, output_path / "task_metric_curves.png", dpi)
    save_global_metric_curves(summary, output_path / "global_metric_curves.png", dpi)
    for metric_name in PLOT_METRICS:
        save_metric_heatmap(
            summary,
            metric_name,
            output_path / f"heatmap_{metric_name}.png",
            dpi,
        )

    with open(output_path / "plot_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "plots": [
                    "task_metric_curves.png",
                    "global_metric_curves.png",
                    *[f"heatmap_{metric_name}.png" for metric_name in PLOT_METRICS],
                ]
            },
            handle,
            indent=2,
        )


def save_spearman_summary_plots(
    summary: dict,
    output_dir: str | Path,
    dpi: int = 200,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    means_csv = output_path / "mean_spearman_by_budget.csv"
    cross_task_png = output_path / "cross_task_mean_spearman_by_budget.png"
    per_task_png = output_path / "per_task_spearman_by_budget.png"

    config_budget_values = _collect_config_budget_values(summary, metric_name="spearman")
    _write_metric_means_table(
        config_budget_values=config_budget_values,
        out_csv=means_csv,
        metric_name="spearman",
    )
    _plot_cross_task_metric_means(
        config_budget_values=config_budget_values,
        metric_name="spearman",
        out_png=cross_task_png,
        dpi=dpi,
    )
    _plot_per_task_metric_subplots(
        summary=summary,
        metric_name="spearman",
        out_png=per_task_png,
        dpi=dpi,
    )

    with open(output_path / "summary_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "plots": [
                    "cross_task_mean_spearman_by_budget.png",
                    "per_task_spearman_by_budget.png",
                ],
                "tables": ["mean_spearman_by_budget.csv"],
            },
            handle,
            indent=2,
        )


def save_hyperparameter_diagnostic_plots(
    diagnostic_records: list[dict],
    output_dir: str | Path,
    dpi: int = 200,
) -> None:
    scored_records = [
        record
        for record in diagnostic_records
        if record.get("score") is not None and record.get("search_value") is not None
    ]
    if not scored_records:
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    task_names = sorted({str(record["task"]) for record in scored_records})
    plot_files: list[str] = []
    for task_name in task_names:
        filename = f"{task_name}_alpha_sweep.png"
        _plot_task_hyperparameter_sweeps(
            task_name=task_name,
            task_records=[record for record in scored_records if record["task"] == task_name],
            out_png=output_path / filename,
            dpi=dpi,
        )
        plot_files.append(filename)

    with open(output_path / "supplementary_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "plots": plot_files,
                "table": "alpha_diagnostics.csv",
            },
            handle,
            indent=2,
        )


def save_task_metric_curves(summary: dict, out_path: Path, dpi: int) -> None:
    task_names = sorted(summary["tasks"])
    colors = _task_colors(_sorted_config_names(summary))
    fig, axes = plt.subplots(
        len(task_names),
        len(PLOT_METRICS),
        figsize=(5 * len(PLOT_METRICS), 3.5 * len(task_names)),
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, task_name in enumerate(task_names):
        task_payload = summary["tasks"][task_name]
        for col_idx, metric_name in enumerate(PLOT_METRICS):
            ax = axes[row_idx][col_idx]
            for config_name in sorted(task_payload["configs"]):
                budget_entries = task_payload["configs"][config_name]["budgets"]
                budgets = np.array([entry["budget"] for entry in budget_entries], dtype=float)
                means = np.array(
                    [entry["metrics"][metric_name]["mean"] for entry in budget_entries],
                    dtype=float,
                )
                stds = np.array(
                    [entry["metrics"][metric_name]["std"] for entry in budget_entries],
                    dtype=float,
                )
                ax.plot(
                    budgets,
                    means,
                    label=config_name,
                    color=colors[config_name],
                    linewidth=2,
                )
                ax.fill_between(
                    budgets,
                    means - stds,
                    means + stds,
                    color=colors[config_name],
                    alpha=0.15,
                )
            ax.set_title(f"{task_name}: {metric_name}")
            ax.set_xlabel("Train budget")
            ax.set_ylabel(metric_name)
            ax.grid(alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncols=min(3, len(labels)),
        )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_task_hyperparameter_sweeps(
    task_name: str,
    task_records: list[dict],
    out_png: Path,
    dpi: int,
) -> None:
    config_names = sorted({str(record["config"]) for record in task_records})
    budget_names = [str(budget) for budget in sorted({int(record["budget"]) for record in task_records})]
    colors = _task_colors(budget_names)
    ncols = 3
    nrows = int(np.ceil(len(config_names) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.4 * ncols, 3.8 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    for idx, config_name in enumerate(config_names):
        ax = axes[idx // ncols][idx % ncols]
        config_records = [record for record in task_records if record["config"] == config_name]
        budgets = sorted({int(record["budget"]) for record in config_records})
        for budget in budgets:
            budget_records = [record for record in config_records if int(record["budget"]) == budget]
            search_values = sorted(
                {float(record["search_value"]) for record in budget_records},
            )
            means = []
            stds = []
            for search_value in search_values:
                values = np.array(
                    [
                        float(record["score"])
                        for record in budget_records
                        if np.isclose(float(record["search_value"]), search_value)
                    ],
                    dtype=float,
                )
                means.append(float(np.mean(values)))
                stds.append(float(np.std(values)))
            x = np.array(search_values, dtype=float)
            y = np.array(means, dtype=float)
            yerr = np.array(stds, dtype=float)
            ax.plot(
                x,
                y,
                linewidth=2,
                marker="o",
                color=colors[str(budget)],
                label=f"budget={budget}",
            )
            ax.fill_between(
                x,
                y - yerr,
                y + yerr,
                color=colors[str(budget)],
                alpha=0.15,
            )
            best_idx = int(np.argmax(y))
            ax.scatter(
                [x[best_idx]],
                [y[best_idx]],
                color=colors[str(budget)],
                s=35,
                zorder=3,
            )
        ax.set_xscale("log")
        ax.set_title(config_name)
        ax.set_xlabel("alpha / alpha_max")
        ax.set_ylabel("Tune Spearman")
        ax.grid(alpha=0.25)

    for idx in range(len(config_names), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(f"{task_name}: hyperparameter sweep", fontsize=14)
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


def save_global_metric_curves(summary: dict, out_path: Path, dpi: int) -> None:
    config_names = _sorted_config_names(summary)
    colors = _task_colors(config_names)
    fig, axes = plt.subplots(
        1,
        len(PLOT_METRICS),
        figsize=(5 * len(PLOT_METRICS), 4.5),
        squeeze=False,
        constrained_layout=True,
    )

    for col_idx, metric_name in enumerate(PLOT_METRICS):
        ax = axes[0][col_idx]
        for config_name in config_names:
            budget_to_values: dict[int, list[float]] = {}
            for task_payload in summary["tasks"].values():
                config_payload = task_payload["configs"].get(config_name)
                if config_payload is None:
                    continue
                for entry in config_payload["budgets"]:
                    budget_to_values.setdefault(entry["budget"], []).append(
                        float(entry["metrics"][metric_name]["mean"])
                    )

            budgets = sorted(budget_to_values)
            means = np.array(
                [np.mean(budget_to_values[budget]) for budget in budgets], dtype=float
            )
            stds = np.array(
                [np.std(budget_to_values[budget]) for budget in budgets], dtype=float
            )
            ax.plot(budgets, means, label=config_name, color=colors[config_name], linewidth=2)
            ax.fill_between(
                budgets,
                means - stds,
                means + stds,
                color=colors[config_name],
                alpha=0.15,
            )

        ax.set_title(f"Cross-task average: {metric_name}")
        ax.set_xlabel("Train budget")
        ax.set_ylabel(metric_name)
        ax.grid(alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            ncols=min(3, len(labels)),
        )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _collect_config_budget_values(summary: dict, metric_name: str) -> dict[str, dict[int, list[float]]]:
    config_budget_values: dict[str, dict[int, list[float]]] = {}
    for task_payload in summary["tasks"].values():
        for config_name, config_payload in task_payload["configs"].items():
            budget_map = config_budget_values.setdefault(config_name, {})
            for entry in config_payload["budgets"]:
                budget = int(entry["budget"])
                value = float(entry["metrics"][metric_name]["mean"])
                budget_map.setdefault(budget, []).append(value)
    return config_budget_values


def _write_metric_means_table(
    config_budget_values: dict[str, dict[int, list[float]]],
    out_csv: Path,
    metric_name: str,
) -> None:
    rows: list[dict[str, object]] = []
    for config_name in sorted(config_budget_values):
        for budget in sorted(config_budget_values[config_name]):
            values = np.array(config_budget_values[config_name][budget], dtype=float)
            rows.append(
                {
                    "config": config_name,
                    "budget": budget,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "n_tasks": int(len(values)),
                    "metric": metric_name,
                }
            )
    with open(out_csv, "w", encoding="utf-8", newline="") as handle:
        handle.write("config,budget,mean,std,n_tasks,metric\n")
        for row in rows:
            handle.write(
                f"{row['config']},{row['budget']},{row['mean']},{row['std']},{row['n_tasks']},{row['metric']}\n"
            )


def _plot_cross_task_metric_means(
    config_budget_values: dict[str, dict[int, list[float]]],
    metric_name: str,
    out_png: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    colors = _task_colors(sorted(config_budget_values))

    for config_name in sorted(config_budget_values):
        budgets = sorted(config_budget_values[config_name])
        means = np.array(
            [np.mean(config_budget_values[config_name][budget]) for budget in budgets],
            dtype=float,
        )
        stds = np.array(
            [np.std(config_budget_values[config_name][budget]) for budget in budgets],
            dtype=float,
        )
        ax.plot(
            budgets,
            means,
            label=config_name,
            linewidth=2,
            marker="o",
            color=colors[config_name],
        )
        ax.fill_between(
            budgets,
            means - stds,
            means + stds,
            color=colors[config_name],
            alpha=0.15,
        )

    ax.set_title(f"Cross-task mean {metric_name} by train budget")
    ax.set_xlabel("Train budget")
    ax.set_ylabel(metric_name)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _plot_per_task_metric_subplots(
    summary: dict,
    metric_name: str,
    out_png: Path,
    dpi: int,
) -> None:
    task_names = sorted(summary["tasks"])
    config_names = _sorted_config_names(summary)
    colors = _task_colors(config_names)
    ncols = 3
    nrows = int(np.ceil(len(task_names) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.3 * ncols, 3.4 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    for idx, task_name in enumerate(task_names):
        ax = axes[idx // ncols][idx % ncols]
        task_payload = summary["tasks"][task_name]
        for config_name in config_names:
            config_payload = task_payload["configs"].get(config_name)
            if config_payload is None:
                continue
            budgets = np.array(
                [int(entry["budget"]) for entry in config_payload["budgets"]],
                dtype=float,
            )
            means = np.array(
                [float(entry["metrics"][metric_name]["mean"]) for entry in config_payload["budgets"]],
                dtype=float,
            )
            ax.plot(
                budgets,
                means,
                linewidth=2,
                marker="o",
                label=config_name,
                color=colors[config_name],
            )
        ax.set_title(task_name)
        ax.set_xlabel("Train budget")
        ax.set_ylabel(metric_name)
        ax.grid(alpha=0.25)

    for idx in range(len(task_names), nrows * ncols):
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


def save_metric_heatmap(summary: dict, metric_name: str, out_path: Path, dpi: int) -> None:
    task_names, config_names, matrix = _collect_global_matrix(summary, metric_name)

    fig, ax = plt.subplots(figsize=(1.8 * len(config_names) + 2, 0.5 * len(task_names) + 2))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(config_names)))
    ax.set_xticklabels(config_names, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(task_names)))
    ax.set_yticklabels(task_names)
    ax.set_title(f"Best-budget {metric_name}")

    for row_idx in range(len(task_names)):
        for col_idx in range(len(config_names)):
            value = matrix[row_idx, col_idx]
            if np.isnan(value):
                label = "NA"
            else:
                label = f"{value:.2f}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
