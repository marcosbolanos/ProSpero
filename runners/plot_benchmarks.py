import argparse
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it with `pip install matplotlib`."
    ) from exc


METRICS = [
    ("Mean max score", "Std max score", "mean_max_score"),
    ("Mean performance", "Std performance", "mean_performance"),
    ("Mean novelty", "Std novelty", "mean_novelty"),
    ("Mean diversity", "Std diversity", "mean_diversity"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate benchmark plots for a results folder."
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        required=True,
        help="Path to outputs folder (example: outputs/out_190226)",
    )
    parser.add_argument(
        "--plots_dirname",
        type=str,
        default="plots",
        help="Name of plot output subdirectory inside outputs_dir.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure export DPI.",
    )
    parser.add_argument(
        "--no_seed_fallback",
        action="store_true",
        help="Disable computing aggregate curves from seed_*.pkl when transformed_results.json is missing.",
    )
    parser.add_argument(
        "--k_task_name",
        type=str,
        default="AAV",
        help="Task name to use for K-comparison plots in nested layouts.",
    )
    parser.add_argument(
        "--k_parent_prefix",
        type=str,
        default="n_samples_",
        help="Directory prefix that identifies K buckets in nested layouts.",
    )
    return parser.parse_args()


def sorted_iter_keys(data_dict):
    return sorted((int(k) for k in data_dict.keys()))


def load_from_transformed(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    iters = sorted_iter_keys(raw)
    result = {"iters": iters, "source": "transformed_results.json"}
    for mean_key, std_key, out_name in METRICS:
        mean_vals = []
        std_vals = []
        for it in iters:
            entry = raw[str(it)]
            mean_vals.append(float(entry[mean_key]))
            std_vals.append(float(entry[std_key]))
        result[out_name] = np.array(mean_vals, dtype=float)
        result[out_name + "_std"] = np.array(std_vals, dtype=float)
    return result


def aggregate_seed_iteration(iter_entries):
    best_scores = np.array([float(x["Best score"]) for x in iter_entries], dtype=float)
    performances = np.array(
        [float(x["Performance"]) for x in iter_entries], dtype=float
    )
    diversity = np.array([float(x["Diversity"]) for x in iter_entries], dtype=float)
    novelty = np.array([float(x["WT Novelty"]) for x in iter_entries], dtype=float)

    return {
        "Mean max score": float(np.mean(best_scores)),
        "Std max score": float(np.std(best_scores)),
        "Mean performance": float(np.mean(performances)),
        "Std performance": float(np.std(performances)),
        "Mean novelty": float(np.mean(novelty)),
        "Std novelty": float(np.std(novelty)),
        "Mean diversity": float(np.mean(diversity)),
        "Std diversity": float(np.std(diversity)),
    }


def load_from_seeds(task_dir):
    seed_paths = sorted(task_dir.glob("seed_*.pkl"))
    if not seed_paths:
        return None

    seed_data = []
    for p in seed_paths:
        with open(p, "rb") as f:
            seed_data.append(pickle.load(f))

    common_iters = None
    for run in seed_data:
        keys = set(int(k) for k in run.keys())
        if common_iters is None:
            common_iters = keys
        else:
            common_iters &= keys

    if not common_iters:
        return None

    iters = sorted(common_iters)
    packed = {"iters": iters, "source": f"{len(seed_paths)} seed pkl files"}
    metric_storage = {
        "Mean max score": [],
        "Std max score": [],
        "Mean performance": [],
        "Std performance": [],
        "Mean novelty": [],
        "Std novelty": [],
        "Mean diversity": [],
        "Std diversity": [],
    }

    for it in iters:
        iter_entries = [run[it] for run in seed_data]
        agg = aggregate_seed_iteration(iter_entries)
        for k, v in agg.items():
            metric_storage[k].append(v)

    for mean_key, std_key, out_name in METRICS:
        packed[out_name] = np.array(metric_storage[mean_key], dtype=float)
        packed[out_name + "_std"] = np.array(metric_storage[std_key], dtype=float)

    return packed


def collect_task_data(outputs_dir, allow_seed_fallback=True, excluded_dirs=None):
    task_data = {}
    skipped = {}
    excluded_dirs = set() if excluded_dirs is None else set(excluded_dirs)

    for entry in sorted(outputs_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in excluded_dirs:
            continue

        transformed = entry / "transformed_results.json"
        if transformed.exists():
            task_data[entry.name] = load_from_transformed(transformed)
            continue

        if allow_seed_fallback:
            seed_based = load_from_seeds(entry)
            if seed_based is not None:
                task_data[entry.name] = seed_based
                continue

        skipped[entry.name] = "no transformed_results.json and no usable seed_*.pkl"

    return task_data, skipped


def parse_trailing_int(name):
    match = re.search(r"(\d+)$", name)
    if match is None:
        return None
    return int(match.group(1))


def collect_k_task_data(
    outputs_dir,
    task_name,
    parent_prefix="n_samples_",
    allow_seed_fallback=True,
    excluded_dirs=None,
):
    task_data = {}
    skipped = {}
    excluded_dirs = set() if excluded_dirs is None else set(excluded_dirs)

    entries = []
    for entry in sorted(outputs_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in excluded_dirs:
            continue
        if not entry.name.startswith(parent_prefix):
            continue
        entries.append(entry)

    for entry in entries:
        task_dir = entry / task_name
        label = entry.name
        k_value = parse_trailing_int(entry.name)
        if k_value is not None:
            label = f"K={k_value}"

        if not task_dir.exists() or not task_dir.is_dir():
            skipped[entry.name] = f"missing nested task directory {task_name}"
            continue

        transformed = task_dir / "transformed_results.json"
        payload = None
        if transformed.exists():
            payload = load_from_transformed(transformed)
        elif allow_seed_fallback:
            payload = load_from_seeds(task_dir)

        if payload is None:
            skipped[entry.name] = "no transformed_results.json and no usable seed_*.pkl"
            continue

        payload["k_value"] = k_value
        payload["bucket_name"] = entry.name
        task_data[label] = payload

    return task_data, skipped


def task_colors(task_names):
    cmap = plt.get_cmap("tab10")
    return {name: cmap(i % 10) for i, name in enumerate(task_names)}


def save_metric_curves(task_data, out_path, dpi):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    colors = task_colors(sorted(task_data))

    for idx, (mean_key, _, title_key) in enumerate(METRICS):
        ax = axes[idx // 2, idx % 2]
        std_key = title_key + "_std"
        for task in sorted(task_data):
            payload = task_data[task]
            x = np.array(payload["iters"], dtype=float)
            y = payload[title_key]
            y_std = payload[std_key]

            ax.plot(x, y, label=task, color=colors[task], linewidth=2)
            ax.fill_between(x, y - y_std, y + y_std, color=colors[task], alpha=0.15)

        ax.set_title(mean_key)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def final_metric_values(task_data, metric_key):
    values = []
    for task in sorted(task_data):
        arr = task_data[task][metric_key]
        values.append(float(arr[-1]))
    return np.array(values, dtype=float)


def save_grouped_final_bars(task_data, out_path, dpi):
    tasks = sorted(task_data)
    x = np.arange(len(tasks), dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for idx, (mean_key, _, data_key) in enumerate(METRICS):
        ax = axes[idx // 2, idx % 2]
        vals = final_metric_values(task_data, data_key)
        ax.bar(x, vals)
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=40, ha="right")
        ax.set_title("Final " + mean_key)
        ax.grid(axis="y", alpha=0.3)

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def normalize_gain(arr):
    baseline = float(arr[0])
    if np.isclose(baseline, 0.0):
        return arr - baseline
    return (arr - baseline) / abs(baseline)


def save_normalized_gains(task_data, out_path, dpi):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    colors = task_colors(sorted(task_data))

    for idx, (mean_key, _, data_key) in enumerate(METRICS):
        ax = axes[idx // 2, idx % 2]
        for task in sorted(task_data):
            payload = task_data[task]
            x = np.array(payload["iters"], dtype=float)
            y = normalize_gain(payload[data_key])
            ax.plot(x, y, label=task, color=colors[task], linewidth=2)

        ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
        ax.set_title(mean_key + " normalized gain")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Relative change vs iter1")
        ax.grid(alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_pareto(task_data, out_path, dpi):
    tasks = sorted(task_data)
    perf = final_metric_values(task_data, "mean_performance")
    nov = final_metric_values(task_data, "mean_novelty")
    div = final_metric_values(task_data, "mean_diversity")

    marker_sizes = 80 + 20 * (div - div.min()) / (np.ptp(div) + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    sc = ax.scatter(perf, nov, s=marker_sizes, c=div, cmap="viridis", alpha=0.9)

    for i, task in enumerate(tasks):
        ax.annotate(task, (perf[i], nov[i]), xytext=(5, 4), textcoords="offset points")

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Final mean diversity")
    ax.set_xlabel("Final mean performance")
    ax.set_ylabel("Final mean novelty")
    ax.set_title("Final Pareto view: performance vs novelty")
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def first_iter_reaching_ratio(arr, ratio):
    target = float(arr[-1]) * ratio
    for i, value in enumerate(arr):
        if value >= target:
            return i + 1
    return len(arr)


def save_convergence(task_data, out_path, dpi, ratio=0.9):
    tasks = sorted(task_data)
    iters = [
        first_iter_reaching_ratio(task_data[t]["mean_max_score"], ratio) for t in tasks
    ]

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    x = np.arange(len(tasks), dtype=float)
    ax.bar(x, iters)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=40, ha="right")
    ax.set_ylabel("Iteration index")
    ax.set_title(f"Convergence speed (first iter >= {int(ratio * 100)}% of final max)")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def zscore(v):
    std = np.std(v)
    if np.isclose(std, 0.0):
        return np.zeros_like(v)
    return (v - np.mean(v)) / std


def save_final_heatmap(task_data, out_path, dpi):
    tasks = sorted(task_data)
    metric_names = ["max", "perf", "novelty", "diversity"]
    matrix = np.vstack(
        [
            final_metric_values(task_data, "mean_max_score"),
            final_metric_values(task_data, "mean_performance"),
            final_metric_values(task_data, "mean_novelty"),
            final_metric_values(task_data, "mean_diversity"),
        ]
    ).T

    matrix_z = np.column_stack([zscore(matrix[:, i]) for i in range(matrix.shape[1])])

    fig, ax = plt.subplots(
        figsize=(9, max(5, 0.6 * len(tasks))), constrained_layout=True
    )
    im = ax.imshow(matrix_z, aspect="auto", cmap="coolwarm")
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels(tasks)
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_xticklabels(metric_names)
    ax.set_title("Final metrics heatmap (z-score by metric)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("z-score")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_prospero_reproduction(task_data, out_path, dpi):
    ordered_tasks = ["AMIE", "TEM", "E4B", "Pab1", "AAV", "GFP", "UBE2I", "LGK"]
    present_tasks = [task for task in ordered_tasks if task in task_data]
    if not present_tasks:
        return

    fig, axes = plt.subplots(2, 4, figsize=(12.2, 5.7), constrained_layout=True)
    x_ticks = [2, 4, 6, 8, 10]

    for i, ax in enumerate(axes.flat):
        if i >= len(present_tasks):
            ax.axis("off")
            continue

        task = present_tasks[i]
        payload = task_data[task]
        x = np.array(payload["iters"], dtype=float)
        y = payload["mean_max_score"]
        y_std = payload["mean_max_score_std"]

        ax.set_facecolor("#f0f0f0")
        ax.grid(True, color="#bfbfbf", linewidth=1.1)
        ax.plot(
            x,
            y,
            color="red",
            marker="o",
            linewidth=2.0,
            markersize=3,
            label="ProSpero (ours)",
        )
        ax.fill_between(x, y - y_std, y + y_std, color="red", alpha=0.18)

        y_min = float(np.min(y - y_std))
        y_max = float(np.max(y + y_std))
        pad = max((y_max - y_min) * 0.14, 1e-4)
        ax.set_ylim(y_min - pad, y_max + pad)

        ax.set_title(task)
        ax.set_xlim(float(np.min(x)), float(np.max(x)))
        ax.set_xticks([tick for tick in x_ticks if tick <= x.max()])

        if i < 4:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Active learning rounds")

        if i % 4 == 0:
            ax.set_ylabel("Maximum")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, frameon=False)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def sort_k_labels(k_task_data):
    items = []
    for label, payload in k_task_data.items():
        k_value = payload.get("k_value")
        rank = float("inf") if k_value is None else float(k_value)
        items.append((rank, label))
    return [label for _, label in sorted(items, key=lambda x: (x[0], x[1]))]


def save_k_comparison_curve(k_task_data, out_path, dpi, task_name="AAV"):
    ordered_labels = sort_k_labels(k_task_data)
    if not ordered_labels:
        return

    fig, ax = plt.subplots(figsize=(10.5, 6.0), constrained_layout=True)
    colors = task_colors(ordered_labels)

    for label in ordered_labels:
        payload = k_task_data[label]
        x = np.array(payload["iters"], dtype=float)
        y = payload["mean_max_score"]
        y_std = payload["mean_max_score_std"]
        ax.plot(x, y, label=label, color=colors[label], linewidth=2.2)
        ax.fill_between(x, y - y_std, y + y_std, color=colors[label], alpha=0.16)

    ax.set_title(f"{task_name}: fitness by active learning round across K")
    ax.set_xlabel("Active learning rounds")
    ax.set_ylabel("Mean max score")
    ax.grid(alpha=0.3)
    ax.legend(title="K", ncol=min(3, len(ordered_labels)), frameon=False)

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def write_manifest(task_data, skipped, out_path):
    payload = {
        "tasks_plotted": {
            task: {
                "source": task_data[task]["source"],
                "n_iterations": len(task_data[task]["iters"]),
            }
            for task in sorted(task_data)
        },
        "tasks_skipped": skipped,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists() or not outputs_dir.is_dir():
        raise SystemExit(f"Invalid outputs directory: {outputs_dir}")

    plots_dir = outputs_dir / args.plots_dirname
    os.makedirs(plots_dir, exist_ok=True)

    task_data, skipped = collect_task_data(
        outputs_dir,
        allow_seed_fallback=(not args.no_seed_fallback),
        excluded_dirs={args.plots_dirname},
    )

    k_task_data, k_skipped = collect_k_task_data(
        outputs_dir,
        task_name=args.k_task_name,
        parent_prefix=args.k_parent_prefix,
        allow_seed_fallback=(not args.no_seed_fallback),
        excluded_dirs={args.plots_dirname},
    )

    # If the folder is organized in nested K buckets (for example n_samples_*/AAV),
    # these parent folders are not direct tasks and should not be reported as skipped
    # by the direct-task collector.
    if k_task_data:
        skipped = {
            name: reason
            for name, reason in skipped.items()
            if not name.startswith(args.k_parent_prefix)
        }

    if not task_data and not k_task_data:
        raise SystemExit(
            "No plottable task data found. Expected direct task dirs or nested K dirs with transformed_results.json or seed_*.pkl."
        )

    if task_data:
        save_metric_curves(
            task_data, plots_dir / "metric_curves_mean_std.png", args.dpi
        )
        save_grouped_final_bars(
            task_data, plots_dir / "final_iteration_grouped_bars.png", args.dpi
        )
        save_normalized_gains(
            task_data, plots_dir / "normalized_gain_curves.png", args.dpi
        )
        save_pareto(task_data, plots_dir / "pareto_final_perf_vs_novelty.png", args.dpi)
        save_convergence(task_data, plots_dir / "convergence_speed.png", args.dpi)
        save_final_heatmap(
            task_data, plots_dir / "heatmap_final_metrics_zscore.png", args.dpi
        )
        save_prospero_reproduction(
            task_data, plots_dir / "prospero_maximum_grid.png", args.dpi
        )
        write_manifest(task_data, skipped, plots_dir / "plots_manifest.json")

    if k_task_data:
        k_plot_name = f"k_comparison_{args.k_task_name}_mean_max_score.png"
        save_k_comparison_curve(
            k_task_data,
            plots_dir / k_plot_name,
            args.dpi,
            task_name=args.k_task_name,
        )

    print(f"Saved plots to: {plots_dir}")
    if task_data:
        print(f"Tasks plotted: {', '.join(sorted(task_data))}")
    if k_task_data:
        print(f"K-comparison plotted: {', '.join(sort_k_labels(k_task_data))}")

    all_skipped = dict(skipped)
    all_skipped.update({f"K/{k}": v for k, v in k_skipped.items()})
    if all_skipped:
        print("Tasks skipped:")
        for task, reason in sorted(all_skipped.items()):
            print(f"- {task}: {reason}")


if __name__ == "__main__":
    main()
