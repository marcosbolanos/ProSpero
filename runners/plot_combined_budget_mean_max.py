from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from prospero.plotting_style import COLORS, set_prospero_style


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "rl_vs_og" / "zero_shotprot_combined_budgets"
TASKS = ["AAV", "LGK", "GFP", "Pab1", "AMIE", "E4B", "TEM", "UBE2I"]
BUDGETS = [128, 8]

DEFAULT_OG_ROOTS = {
    "AAV": ROOT / "outputs/variable_k_cnn_excl_set_noa6000_20260504_175400/AAV_cnn",
    "LGK": ROOT / "outputs/out_240226_lgk_cnn",
    "GFP": ROOT / "outputs/out_240226_gfp_cnn",
    "Pab1": ROOT / "outputs/out_240226_pab1_cnn",
    "AMIE": ROOT / "outputs/out_240226_amie_cnn",
    "E4B": ROOT / "outputs/variable_k_cnn_excl_set_noa6000_20260504_175400/E4B_cnn",
    "TEM": ROOT / "outputs/out_240226_tem_cnn",
    "UBE2I": ROOT / "outputs/variable_k_cnn_excl_set_noa6000_20260504_175400/UBE2I_cnn",
}

DEFAULT_PROSST_ROOTS = {
    8: ROOT / "outputs/prosst_ft_all_landscapes_n8_grpo_cluster_20260604",
    128: ROOT / "outputs/prosst_ft_all_landscapes_n128_grpo_cluster_20260604",
}


@dataclass(frozen=True)
class Method:
    label: str
    color: str
    marker: str
    root_kind: str
    root: Path
    strategy: str | None = None


METHODS = [
    Method("ProSST", COLORS["prosst"], "s", "prosst", Path(".")),
    Method("ProSpero", COLORS["ink"], "o", "prospero", Path(".")),
]


def pastel(hex_color: str, amount: float = 0.62) -> str:
    hex_color = hex_color.lstrip("#")
    rgb = np.array([int(hex_color[i : i + 2], 16) for i in (0, 2, 4)], dtype=float)
    mixed = rgb * (1.0 - amount) + np.array([255.0, 255.0, 255.0]) * amount
    return "#" + "".join(f"{int(round(v)):02X}" for v in mixed)


def seed_paths(
    method: Method,
    task: str,
    budget: int,
    prosst_roots: dict[int, Path],
    prospero_roots: dict[str, Path],
) -> list[Path]:
    if method.root_kind == "prosst":
        return sorted((prosst_roots[budget] / task).glob("seed_*.pkl"))
    if method.root_kind == "prospero":
        return sorted((prospero_roots[task] / f"n_samples_{budget}" / task).glob("seed_*.pkl"))
    raise ValueError(method.root_kind)


def load_seed(path: Path) -> dict[int, float]:
    try:
        with path.open("rb") as handle:
            data = pickle.load(handle)
    except Exception as exc:
        print(f"skip unreadable {path}: {exc}")
        return {}
    return {
        int(k): float(v["Best score"])
        for k, v in data.items()
        if isinstance(k, int) and isinstance(v, dict) and "Best score" in v
    }


def aggregate(
    method: Method,
    task: str,
    budget: int,
    prosst_roots: dict[int, Path],
    prospero_roots: dict[str, Path],
):
    rows = [load_seed(path) for path in seed_paths(method, task, budget, prosst_roots, prospero_roots)]
    xs = np.arange(1, 11)
    means, sems, counts = [], [], []
    for it in xs:
        vals = np.array([row[it] for row in rows if it in row], dtype=float)
        counts.append(int(len(vals)))
        if len(vals) == 0:
            means.append(np.nan)
            sems.append(np.nan)
            continue
        means.append(float(vals.mean()))
        sems.append(float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0)
    return xs, np.array(means), np.array(sems), counts


def should_add_inset(series: list[tuple[Method, int, np.ndarray, np.ndarray, np.ndarray]]) -> bool:
    all_values = []
    late_values = []
    for _, _, x, y, e in series:
        valid = np.isfinite(y)
        if not valid.any():
            continue
        all_values.extend((y[valid] - e[valid]).tolist())
        all_values.extend((y[valid] + e[valid]).tolist())
        late = valid & (x >= 6)
        late_values.extend((y[late] - e[late]).tolist())
        late_values.extend((y[late] + e[late]).tolist())
    if not all_values or not late_values:
        return False
    full_range = float(np.nanmax(all_values) - np.nanmin(all_values))
    late_range = float(np.nanmax(late_values) - np.nanmin(late_values))
    if full_range <= 0:
        return False
    return late_range / full_range <= 0.36


def add_zoom_inset(
    ax,
    series: list[tuple[Method, int, np.ndarray, np.ndarray, np.ndarray]],
):
    if not should_add_inset(series):
        return None

    x_min, x_max = 6, 10
    zoom_values = []
    for _, _, x, y, e in series:
        keep = np.isfinite(y) & (x >= x_min) & (x <= x_max)
        if keep.any():
            zoom_values.extend((y[keep] - e[keep]).tolist())
            zoom_values.extend((y[keep] + e[keep]).tolist())
    if not zoom_values:
        return None
    y_min = float(np.nanmin(zoom_values))
    y_max = float(np.nanmax(zoom_values))
    pad = max((y_max - y_min) * 0.18, abs(y_max) * 0.002, 1e-6)

    axins = inset_axes(
        ax,
        width="54%",
        height="48%",
        loc="lower right",
        borderpad=1.35,
    )
    for method, budget, x, y, e in series:
        keep = np.isfinite(y) & (x >= x_min) & (x <= x_max)
        if not keep.any():
            continue
        color = method.color if budget == 128 else pastel(method.color)
        linewidth = 2.25 if budget == 128 else 1.85
        axins.plot(
            x[keep],
            y[keep],
            color=color,
            marker=method.marker,
            linewidth=linewidth,
            markersize=3.9,
            alpha=0.98,
        )
        axins.fill_between(
            x[keep],
            y[keep] - e[keep],
            y[keep] + e[keep],
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    axins.set_xlim(x_min, x_max)
    axins.set_ylim(y_min - pad, y_max + pad)
    axins.set_xticks([6, 8, 10])
    axins.tick_params(axis="both", labelsize=13, pad=1)
    axins.grid(True, axis="y", linewidth=0.45, alpha=0.35)
    for spine in axins.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color(COLORS["muted"])
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec=COLORS["muted"], lw=0.85, alpha=0.85)
    return {"xlim": (x_min, x_max), "ylim": (y_min - pad, y_max + pad)}


def legend_handles():
    handles = []
    labels = []
    for method in METHODS:
        handles.append(
            Line2D(
                [0],
                [0],
                color=method.color,
                marker=method.marker,
                linewidth=3.2,
                markersize=7.0,
            )
        )
        labels.append(f"{method.label} (K=128)")
        handles.append(
            Line2D(
                [0],
                [0],
                color=pastel(method.color),
                marker=method.marker,
                linewidth=2.7,
                markersize=7.0,
            )
        )
        labels.append(f"{method.label} (K=8)")
    return handles, labels


def plot(
    output_dir: Path = OUT,
    prosst_k8_root: Path = DEFAULT_PROSST_ROOTS[8],
    prosst_k128_root: Path = DEFAULT_PROSST_ROOTS[128],
    prospero_results_dir: Path | None = None,
):
    prosst_roots = {8: Path(prosst_k8_root), 128: Path(prosst_k128_root)}
    prospero_roots = (
        {task: Path(prospero_results_dir) / f"{task}_cnn" for task in TASKS}
        if prospero_results_dir is not None
        else DEFAULT_OG_ROOTS
    )
    set_prospero_style()
    plt.rcParams.update(
        {
            "axes.titlesize": 32,
            "axes.labelsize": 28,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
        }
    )
    fig, axes = plt.subplots(2, 4, figsize=(30.0, 16.0), sharex=True)
    summary = []
    zoom_summary = []
    for idx, task in enumerate(TASKS):
        ax = axes[idx // 4, idx % 4]
        series = []
        for method in METHODS:
            for budget in BUDGETS:
                x, y, e, counts = aggregate(method, task, budget, prosst_roots, prospero_roots)
                valid = np.isfinite(y)
                if not valid.any():
                    summary.append(f"SKIP {task} {method.label} K={budget}: no data")
                    continue
                color = method.color if budget == 128 else pastel(method.color)
                linewidth = 3.15 if budget == 128 else 2.45
                alpha = 0.98 if budget == 128 else 0.96
                zorder = 4 if budget == 128 else 3
                ax.plot(
                    x[valid],
                    y[valid],
                    color=color,
                    marker=method.marker,
                    linewidth=linewidth,
                    markersize=6.4,
                    alpha=alpha,
                    zorder=zorder,
                )
                ax.fill_between(
                    x[valid],
                    y[valid] - e[valid],
                    y[valid] + e[valid],
                    color=color,
                    alpha=0.10 if budget == 128 else 0.13,
                    linewidth=0,
                    zorder=zorder - 1,
                )
                summary.append(
                    f"{task} {method.label} K={budget}: counts={counts} "
                    f"round10={y[-1]:.6g} sem={e[-1]:.3g}"
                )
                series.append((method, budget, x, y, e))
        ax.set_title(task, loc="left", pad=6)
        ax.set_xlim(1, 10)
        ax.set_xticks(range(1, 11))
        ax.grid(True, axis="y")
        zoom_meta = add_zoom_inset(ax, series)
        if zoom_meta is not None:
            zoom_summary.append(
                f"{task} zoom xlim={zoom_meta['xlim']} "
                f"ylim=({zoom_meta['ylim'][0]:.6g}, {zoom_meta['ylim'][1]:.6g})"
            )
        if idx % 4 == 0:
            ax.set_ylabel("Mean max fitness")
        if idx // 4 == 1:
            ax.set_xlabel("Optimization round")

    handles, labels = legend_handles()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(METHODS),
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        columnspacing=1.8,
        handlelength=2.2,
        handletextpad=0.55,
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.09, top=0.80, wspace=0.42, hspace=0.42)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "zero_shotprot_combined_k8_k128_mean_max.png",
        output_dir / "zero_shotprot_combined_k8_k128_mean_max.pdf",
        output_dir / "zero_shotprot_combined_k8_k128_mean_max.svg",
    ]
    for path in paths:
        fig.savefig(path, dpi=320 if path.suffix == ".png" else None, bbox_inches="tight")
    plt.close(fig)
    (output_dir / "plot_summary.txt").write_text(
        "Written plots:\n"
        + "\n".join(map(str, paths))
        + "\n\nSummary:\n"
        + "\n".join(summary + zoom_summary)
        + "\n",
        encoding="utf-8",
    )
    return paths


def get_parser():
    parser = argparse.ArgumentParser(description="Plot K=8 and K=128 mean-max trajectories.")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--prosst-k8-root", type=Path, default=DEFAULT_PROSST_ROOTS[8])
    parser.add_argument("--prosst-k128-root", type=Path, default=DEFAULT_PROSST_ROOTS[128])
    parser.add_argument("--prospero-results-dir", type=Path, default=None)
    return parser


def main():
    args = get_parser().parse_args()
    paths = plot(
        output_dir=args.output_dir,
        prosst_k8_root=args.prosst_k8_root,
        prosst_k128_root=args.prosst_k128_root,
        prospero_results_dir=args.prospero_results_dir,
    )
    print(f"Wrote {len(paths)} plot files")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
