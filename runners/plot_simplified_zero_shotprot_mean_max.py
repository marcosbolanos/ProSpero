from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from prospero.plotting_style import COLORS, set_prospero_style

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "rl_vs_og" / "zero_shotprot_simplified"
TASKS = ["AAV", "LGK", "GFP", "Pab1", "AMIE", "E4B", "TEM", "UBE2I"]
BUDGETS = [8, 128]

OG_ROOTS = {
    "AAV": ROOT / "outputs/variable_k_cnn_excl_set_noa6000_20260504_175400/AAV_cnn",
    "LGK": ROOT / "outputs/out_240226_lgk_cnn",
    "GFP": ROOT / "outputs/out_240226_gfp_cnn",
    "Pab1": ROOT / "outputs/out_240226_pab1_cnn",
    "AMIE": ROOT / "outputs/out_240226_amie_cnn",
    "E4B": ROOT / "outputs/variable_k_cnn_excl_set_noa6000_20260504_175400/E4B_cnn",
    "TEM": ROOT / "outputs/out_240226_tem_cnn",
    "UBE2I": ROOT / "outputs/variable_k_cnn_excl_set_noa6000_20260504_175400/UBE2I_cnn",
}

EVODIFF_ROOTS = {
    "AAV": (ROOT / "outputs/aav_zero_shot_evodiff_ft_rank_mixed_k4_variable_k_kl2_trace_batch64_20260603", "mixed_explore_exploit"),
    "LGK": (ROOT / "outputs/lgk_zero_shot_evodiff_ft_k4_variable_k_kl2_20260530", "seed_grow"),
    "GFP": (ROOT / "outputs/gfp_zero_shot_evodiff_ft_k4_variable_k_kl2_20260530", "seed_grow"),
    "Pab1": (ROOT / "outputs/pab1_zero_shot_evodiff_ft_k4_variable_k_kl2_20260530", "seed_grow"),
    "AMIE": (ROOT / "outputs/amie_zero_shot_evodiff_ft_k4_variable_k_kl2_20260531", "seed_grow"),
    "E4B": (ROOT / "outputs/e4b_zero_shot_evodiff_ft_k4_variable_k_kl2_20260531", "seed_grow"),
    "TEM": (ROOT / "outputs/tem_zero_shot_evodiff_ft_k4_variable_k_kl2_20260531", "seed_grow"),
    "UBE2I": (ROOT / "outputs/ube2i_zero_shot_evodiff_ft_k4_variable_k_kl2_20260531", "seed_grow"),
}

PROSST_ROOTS = {
    8: ROOT / "outputs/prosst_ft_all_landscapes_n8_rank_cluster_20260603",
    128: ROOT / "outputs/prosst_ft_all_landscapes_n128_rank_cluster_20260603",
}

ZOOM_INSET_TASKS = {
    8: {"GFP", "AMIE", "UBE2I"},
    128: {"TEM"},
}


@dataclass(frozen=True)
class Method:
    label: str
    color: str
    marker: str
    linewidth: float
    root_kind: str
    root: Path
    strategy: str | None = None


INCLUDE_EVODIFF = True


def methods_for(task: str, budget: int) -> list[Method]:
    ev_root, ev_strategy = EVODIFF_ROOTS[task]
    methods = [
        Method("0shotProt (w/ ProSST)", COLORS["prosst"], "s", 2.7, "prosst", PROSST_ROOTS[budget], None),
    ]
    if INCLUDE_EVODIFF:
        methods.append(Method("0shotProt (w/ EvoDiff)", COLORS["evodiff"], "^", 2.7, "standard", ev_root, ev_strategy))
    methods.append(Method("ProSpero", "#17212B", "o", 2.7, "standard", OG_ROOTS[task], None))
    return methods


def seed_paths(method: Method, task: str, budget: int) -> list[Path]:
    if method.root_kind == "prosst":
        return sorted((method.root / task).glob("seed_*.pkl"))
    if method.strategy is None:
        return sorted((method.root / f"n_samples_{budget}" / task).glob("seed_*.pkl"))
    return sorted((method.root / f"n_samples_{budget}" / method.strategy / task).glob("seed_*.pkl"))


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


def aggregate(method: Method, task: str, budget: int):
    rows = [load_seed(path) for path in seed_paths(method, task, budget)]
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


def add_zoom_inset(ax, series: list[tuple[Method, np.ndarray, np.ndarray, np.ndarray]], task: str, budget: int):
    if task not in ZOOM_INSET_TASKS.get(budget, set()):
        return None

    x_min, x_max = 3, 10
    zoom_values = []
    for _, x, y, e in series:
        keep = np.isfinite(y) & (x >= x_min) & (x <= x_max)
        if not keep.any():
            continue
        zoom_values.extend((y[keep] - e[keep]).tolist())
        zoom_values.extend((y[keep] + e[keep]).tolist())
    if not zoom_values:
        return None

    y_min = float(np.nanmin(zoom_values))
    y_max = float(np.nanmax(zoom_values))
    span = y_max - y_min
    if span <= 0:
        span = max(abs(y_max) * 0.02, 1e-3)
    pad = span * 0.18

    # Keep the inset in the lower-right whitespace where these panels have unused vertical room.
    axins = inset_axes(
        ax,
        width="63%",
        height="56%",
        loc="lower right",
        borderpad=1.05,
    )
    axins.set_facecolor("#FFFFFF")
    axins.patch.set_alpha(0.96)
    for method, x, y, e in series:
        valid = np.isfinite(y) & (x >= x_min) & (x <= x_max)
        if not valid.any():
            continue
        axins.plot(
            x[valid],
            y[valid],
            color=method.color,
            marker=method.marker,
            linewidth=2.0,
            markersize=4.2,
            zorder=3,
        )
        axins.fill_between(
            x[valid],
            y[valid] - e[valid],
            y[valid] + e[valid],
            color=method.color,
            alpha=0.10,
            linewidth=0,
            zorder=2,
        )

    axins.set_xlim(x_min, x_max)
    axins.set_ylim(y_min - pad, y_max + pad)
    axins.set_xticks([3, 6, 10])
    axins.tick_params(axis="both", labelsize=8.5, length=2.5, pad=1.5)
    axins.grid(True, axis="y", linewidth=0.55, alpha=0.55)
    for spine in axins.spines.values():
        spine.set_edgecolor(COLORS["muted"])
        spine.set_linewidth(0.8)
        spine.set_visible(True)

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec=COLORS["muted"], lw=0.85, alpha=0.85)
    return {
        "task": task,
        "budget": budget,
        "xlim": (x_min, x_max),
        "ylim": (y_min - pad, y_max + pad),
    }


def plot_budget(budget: int):
    set_prospero_style()
    plt.rcParams.update({
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
    })
    fig, axes = plt.subplots(2, 4, figsize=(18.5, 10.2), sharex=True)
    legend_seen = set()
    summary = []
    zoom_summary = []
    for idx, task in enumerate(TASKS):
        ax = axes[idx // 4, idx % 4]
        series = []
        for method in methods_for(task, budget):
            x, y, e, counts = aggregate(method, task, budget)
            valid = np.isfinite(y)
            if not valid.any():
                summary.append(f"SKIP K={budget} {task} {method.label}: no data")
                continue
            label = method.label if method.label not in legend_seen else "_nolegend_"
            legend_seen.add(method.label)
            ax.plot(
                x[valid],
                y[valid],
                color=method.color,
                marker=method.marker,
                linewidth=method.linewidth,
                markersize=6.5,
                label=label,
            )
            ax.fill_between(
                x[valid],
                y[valid] - e[valid],
                y[valid] + e[valid],
                color=method.color,
                alpha=0.11,
                linewidth=0,
            )
            summary.append(f"K={budget} {task} {method.label}: counts={counts} round10={y[-1]:.6g} sem={e[-1]:.3g}")
            series.append((method, x, y, e))
        ax.set_title(task, loc="left", pad=6)
        ax.set_xlim(1, 10)
        ax.set_xticks(range(1, 11))
        ax.grid(True, axis="y")
        zoom_meta = add_zoom_inset(ax, series, task, budget)
        if zoom_meta is not None:
            zoom_summary.append(
                f"K={budget} {task} zoom xlim={zoom_meta['xlim']} ylim="
                f"({zoom_meta['ylim'][0]:.6g}, {zoom_meta['ylim'][1]:.6g})"
            )
        if idx % 4 == 0:
            ax.set_ylabel("Mean max fitness")
        if idx // 4 == 1:
            ax.set_xlabel("Optimization round")
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.suptitle(f"K={budget} mean-max fitness trajectories", fontsize=23, fontweight="semibold", y=1.045)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.08, top=0.86, wspace=0.18, hspace=0.34)
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"zero_shotprot_vs_prospero_k{budget}_mean_max.png"
    pdf = OUT / f"zero_shotprot_vs_prospero_k{budget}_mean_max.pdf"
    svg = OUT / f"zero_shotprot_vs_prospero_k{budget}_mean_max.svg"
    fig.savefig(png, dpi=320, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf, svg], summary + zoom_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Plot simplified mean-max trajectories.")
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--prosst-k8-root", default=None)
    parser.add_argument("--prosst-k128-root", default=None)
    parser.add_argument("--prospero-results-dir", default=None)
    parser.add_argument("--no-evodiff", action="store_true", default=False)
    return parser.parse_args()


def main():
    global OUT, INCLUDE_EVODIFF
    args = parse_args()
    OUT = Path(args.output_dir)
    INCLUDE_EVODIFF = not args.no_evodiff
    if args.prosst_k8_root is not None:
        PROSST_ROOTS[8] = Path(args.prosst_k8_root)
    if args.prosst_k128_root is not None:
        PROSST_ROOTS[128] = Path(args.prosst_k128_root)
    if args.prospero_results_dir is not None:
        base = Path(args.prospero_results_dir)
        for task in TASKS:
            OG_ROOTS[task] = base / f"{task}_cnn"
    written = []
    summaries = []
    for budget in BUDGETS:
        paths, summary = plot_budget(budget)
        written.extend(paths)
        summaries.extend(summary)
    summary_path = OUT / "plot_summary.txt"
    summary_path.write_text("Written plots:\n" + "\n".join(map(str, written)) + "\n\nSummary:\n" + "\n".join(summaries) + "\n")
    print(f"Wrote {len(written)} plot files")
    print(summary_path)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
