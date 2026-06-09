from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from prospero.plotting_style import COLORS, set_prospero_style


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "rl_vs_og" / "zero_shotprot_combined_budgets_grpo"
TASKS = ["AAV", "LGK", "GFP", "Pab1", "AMIE", "E4B", "TEM", "UBE2I"]
BUDGETS = [128, 8]

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
    Method("ProSST-GRPO", COLORS["prosst"], "s", "prosst", Path(".")),
    Method("EvoDiff", COLORS["evodiff"], "^", "evodiff", Path(".")),
    Method("ProSpero", COLORS["ink"], "o", "prospero", Path(".")),
]


def pastel(hex_color: str, amount: float = 0.62) -> str:
    hex_color = hex_color.lstrip("#")
    rgb = np.array([int(hex_color[i : i + 2], 16) for i in (0, 2, 4)], dtype=float)
    mixed = rgb * (1.0 - amount) + np.array([255.0, 255.0, 255.0]) * amount
    return "#" + "".join(f"{int(round(v)):02X}" for v in mixed)


def seed_paths(method: Method, task: str, budget: int) -> list[Path]:
    if method.root_kind == "prosst":
        return sorted((PROSST_ROOTS[budget] / task).glob("seed_*.pkl"))
    if method.root_kind == "evodiff":
        root, strategy = EVODIFF_ROOTS[task]
        return sorted((root / f"n_samples_{budget}" / strategy / task).glob("seed_*.pkl"))
    if method.root_kind == "prospero":
        return sorted((OG_ROOTS[task] / f"n_samples_{budget}" / task).glob("seed_*.pkl"))
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
    for method in METHODS:
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


def plot():
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
    for idx, task in enumerate(TASKS):
        ax = axes[idx // 4, idx % 4]
        for method in METHODS:
            for budget in BUDGETS:
                x, y, e, counts = aggregate(method, task, budget)
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
        ax.set_title(task, loc="left", pad=6)
        ax.set_xlim(1, 10)
        ax.set_xticks(range(1, 11))
        ax.grid(True, axis="y")
        if idx % 4 == 0:
            ax.set_ylabel("Mean max fitness")
        if idx // 4 == 1:
            ax.set_xlabel("Optimization round")

    handles, labels = legend_handles()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        columnspacing=1.8,
        handlelength=2.2,
        handletextpad=0.55,
    )
    fig.suptitle("Mean-max fitness trajectories by query budget", fontsize=46, fontweight="semibold", y=1.065)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.09, top=0.80, wspace=0.42, hspace=0.42)

    OUT.mkdir(parents=True, exist_ok=True)
    paths = [
        OUT / "zero_shotprot_grpo_combined_k8_k128_mean_max.png",
        OUT / "zero_shotprot_grpo_combined_k8_k128_mean_max.pdf",
        OUT / "zero_shotprot_grpo_combined_k8_k128_mean_max.svg",
    ]
    for path in paths:
        fig.savefig(path, dpi=320 if path.suffix == ".png" else None, bbox_inches="tight")
    plt.close(fig)
    (OUT / "plot_summary.txt").write_text(
        "Written plots:\n" + "\n".join(map(str, paths)) + "\n\nSummary:\n" + "\n".join(summary) + "\n",
        encoding="utf-8",
    )
    return paths


def main():
    paths = plot()
    print(f"Wrote {len(paths)} plot files")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

