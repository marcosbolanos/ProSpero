#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TASKS = ("AAV", "LGK", "GFP", "Pab1", "AMIE", "E4B", "TEM", "UBE2I")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create paper-facing tables from a reproduction root.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--budgets", nargs="+", type=int, default=[8, 128])
    return parser


def final_seed_score(path: Path) -> float | None:
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    rounds = [key for key in payload if isinstance(key, int)]
    if not rounds:
        return None
    entry = payload[max(rounds)]
    if not isinstance(entry, dict) or "Best score" not in entry:
        return None
    return float(entry["Best score"])


def summarize_paths(paths: list[Path]) -> tuple[float, float, int]:
    values = np.asarray(
        [score for path in paths if (score := final_seed_score(path)) is not None],
        dtype=float,
    )
    if values.size == 0:
        return np.nan, np.nan, 0
    sem = float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0
    return float(values.mean()), sem, int(values.size)


def result_paths(results: Path, method: str, task: str, budget: int) -> list[Path]:
    if method == "prospero_cnn":
        root = results / "prospero" / f"{task}_cnn" / f"n_samples_{budget}" / task
    else:
        root = results / f"0shotprot_{method}_n{budget}" / task
    return sorted(root.glob("seed_*.pkl"))


def optimization_summary(results: Path, tasks: list[str], budgets: list[int]) -> pd.DataFrame:
    methods = (
        ("prospero_cnn", "ProSpero"),
        ("prosst_prosst_finetuned", "0shotProt (ProSST)"),
        ("evodiff_evodiff_finetuned", "0shotProt (EvoDiff)"),
        ("prosst_prosst_finetuned_unrestricted", "0shotProt (ProSST, unrestricted)"),
        ("prosst_no_finetune", "0shotProt (ProSST, no fine-tuning)"),
    )
    rows = []
    for method, label in methods:
        for budget in budgets:
            for task in tasks:
                mean, sem, count = summarize_paths(result_paths(results, method, task, budget))
                if count:
                    rows.append(
                        {
                            "method": label,
                            "method_id": method,
                            "task": task,
                            "budget": budget,
                            "mean_final_max_fitness": mean,
                            "sem": sem,
                            "n_seeds": count,
                        }
                    )
    return pd.DataFrame(rows)


def latex_table(frame: pd.DataFrame, methods: list[str], budgets: list[int], tasks: list[str]) -> str:
    selected = frame[frame["method_id"].isin(methods) & frame["budget"].isin(budgets)]
    lines = [
        r"\begin{tabular}{@{}ll" + "r" * len(tasks) + r"@{}}",
        r"\toprule",
        "Method & $K$ & " + " & ".join(tasks) + r" \\",
        r"\midrule",
    ]
    for method in methods:
        for budget in budgets:
            row: pd.DataFrame = selected.loc[
                (selected["method_id"] == method) & (selected["budget"] == budget)
            ]
            if row.empty:
                continue
            label = str(row.iloc[0]["method"])
            cells = []
            for task in tasks:
                task_row: pd.DataFrame = row.loc[row["task"] == task]
                if task_row.empty:
                    cells.append("--")
                else:
                    value = task_row.iloc[0]
                    cells.append(
                        f"${float(value['mean_final_max_fitness']):.4f}"
                        f"\\pm{float(value['sem']):.4f}$"
                    )
            lines.append(f"{label} & {budget} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def alignment_latex(summary: pd.DataFrame, tasks: list[str]) -> str:
    model_order = [("evodiff", "EvoDiff"), ("esm", "ESM-2"), ("prosst", "ProSST")]
    pivot = summary.pivot_table(index="task", columns=["plm", "score_type"], values="spearman")
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{EvoDiff} & \multicolumn{2}{c}{ESM-2} & \multicolumn{2}{c}{ProSST} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Landscape & MMS & PLL & MMS & PLL & MMS & PLL \\",
        r"\midrule",
    ]
    for task in tasks:
        cells = []
        for model, _ in model_order:
            mms = float(pivot.loc[task, (model, "mms")])
            pll = float(pivot.loc[task, (model, "pll")])
            cells.extend(
                [
                    rf"\textbf{{{mms:.3f}}}" if mms >= pll else f"{mms:.3f}",
                    rf"\textbf{{{pll:.3f}}}" if pll > mms else f"{pll:.3f}",
                ]
            )
        lines.append(task + " & " + " & ".join(cells) + r" \\")
    average = summary.groupby(["plm", "score_type"])["spearman"].mean()
    cells = []
    for model, _ in model_order:
        mms = float(average.loc[(model, "mms")])
        pll = float(average.loc[(model, "pll")])
        cells.extend(
            [
                rf"\textbf{{{mms:.3f}}}" if mms >= pll else f"{mms:.3f}",
                rf"\textbf{{{pll:.3f}}}" if pll > mms else f"{pll:.3f}",
            ]
        )
    lines.extend([r"\midrule", r"\textbf{Avg.} & " + " & ".join(cells) + r" \\", r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = get_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = optimization_summary(args.results_root, list(args.tasks), list(args.budgets))
    frame.to_csv(args.output_dir / "optimization_final_max.csv", index=False)
    (args.output_dir / "optimization_main.tex").write_text(
        latex_table(
            frame,
            ["prospero_cnn", "prosst_prosst_finetuned", "evodiff_evodiff_finetuned"],
            list(args.budgets),
            list(args.tasks),
        ),
        encoding="utf-8",
    )
    (args.output_dir / "vocabulary_ablation.tex").write_text(
        latex_table(
            frame,
            ["prosst_prosst_finetuned", "prosst_prosst_finetuned_unrestricted"],
            list(args.budgets),
            list(args.tasks),
        ),
        encoding="utf-8",
    )
    (args.output_dir / "finetuning_ablation.tex").write_text(
        latex_table(
            frame,
            ["prosst_prosst_finetuned", "prosst_no_finetune"],
            list(args.budgets),
            list(args.tasks),
        ),
        encoding="utf-8",
    )

    alignment = args.results_root / "plm_mms_pll_alignment"
    for name in ("summary_long.csv", "spearman_pivot.csv", "run_config.json", "complete.json"):
        source = alignment / name
        if source.exists():
            shutil.copy2(source, args.output_dir / f"alignment_{name}")
    alignment_summary = alignment / "summary_long.csv"
    if alignment_summary.exists():
        (args.output_dir / "alignment_table.tex").write_text(
            alignment_latex(pd.read_csv(alignment_summary), list(args.tasks)),
            encoding="utf-8",
        )
    print(f"Wrote reproduction tables to {args.output_dir}")


if __name__ == "__main__":
    main()
