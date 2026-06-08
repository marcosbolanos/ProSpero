import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


POSITIVE = {"K", "R", "H"}
NEGATIVE = {"D", "E"}


def residue_charge_class(aa: str) -> str:
    if aa in POSITIVE:
        return "positive"
    if aa in NEGATIVE:
        return "negative"
    return "neutral"


def plot_task(task_result: dict, output_dir: Path, bins: int) -> dict:
    task = task_result["task"]
    rows = task_result["per_mutant_rows"]

    same_charge = []
    diff_charge = []
    for row in rows:
        wt = row["wt_residue"]
        mut = row["mutant_residue"]
        delta = float(row["oracle_delta_fitness"])
        if residue_charge_class(wt) == residue_charge_class(mut):
            same_charge.append(delta)
        else:
            diff_charge.append(delta)

    same = np.asarray(same_charge, dtype=np.float32)
    diff = np.asarray(diff_charge, dtype=np.float32)

    plt.figure(figsize=(8.8, 5.4))
    plt.hist(
        same,
        bins=bins,
        density=True,
        alpha=0.45,
        color="gray",
        edgecolor="none",
        label=f"Same charge (n={same.size})",
    )
    plt.hist(
        diff,
        bins=bins,
        density=True,
        alpha=0.45,
        color="red",
        edgecolor="none",
        label=f"Different charge (n={diff.size})",
    )
    plt.xlabel("Oracle delta fitness")
    plt.ylabel("Density")
    plt.title(f"{task}: Oracle delta fitness by charge-conserving vs charge-changing mutations")
    plt.grid(alpha=0.2)
    plt.legend(loc="best")
    plt.tight_layout()
    out_path = output_dir / f"charge_delta_hist_{task.lower()}.png"
    plt.savefig(out_path, dpi=220)
    plt.close()

    return {
        "task": task,
        "plot_path": str(out_path),
        "same_charge_count": int(same.size),
        "different_charge_count": int(diff.size),
        "same_charge_mean_delta_fitness": float(np.mean(same)) if same.size else None,
        "different_charge_mean_delta_fitness": float(np.mean(diff)) if diff.size else None,
        "same_charge_median_delta_fitness": float(np.median(same)) if same.size else None,
        "different_charge_median_delta_fitness": float(np.median(diff)) if diff.size else None,
    }


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot oracle single-mutant delta-fitness distributions for charge-conserving "
            "vs charge-changing mutations."
        )
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default="outputs/0423_experiments/single_mutant_energy_test_aav_lgk_one_hot_ridge_redo_full.json",
        help="Input JSON containing per_mutant_rows for each task.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/0423_experiments",
        help="Directory where plots and summary JSON will be written.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Number of histogram bins.",
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()

    input_path = Path(args.input_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text())
    task_results = payload["results"]

    summaries = []
    for task_result in task_results:
        if task_result["task"] not in {"AAV", "LGK"}:
            continue
        summaries.append(plot_task(task_result, output_dir, bins=args.bins))

    summary_payload = {
        "input_json": str(input_path),
        "definitions": {
            "charge_classes": {
                "positive": sorted(POSITIVE),
                "negative": sorted(NEGATIVE),
                "neutral": "all other amino acids",
            },
            "same_charge": "wt_residue and mutant_residue map to the same charge class",
            "different_charge": "wt_residue and mutant_residue map to different charge classes",
            "metric": "oracle_delta_fitness",
        },
        "summaries": summaries,
    }
    summary_path = output_dir / "charge_delta_hist_summary_aav_lgk.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    print(f"saved: {summary_path}")
    for item in summaries:
        print(f"saved: {item['plot_path']}")


if __name__ == "__main__":
    main()
