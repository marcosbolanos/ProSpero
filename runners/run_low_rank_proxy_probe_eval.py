from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from prospero.dataset import RegressionDataset
from prospero.experiments_config import WT_SEQUENCES
from prospero.probe_benchmark.metrics import regression_metrics
from prospero.runners.run_protein import FROZEN_ESM_SURROGATE_ARCHS, get_parser as get_base_parser
from prospero.surrogate import (
    Ensemble,
    build_surrogate_model,
    normalize_sequences,
    prepare_shared_esm_components,
)
from prospero.utils import set_seed


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _prepare_proxy(args, sequence_length: int, dataset: RegressionDataset) -> Ensemble:
    if getattr(args, "disable_esm_cache", False):
        args.cache_allowed_sequences = set()
        args.cache_allowed_sequences_ordered = []
        args.dataset_cache_task = None
    else:
        ordered = _dedupe_preserving_order(
            normalize_sequences(list(dataset.train) + list(dataset.valid))
        )
        args.cache_allowed_sequences = set(ordered)
        args.cache_allowed_sequences_ordered = ordered
        args.dataset_cache_task = args.task

    shared_esm_components = None
    if args.surrogate_arch in FROZEN_ESM_SURROGATE_ARCHS:
        shared_esm_components = prepare_shared_esm_components(args)

    proxy = Ensemble(
        [
            build_surrogate_model(
                sequence_length,
                args,
                shared_esm_components=shared_esm_components,
            )
            for _ in range(args.ensemble_size)
        ]
    )
    proxy.train(dataset)
    return proxy


def _evaluate_proxy(proxy: Ensemble, dataset: RegressionDataset) -> dict[str, dict[str, float]]:
    train_sequences = normalize_sequences(dataset.train.tolist())
    valid_sequences = normalize_sequences(dataset.valid.tolist())
    train_true = np.asarray(dataset.train_scores, dtype=np.float64)
    valid_true = np.asarray(dataset.valid_scores, dtype=np.float64)
    train_pred = proxy.get_scores(train_sequences).detach().cpu().numpy().astype(np.float64)
    valid_pred = proxy.get_scores(valid_sequences).detach().cpu().numpy().astype(np.float64)
    return {
        "train": regression_metrics(train_true, train_pred),
        "valid": regression_metrics(valid_true, valid_pred),
    }


def _run_task(args, task: str, low_rank_dims: list[int]) -> dict[str, object]:
    dataset = RegressionDataset(task)
    results: list[dict[str, object]] = []
    seq_length = len(WT_SEQUENCES[task])

    model_specs = [("interplm_mean_pool_ridge", None)] + [
        ("interplm_low_rank_positional", int(rank)) for rank in low_rank_dims
    ]
    for surrogate_arch, rank in model_specs:
        args.task = task
        args.surrogate_arch = surrogate_arch
        if rank is not None:
            args.low_rank_positional_rank = int(rank)

        t0 = time.perf_counter()
        proxy = _prepare_proxy(args, seq_length, dataset)
        train_seconds = float(time.perf_counter() - t0)

        t1 = time.perf_counter()
        metrics = _evaluate_proxy(proxy, dataset)
        eval_seconds = float(time.perf_counter() - t1)
        results.append(
            {
                "surrogate_arch": surrogate_arch,
                "low_rank_positional_rank": rank,
                "train_seconds": train_seconds,
                "eval_seconds": eval_seconds,
                "metrics": metrics,
            }
        )

    return {
        "task": task,
        "train_size": int(len(dataset.train)),
        "valid_size": int(len(dataset.valid)),
        "sequence_length": int(seq_length),
        "models": results,
    }


def get_parser() -> argparse.ArgumentParser:
    parser = get_base_parser()
    parser.description = (
        "Run proxy/probing evaluation for low-rank positional surrogate vs baseline."
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["AAV", "LGK"],
        help="Tasks to evaluate.",
    )
    parser.add_argument(
        "--low-rank-dims",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64],
        help="Rank values for interplm_low_rank_positional.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/0423_experiments/low_rank_proxy_probe_aav_lgk.json",
    )
    parser.set_defaults(
        ensemble_size=1,
        surrogate_arch="interplm_low_rank_positional",
        results_dirpath="outputs/0423_experiments",
        task="AAV",
    )
    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed, args.full_deterministic)

    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    tasks = [str(task) for task in args.tasks]
    low_rank_dims = [int(x) for x in args.low_rank_dims]
    task_results = [_run_task(args, task, low_rank_dims) for task in tasks]
    output = {
        "run_started_at_utc": started,
        "run_finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_runtime_seconds": float(time.perf_counter() - start),
        "seed": int(args.seed),
        "tasks": tasks,
        "low_rank_dims": low_rank_dims,
        "settings": {
            "esm_model_name": str(args.esm_model_name),
            "interplm_layer": int(args.interplm_layer),
            "low_rank_positional_l2": float(args.low_rank_positional_l2),
            "low_rank_positional_lr": float(args.low_rank_positional_lr),
        },
        "results": task_results,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
