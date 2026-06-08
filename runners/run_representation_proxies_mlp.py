from __future__ import annotations

import argparse
import logging
from pathlib import Path

from prospero.experiments_config import WT_SEQUENCES
from prospero.probe_benchmark import BenchmarkRunner
from prospero.probe_benchmark.models import ProbeSpec


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


MLP_ARCHITECTURES: tuple[tuple[int, ...], ...] = (
    (4,),
    (8,),
    (12,),
    (16,),
    (4, 4),
    (8, 8),
    (12, 12),
    (16, 16),
    (4, 4, 4),
    (8, 8, 8),
    (12, 12, 12),
    (16, 16, 16),
)

DEFAULT_MLP_PROBE_SPECS: tuple[ProbeSpec, ...] = (
    ProbeSpec(
        name="mean_pool_mlp",
        feature_name="mean_pool",
        estimator_name="mlp",
        search_grid=MLP_ARCHITECTURES,
        description=(
            "Mean-pooled residues with MLP regression (ReLU, AdamW, L2=1e-4, "
            "early-stop patience=10 on held-out loss)."
        ),
    ),
    ProbeSpec(
        name="mean_max_mlp",
        feature_name="mean_max",
        estimator_name="mlp",
        search_grid=MLP_ARCHITECTURES,
        description=(
            "Mean+max pooled residues with MLP regression (ReLU, AdamW, L2=1e-4, "
            "early-stop patience=10 on held-out loss)."
        ),
    ),
    ProbeSpec(
        name="flatten_mlp",
        feature_name="flatten",
        estimator_name="mlp",
        search_grid=MLP_ARCHITECTURES,
        description=(
            "Flattened per-residue features with MLP regression (ReLU, AdamW, "
            "L2=1e-4, early-stop patience=10 on held-out loss)."
        ),
    ),
    ProbeSpec(
        name="flatten_pca_mlp",
        feature_name="flatten",
        estimator_name="mlp",
        search_grid=MLP_ARCHITECTURES,
        pca_components=128,
        description=(
            "Flattened per-residue features with PCA(128) + MLP regression "
            "(ReLU, AdamW, L2=1e-4, early-stop patience=10 on held-out loss)."
        ),
    ),
)


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_task_list(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark representation-quality proxies with MLP regressors."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where benchmark tables and plots will be written.",
    )
    parser.add_argument(
        "--tasks",
        type=parse_task_list,
        default=list(WT_SEQUENCES),
        help="Comma-separated list of tasks to benchmark.",
    )
    parser.add_argument(
        "--budgets",
        type=parse_int_list,
        default=[16, 32, 64, 128, 256, 512],
        help="Comma-separated training budgets sampled from the initial dataset.",
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=[1, 2, 3, 4, 5],
        help="Comma-separated random seeds used for subset sampling.",
    )
    parser.add_argument(
        "--esm-model-name",
        default="facebook/esm2_t6_8M_UR50D",
        help="Model namespace used by the on-disk cache.",
    )
    parser.add_argument(
        "--esm-max-length",
        type=int,
        default=None,
        help="Max length used by the cache namespace.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Override cache root. Defaults to the repository .cache/esm_embeddings path.",
    )
    parser.add_argument(
        "--allow-missing-cache",
        action="store_true",
        default=False,
        help="Skip tasks with incomplete cache coverage instead of failing the whole run.",
    )
    parser.add_argument(
        "--strict-cache-only",
        action="store_true",
        default=False,
        help="Fail on missing cache entries instead of recomputing uncached embeddings.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Batch size for on-the-fly ESM recomputation of missing embeddings.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure export DPI.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runner = BenchmarkRunner(
        tasks=args.tasks,
        budgets=args.budgets,
        seeds=args.seeds,
        model_name=args.esm_model_name,
        max_length=args.esm_max_length,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        probe_specs=DEFAULT_MLP_PROBE_SPECS,
        require_cache=args.strict_cache_only,
        compute_missing_embeddings=not args.strict_cache_only,
        embedding_batch_size=args.embedding_batch_size,
        skip_missing_tasks=args.allow_missing_cache,
        dpi=args.dpi,
    )
    summary = runner.run()
    logger.info(
        "Finished MLP representation proxy benchmark for %d tasks. Results in %s",
        len(summary["tasks"]),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
