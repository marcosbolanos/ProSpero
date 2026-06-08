from __future__ import annotations

import argparse
import logging
from pathlib import Path

from prospero.experiments_config import WT_SEQUENCES
from prospero.probe_benchmark.pipeline import BenchmarkRunner
from prospero.probe_benchmark.models import ProbeSpec


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


ONEHOT_COMBO_PROBE_SPECS: tuple[ProbeSpec, ...] = (
    ProbeSpec(
        name="one_hot_ridge",
        feature_name="one_hot_flat",
        estimator_name="ridge",
        search_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
        description="AA one-hot only with ridge regression.",
    ),
    ProbeSpec(
        name="one_hot_pca_ridge",
        feature_name="one_hot_flat",
        estimator_name="ridge",
        search_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
        pca_components=128,
        description="AA one-hot only with PCA + ridge regression.",
    ),
    ProbeSpec(
        name="mean_pool_plus_one_hot_ridge",
        feature_name="mean_pool_plus_one_hot",
        estimator_name="ridge",
        search_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
        description="Raw mean-pooled ESM + AA one-hot with ridge.",
    ),
    ProbeSpec(
        name="mean_pool_l2_plus_one_hot_ridge",
        feature_name="mean_pool_l2_plus_one_hot",
        estimator_name="ridge",
        search_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
        description="L2-normalized mean-pooled ESM + AA one-hot with ridge.",
    ),
    ProbeSpec(
        name="mean_pool_zscore_plus_one_hot_ridge",
        feature_name="mean_pool_zscore_plus_one_hot",
        estimator_name="ridge",
        search_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
        description="Z-scored mean-pooled ESM + AA one-hot with ridge.",
    ),
)


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_task_list(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark one-hot and mean-pooled+one-hot representation proxies."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        type=parse_task_list,
        default=list(WT_SEQUENCES),
        help="Comma-separated list of tasks.",
    )
    parser.add_argument(
        "--budgets",
        type=parse_int_list,
        default=[16, 32, 64, 128, 256, 512],
        help="Comma-separated training budgets.",
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=[1, 2, 3, 4, 5],
        help="Comma-separated random seeds.",
    )
    parser.add_argument("--esm-model-name", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--esm-max-length", type=int, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument(
        "--strict-cache-only",
        action="store_true",
        default=False,
        help="Fail if a needed embedding is missing from cache.",
    )
    parser.add_argument(
        "--allow-missing-cache",
        action="store_true",
        default=False,
        help="Skip tasks that fail to load embeddings.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Batch size for on-the-fly missing embedding recomputation.",
    )
    parser.add_argument("--dpi", type=int, default=200)
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
        probe_specs=ONEHOT_COMBO_PROBE_SPECS,
        require_cache=args.strict_cache_only,
        compute_missing_embeddings=not args.strict_cache_only,
        embedding_batch_size=args.embedding_batch_size,
        skip_missing_tasks=args.allow_missing_cache,
        dpi=args.dpi,
    )
    summary = runner.run()
    logger.info(
        "Finished onehot+combo benchmark for %d tasks. Results in %s",
        len(summary["tasks"]),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
