from __future__ import annotations

import argparse
import logging
from pathlib import Path

from prospero.experiments_config import WT_SEQUENCES
from prospero.probe_benchmark.models import ProbeSpec
from prospero.probe_benchmark.pipeline import BenchmarkRunner


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


FLATTEN_PCA_COMBO_PROBE_SPECS: tuple[ProbeSpec, ...] = (
    ProbeSpec(
        name="flatten_plus_one_hot_ridge",
        feature_name="flatten_plus_one_hot",
        estimator_name="ridge",
        search_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
        description="Flattened per-residue ESM + AA one-hot with ridge.",
    ),
    ProbeSpec(
        name="flatten_pca128_ridge",
        feature_name="flatten",
        estimator_name="ridge",
        pca_components=128,
        search_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
        description="Flattened per-residue ESM, PCA rank=128, ridge.",
    ),
    ProbeSpec(
        name="flatten_plus_one_hot_pca128_ridge",
        feature_name="flatten_plus_one_hot",
        estimator_name="ridge",
        pca_components=128,
        search_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
        description="Flattened ESM + one-hot, PCA rank=128, ridge.",
    ),
    ProbeSpec(
        name="flatten_pca64_ridge",
        feature_name="flatten",
        estimator_name="ridge",
        pca_components=64,
        search_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
        description="Flattened per-residue ESM, PCA rank=64, ridge.",
    ),
    ProbeSpec(
        name="flatten_plus_one_hot_pca64_ridge",
        feature_name="flatten_plus_one_hot",
        estimator_name="ridge",
        pca_components=64,
        search_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
        description="Flattened ESM + one-hot, PCA rank=64, ridge.",
    ),
    ProbeSpec(
        name="flatten_pca32_ridge",
        feature_name="flatten",
        estimator_name="ridge",
        pca_components=32,
        search_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
        description="Flattened per-residue ESM, PCA rank=32, ridge.",
    ),
    ProbeSpec(
        name="flatten_plus_one_hot_pca32_ridge",
        feature_name="flatten_plus_one_hot",
        estimator_name="ridge",
        pca_components=32,
        search_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
        description="Flattened ESM + one-hot, PCA rank=32, ridge.",
    ),
)


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_task_list(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark flatten(+one-hot) and PCA-rank ablations."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        type=parse_task_list,
        default=list(WT_SEQUENCES),
    )
    parser.add_argument(
        "--budgets",
        type=parse_int_list,
        default=[16, 32, 64, 128, 256, 512],
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=[1, 2, 3, 4, 5],
    )
    parser.add_argument("--esm-model-name", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--esm-max-length", type=int, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--strict-cache-only", action="store_true", default=False)
    parser.add_argument("--allow-missing-cache", action="store_true", default=False)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
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
        probe_specs=FLATTEN_PCA_COMBO_PROBE_SPECS,
        require_cache=args.strict_cache_only,
        compute_missing_embeddings=not args.strict_cache_only,
        embedding_batch_size=args.embedding_batch_size,
        skip_missing_tasks=args.allow_missing_cache,
        dpi=args.dpi,
    )
    summary = runner.run()
    logger.info(
        "Finished flatten+pca combo benchmark for %d tasks. Results in %s",
        len(summary["tasks"]),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
