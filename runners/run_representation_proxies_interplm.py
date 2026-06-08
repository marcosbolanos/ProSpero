from __future__ import annotations

import argparse
import logging
from pathlib import Path

from prospero.experiments_config import WT_SEQUENCES
from prospero.probe_benchmark.interplm import DEFAULT_INTERPLM_REPO_ID
from prospero.probe_benchmark.interplm_pipeline import (
    DEFAULT_ELASTICNET_L1_RATIO,
    DEFAULT_INTERPLM_LAYERS,
    DEFAULT_L1_ALPHA_SCALE_GRID,
    InterPLMBenchmarkRunner,
)


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_task_list(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one-hot and InterPLM mean-pooled SAE features with linear probes."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--estimator",
        default="ridge",
        choices=("ridge", "lasso", "elasticnet"),
        help="Linear probe family to benchmark.",
    )
    parser.add_argument(
        "--alpha-scale-grid",
        type=parse_float_list,
        default=list(DEFAULT_L1_ALPHA_SCALE_GRID),
        help=(
            "Comma-separated alpha/alpha_max values used for lasso or elasticnet. "
            "Ignored for ridge."
        ),
    )
    parser.add_argument(
        "--elasticnet-l1-ratio",
        type=float,
        default=DEFAULT_ELASTICNET_L1_RATIO,
        help="ElasticNet l1_ratio value when --estimator elasticnet.",
    )
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
        "--interplm-repo-id",
        default=DEFAULT_INTERPLM_REPO_ID,
        help="Hugging Face repo containing the InterPLM checkpoints.",
    )
    parser.add_argument(
        "--interplm-layers",
        type=parse_int_list,
        default=list(DEFAULT_INTERPLM_LAYERS),
        help="Comma-separated InterPLM/ESM layer ids to benchmark.",
    )
    parser.add_argument(
        "--interplm-unnormalized",
        action="store_true",
        default=False,
        help="Use ae_unnormalized.pt instead of ae_normalized.pt.",
    )
    parser.add_argument(
        "--strict-cache-only",
        action="store_true",
        default=False,
        help="Fail if a needed cached InterPLM feature is missing.",
    )
    parser.add_argument(
        "--allow-missing-cache",
        action="store_true",
        default=False,
        help="Skip tasks that fail to load features.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Batch size for on-the-fly ESM forward passes.",
    )
    parser.add_argument(
        "--sae-token-chunk-size",
        type=int,
        default=1024,
        help="Token chunk size used when encoding SAE activations to limit memory.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Execution device for ESM and SAE feature extraction.",
    )
    parser.add_argument(
        "--cache-source-embeddings",
        action="store_true",
        default=False,
        help="Persist per-layer ESM residue embeddings for later SAE iterations.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runner = InterPLMBenchmarkRunner(
        tasks=args.tasks,
        budgets=args.budgets,
        seeds=args.seeds,
        model_name=args.esm_model_name,
        max_length=args.esm_max_length,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        interplm_repo_id=args.interplm_repo_id,
        interplm_layers=tuple(args.interplm_layers),
        interplm_normalized=not args.interplm_unnormalized,
        require_cache=args.strict_cache_only,
        compute_missing_features=not args.strict_cache_only,
        embedding_batch_size=args.embedding_batch_size,
        sae_token_chunk_size=args.sae_token_chunk_size,
        skip_missing_tasks=args.allow_missing_cache,
        dpi=args.dpi,
        device=args.device,
        cache_source_embeddings=args.cache_source_embeddings,
        estimator_name=args.estimator,
        search_grid=tuple(args.alpha_scale_grid) if args.estimator != "ridge" else None,
        elasticnet_l1_ratio=args.elasticnet_l1_ratio,
    )
    summary = runner.run()
    logger.info(
        "Finished InterPLM representation proxy benchmark for %d tasks. Results in %s",
        len(summary["tasks"]),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
