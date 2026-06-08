from __future__ import annotations

import argparse
import logging
from pathlib import Path

from prospero.experiments_config import WT_SEQUENCES
from prospero.probe_benchmark.models import ProbeSpec
from prospero.probe_benchmark.pipeline import BenchmarkRunner
from prospero.runners.run_representation_proxies_mlp import MLP_ARCHITECTURES


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def _arch_label(arch: tuple[int, ...]) -> str:
    return "x".join(str(width) for width in arch)


FLATTEN_MLP_ARCH_SWEEP_PROBE_SPECS: tuple[ProbeSpec, ...] = tuple(
    ProbeSpec(
        name=f"flatten_mlp_arch_{_arch_label(arch)}",
        feature_name="flatten",
        estimator_name="mlp",
        # Single-point grid to force fixed architecture evaluation per config.
        search_grid=(arch,),
        description=(
            "Flattened per-residue features with fixed MLP architecture "
            f"{arch} (ReLU, AdamW, L2=1e-4, early-stop patience=10)."
        ),
        metadata={"architecture": str(arch)},
    )
    for arch in MLP_ARCHITECTURES
)


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_task_list(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark flatten_mlp architecture sweep with fixed architectures, "
            "using the same task/budget/seed sampling points as representation-proxy runs."
        )
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
        probe_specs=FLATTEN_MLP_ARCH_SWEEP_PROBE_SPECS,
        require_cache=args.strict_cache_only,
        compute_missing_embeddings=not args.strict_cache_only,
        embedding_batch_size=args.embedding_batch_size,
        skip_missing_tasks=args.allow_missing_cache,
        dpi=args.dpi,
    )
    summary = runner.run()
    logger.info(
        "Finished flatten MLP architecture sweep for %d tasks. Results in %s",
        len(summary["tasks"]),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
