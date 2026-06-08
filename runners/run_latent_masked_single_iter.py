from __future__ import annotations

import argparse

from prospero.latent_masked_search import run_latent_masked_single_iter
from prospero.runners.run_protein import get_parser as get_base_parser


def get_parser() -> argparse.ArgumentParser:
    parser = get_base_parser()
    parser.description = (
        "Single-iteration latent masked search: targeted masking + latent steering"
    )
    parser.add_argument("--top-features", type=int, default=3)
    parser.add_argument(
        "--top-feature-counts",
        type=int,
        nargs="+",
        default=[1, 3, 8, 15, 30],
    )
    parser.add_argument("--steering-layer", type=int, default=2)
    parser.add_argument(
        "--steering-scalars",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.2, 0.7],
    )
    parser.add_argument("--combo-chunk-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=256)
    parser.set_defaults(n_iters=1, surrogate_arch="interplm_mean_pool_ridge")
    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    output_path = run_latent_masked_single_iter(args)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
