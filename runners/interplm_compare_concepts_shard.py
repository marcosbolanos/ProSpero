from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run InterPLM concept comparison for a single evaluation shard."
    )
    parser.add_argument("--sae-dir", type=Path, required=True)
    parser.add_argument("--aa-embds-dir", type=Path, required=True)
    parser.add_argument("--eval-set-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from interplm.analysis.concepts.compare_activations import analyze_concepts

    analyze_concepts(
        sae_dir=args.sae_dir,
        aa_embds_dir=args.aa_embds_dir,
        eval_set_dir=args.eval_set_dir,
        output_dir=args.output_dir,
        shard=args.shard,
    )


if __name__ == "__main__":
    main()
