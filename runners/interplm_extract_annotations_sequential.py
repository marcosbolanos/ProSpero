from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run InterPLM UniProt annotation extraction sequentially."
    )
    parser.add_argument("--input_uniprot_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--n_shards", type=int, default=5)
    parser.add_argument("--min_required_instances", type=int, default=100)
    parser.add_argument("--min_protein_length", type=int, default=1022)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from interplm.analysis.concepts.extract_annotations import (
        binary_meta_cols,
        convert_shard_to_amino_acid_features,
        enumerate_protein_subcategories,
        paired_binary_cols,
        preprocess_proteins,
        shard_protein_data,
    )
    import pandas as pd

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_uniprot_path, sep="\t")
    df = preprocess_proteins(df, args.min_protein_length)
    shard_protein_data(df, args.output_dir, args.n_shards)
    categorical_options = enumerate_protein_subcategories(
        df, args.min_required_instances
    )

    for shard in range(args.n_shards):
        convert_shard_to_amino_acid_features(
            shard_id=shard,
            input_path=args.output_dir / f"shard_{shard}" / "protein_data.tsv",
            output_dir=args.output_dir,
            categorical_options=categorical_options,
            binary_cols=binary_meta_cols,
            interaction_cols=paired_binary_cols,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
