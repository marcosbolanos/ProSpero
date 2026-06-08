from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed selected InterPLM annotation shards."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, nargs="+", required=True)
    parser.add_argument("--embedder-type", default="esm")
    parser.add_argument("--model-name", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-column", default="Sequence")
    return parser.parse_args()


def _load_sequences(shard_file: Path, sequence_column: str) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(shard_file, sep="\t")

    seq_col = None
    for col in df.columns:
        if col.lower() == sequence_column.lower():
            seq_col = col
            break

    if seq_col is None:
        raise ValueError(
            f"Column '{sequence_column}' not found in {shard_file}. "
            f"Available columns: {list(df.columns)}"
        )

    return df, seq_col


def main() -> None:
    args = parse_args()

    # Imported lazily so this wrapper can live in this repo without depending on
    # InterPLM at import time.
    from interplm.embedders import get_embedder

    args.output_dir.mkdir(parents=True, exist_ok=True)

    embedder = get_embedder(args.embedder_type, model_name=args.model_name)
    shard_ids = sorted(set(args.shards))

    for shard in tqdm(shard_ids, desc="Embedding annotation shards"):
        shard_file = args.input_dir / f"shard_{shard}" / "protein_data.tsv"
        if not shard_file.exists():
            raise FileNotFoundError(f"Missing annotation shard: {shard_file}")

        df, seq_col = _load_sequences(shard_file, args.sequence_column)
        sequences = df[seq_col].tolist()
        embeddings_dict = embedder.extract_embeddings_with_boundaries(
            sequences,
            layer=args.layer,
            batch_size=args.batch_size,
        )

        output_shard_dir = args.output_dir / f"shard_{shard}"
        output_shard_dir.mkdir(parents=True, exist_ok=True)

        protein_ids = (
            df["Entry"].tolist() if "Entry" in df.columns else list(range(len(sequences)))
        )
        torch.save(
            {
                "embeddings": embeddings_dict["embeddings"],
                "boundaries": embeddings_dict["boundaries"],
                "protein_ids": protein_ids,
            },
            output_shard_dir / "embeddings.pt",
        )


if __name__ == "__main__":
    main()
