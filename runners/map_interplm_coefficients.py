from __future__ import annotations

import argparse
import json
from pathlib import Path

from prospero.probe_benchmark.interplm_annotations import (
    build_annotation_report,
    parse_csv_list,
    parse_int_csv_list,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Map InterPLM coefficient artifacts to InterPLM concept annotations."
        )
    )
    parser.add_argument(
        "--coefficient-dir",
        type=Path,
        required=True,
        help="Directory containing coefficient_index.csv and coefficient npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the annotated coefficient reports. Defaults to <coefficient-dir>/annotation_maps.",
    )
    parser.add_argument(
        "--dashboard-cache-dir",
        type=Path,
        default=None,
        help=(
            "InterPLM dashboard cache root. If set, annotations are read from "
            "<dashboard-cache-dir>/layer_{layer}/Sig_concepts_per_feature.csv."
        ),
    )
    parser.add_argument(
        "--annotation-file-template",
        default=None,
        help=(
            "Explicit annotation file template, e.g. "
            "'/path/to/cache/layer_{layer}/Sig_concepts_per_feature.csv'. "
            "Overrides --dashboard-cache-dir."
        ),
    )
    parser.add_argument(
        "--annotation-download-template",
        default=None,
        help=(
            "Optional URL template for downloading missing per-layer annotation files, "
            "e.g. 'https://host/cache/layer_{layer}/Sig_concepts_per_feature.csv'."
        ),
    )
    parser.add_argument(
        "--annotation-download-cache-dir",
        type=Path,
        default=None,
        help=(
            "Cache directory for downloaded annotation files. Defaults to "
            "<output-dir>/downloaded_annotations when --annotation-download-template is used."
        ),
    )
    parser.add_argument("--tasks", default=None, help="Comma-separated task filter.")
    parser.add_argument("--configs", default=None, help="Comma-separated config filter.")
    parser.add_argument("--budgets", default=None, help="Comma-separated budget filter.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed filter.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Keep up to this many top positive and top negative features per run.",
    )
    parser.add_argument(
        "--coefficient-threshold",
        type=float,
        default=0.0,
        help="Minimum absolute coefficient value to include.",
    )
    parser.add_argument(
        "--max-concepts-per-feature",
        type=int,
        default=3,
        help="Maximum number of concept annotations to attach to each feature.",
    )
    parser.add_argument(
        "--include-all-nonzero",
        action="store_true",
        default=False,
        help="Export every feature above --coefficient-threshold instead of only top-k positive/negative.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    annotation_template = args.annotation_file_template
    if annotation_template is None and args.dashboard_cache_dir is not None:
        annotation_template = str(args.dashboard_cache_dir / "layer_{layer}" / "Sig_concepts_per_feature.csv")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.coefficient_dir / "annotation_maps"
    annotation_download_cache_dir = args.annotation_download_cache_dir
    if annotation_download_cache_dir is None and args.annotation_download_template is not None:
        annotation_download_cache_dir = output_dir / "downloaded_annotations"

    manifest = build_annotation_report(
        coefficient_dir=args.coefficient_dir,
        output_dir=output_dir,
        tasks=parse_csv_list(args.tasks),
        configs=parse_csv_list(args.configs),
        budgets=parse_int_csv_list(args.budgets),
        seeds=parse_int_csv_list(args.seeds),
        annotation_template=annotation_template,
        annotation_download_template=args.annotation_download_template,
        annotation_download_cache_dir=annotation_download_cache_dir,
        top_k=int(args.top_k),
        coefficient_threshold=float(args.coefficient_threshold),
        max_concepts_per_feature=int(args.max_concepts_per_feature),
        include_all_nonzero=bool(args.include_all_nonzero),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
