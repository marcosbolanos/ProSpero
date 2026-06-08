from __future__ import annotations

import argparse
import json
from pathlib import Path

from prospero.probe_benchmark.plotting import save_plots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate plots from benchmark_results.json."
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        required=True,
        help="Path to benchmark_results.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional plot output directory. Defaults to sibling plots/ next to the JSON file.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with open(args.results_json, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.results_json.parent / "plots"

    save_plots(summary, output_dir, dpi=args.dpi)


if __name__ == "__main__":
    main()
