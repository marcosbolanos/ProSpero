from __future__ import annotations

import argparse
from pathlib import Path

from prospero.reproduction.types import RuntimeOptions


def parse_runtime_options(default_stage_names: tuple[str, ...]) -> RuntimeOptions:
    parser = argparse.ArgumentParser(
        description="Reproduce paper-relevant ProSpero/0shotProt runs and plots into outputs/reproduction/<timestamp>.",
    )
    parser.add_argument("--output-root", default="outputs/reproduction")
    parser.add_argument(
        "--timestamp", default=None, help="Override timestamp folder name."
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None, help="Optional task filter."
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None, help="Optional seed filter."
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=None,
        help="Optional query-budget filter.",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value, preferably a GPU UUID.",
    )
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument(
        "--no-skip-existing", dest="skip_existing", action="store_false"
    )
    parser.add_argument("--plots-only", action="store_true", default=False)
    parser.add_argument("--no-plots", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=None,
        choices=default_stage_names,
        help="Optional stage filter.",
    )
    args = parser.parse_args()
    return RuntimeOptions(
        output_root=Path(args.output_root),
        timestamp=args.timestamp,
        gpu=args.gpu,
        dry_run=args.dry_run,
        plots_only=args.plots_only,
        no_plots=args.no_plots,
        skip_existing=args.skip_existing,
        selected_stages=set(args.stages) if args.stages is not None else None,
        task_filter=tuple(args.tasks) if args.tasks is not None else None,
        seed_filter=tuple(args.seeds) if args.seeds is not None else None,
        budget_filter=tuple(args.budgets) if args.budgets is not None else None,
        extra_manifest={"cli_args": vars(args)},
    )
