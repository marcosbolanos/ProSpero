from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "-"


def _top_models_by_family(models: list[dict[str, Any]], per_family: int = 20) -> dict[str, list[dict[str, Any]]]:
    bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in models:
        bucket[row.get("family", "other")].append(row)

    for family, rows in bucket.items():
        rows.sort(key=lambda x: (-int(x.get("downloads", 0)), x.get("model_id", "")))
        bucket[family] = rows[:per_family]

    return dict(sorted(bucket.items(), key=lambda x: x[0]))


def _top_models_by_family_recent(models: list[dict[str, Any]], per_family: int = 20) -> dict[str, list[dict[str, Any]]]:
    bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in models:
        bucket[row.get("family", "other")].append(row)

    for family, rows in bucket.items():
        rows.sort(
            key=lambda x: (
                str(x.get("last_modified") or ""),
                int(x.get("downloads", 0)),
                x.get("model_id", ""),
            ),
            reverse=True,
        )
        bucket[family] = rows[:per_family]

    return dict(sorted(bucket.items(), key=lambda x: x[0]))


def _probe_summary(reports: list[dict[str, Any]]) -> tuple[int, int]:
    total = len(reports)
    ok = sum(1 for rep in reports if rep.get("status") == "ok")
    return ok, total


def _top_error_reasons(reports: list[dict[str, Any]], limit: int = 8) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for rep in reports:
        if rep.get("status") == "ok":
            continue
        reason = str(rep.get("error") or "unknown error")
        reason = reason.splitlines()[0].strip()
        counts[reason] = counts.get(reason, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]


def _row_for_probe(rep: dict[str, Any]) -> str:
    if rep.get("status") != "ok":
        err = str(rep.get("error", ""))
        err = err.replace("|", "/")[:120]
        return (
            f"| `{rep.get('model_id')}` | {rep.get('family')} | error | - | - | - | - | - | {err} |"
        )

    shape = rep.get("main_output_shape")
    shape_str = "x".join(str(x) for x in shape) if isinstance(shape, list) else "-"
    peak = rep.get("gpu_peak_memory_gb")
    peak_str = f"{peak:.3f}" if isinstance(peak, (float, int)) else "-"
    forward_s = rep.get("timings_sec", {}).get("forward")
    forward_str = f"{forward_s:.4f}" if isinstance(forward_s, (float, int)) else "-"
    return (
        f"| `{rep.get('model_id')}` | {rep.get('family')} | ok | `{rep.get('selected_tokenization_mode')}` | "
        f"{_fmt_int(rep.get('num_parameters'))} | {rep.get('hidden_size', '-')} | {shape_str} | "
        f"{peak_str} | {forward_str} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render PLM inventory + probe report to models.md")
    parser.add_argument(
        "--curated-json",
        type=Path,
        default=Path(__file__).resolve().parent / "hf_protein_models_inventory_curated.json",
    )
    parser.add_argument(
        "--probe-json",
        type=Path,
        default=Path(__file__).resolve().parent / "hf_plm_probe_results.json",
    )
    parser.add_argument(
        "--runner-json",
        type=Path,
        default=Path(__file__).resolve().parent / "hf_protein_models_runner_candidates.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(__file__).resolve().parent / "models.md",
    )
    parser.add_argument("--per-family", type=int, default=20)
    parser.add_argument(
        "--families",
        type=str,
        nargs="*",
        default=None,
        help="Optional family filter for report sections.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        choices=["downloads", "last_modified"],
        default="downloads",
        help="Ordering criterion for per-family model tables.",
    )
    args = parser.parse_args()

    curated = json.loads(args.curated_json.read_text(encoding="utf-8"))
    runner = json.loads(args.runner_json.read_text(encoding="utf-8"))
    probe = (
        json.loads(args.probe_json.read_text(encoding="utf-8"))
        if args.probe_json.exists()
        else {"reports": []}
    )

    models = curated.get("models", [])
    runner_models = runner.get("models", [])
    allowed_families = set(args.families) if args.families else None
    if allowed_families is not None:
        models = [row for row in models if row.get("family") in allowed_families]
        runner_models = [row for row in runner_models if row.get("family") in allowed_families]
    family_counts = curated.get("family_counts", {})
    if allowed_families is not None:
        family_counts = {k: v for k, v in family_counts.items() if k in allowed_families}
    top = (
        _top_models_by_family_recent(runner_models, per_family=args.per_family)
        if args.sort_by == "last_modified"
        else _top_models_by_family(runner_models, per_family=args.per_family)
    )
    reports = probe.get("reports", [])
    ok, total = _probe_summary(reports)
    top_errors = _top_error_reasons(reports)

    lines: list[str] = []
    lines.append("# Hugging Face Protein Language Models")
    lines.append("")
    lines.append(
        "This file is auto-generated from live Hugging Face Hub metadata and local compatibility probes "
        "for the universal runner (`src/prospero/plm/universal_hf.py`)."
    )
    lines.append("")
    lines.append(f"Generated at (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Curated inventory source: `{args.curated_json}`")
    lines.append(f"Runner-candidate source: `{args.runner_json}`")
    lines.append(f"Probe source: `{args.probe_json}`")
    lines.append("")
    lines.append("## Inventory Summary")
    lines.append("")
    lines.append(f"- Candidate count (pre-curation): {curated.get('candidate_count', 0):,}")
    lines.append(f"- Curated PLM-like count: {len(models):,}")
    lines.append(f"- Runner-compatible candidate count: {len(runner_models):,}")
    lines.append("- Family counts:")
    for family, count in sorted(family_counts.items()):
        lines.append(f"  - `{family}`: {count:,}")

    lines.append("")
    lines.append("## Exhaustive Lists")
    lines.append("")
    lines.append(
        "- Full exhaustive PLM-like inventory: `src/prospero/plm/hf_protein_models_inventory_curated.json`"
    )
    lines.append(
        "- Filtered runner-ready candidate list: `src/prospero/plm/hf_protein_models_runner_candidates.json`"
    )
    lines.append("")
    if args.sort_by == "last_modified":
        lines.append("## Runner Candidates (Most Recent Per Family)")
    else:
        lines.append("## Runner Candidates (Top by Downloads Per Family)")
    lines.append("")
    for family, rows in top.items():
        lines.append(f"### {family}")
        lines.append("")
        lines.append("| Model | Downloads | Pipeline | Library | Last Modified |")
        lines.append("|---|---:|---|---|---|")
        for row in rows:
            lines.append(
                f"| `{row['model_id']}` | {_fmt_int(row.get('downloads'))} | "
                f"{row.get('pipeline_tag') or '-'} | {row.get('library_name') or '-'} | "
                f"{row.get('last_modified') or '-'} |"
            )
        lines.append("")

    lines.append("## Manual Pull + Probe Results")
    lines.append("")
    lines.append(
        "Each probed model was loaded from Hugging Face via `AutoTokenizer` + `AutoModel`, tokenized in "
        "`auto` mode (raw/spaced/split fallback), and run for a single forward pass to verify per-token outputs."
    )
    lines.append("")
    lines.append(f"Probe success: {ok}/{total}")
    lines.append("")
    lines.append("| Model | Family | Status | Tokenization Mode | Params | Hidden | Main Output Shape | GPU Peak GB | Forward s |")
    lines.append("|---|---|---|---|---:|---:|---|---:|---:|")
    for rep in reports:
        lines.append(_row_for_probe(rep))

    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- The Hub is dynamic; rerun inventory/probe scripts to refresh this file.")
    lines.append(
        "- Some repos are adapters, quantized exports, or task heads; they can appear in search and may be filtered heuristically."
    )
    lines.append("- Some models require `trust_remote_code` and custom forward/output handling.")
    if top_errors:
        lines.append("- Most common probe failures in this run:")
        for reason, count in top_errors:
            lines.append(f"  - ({count}) {reason}")
    lines.append(
        "- Per-token shape compatibility here means accessible hidden states; it does not imply biological benchmarking quality."
    )

    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote markdown report: {args.output_md}")


if __name__ == "__main__":
    main()
