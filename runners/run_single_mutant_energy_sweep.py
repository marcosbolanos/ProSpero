from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class SweepModel:
    model_id: str
    family: str
    last_modified: str | None


def _parse_size_million(model_id: str, family: str) -> float:
    m = re.search(r"(?:^|[_-])t\d+_(\d+)M", model_id)
    if m:
        return float(m.group(1))
    m = re.search(r"(?:^|[_-])(\d+)m(?:$|[_-])", model_id, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    if family == "esm3":
        return 1400.0
    m = re.search(r"(?:^|[_-])(\d+)B(?:$|[_-])", model_id, flags=re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1000.0
    return 1e9


def _load_models(inventory_json: Path) -> list[SweepModel]:
    payload = json.loads(inventory_json.read_text(encoding="utf-8"))
    allowed = {"esm2", "esm3", "esmc"}
    models: list[SweepModel] = []
    seen: set[str] = set()
    for row in payload.get("models", []):
        family = str(row.get("family", ""))
        model_id = str(row.get("model_id", ""))
        if family not in allowed:
            continue
        if model_id in seen:
            continue
        seen.add(model_id)
        models.append(
            SweepModel(
                model_id=model_id,
                family=family,
                last_modified=row.get("last_modified"),
            )
        )

    models.sort(
        key=lambda m: (
            _parse_size_million(m.model_id, m.family),
            m.family,
            m.model_id,
        )
    )
    return models


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "runs": {},
            "completed": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _save_checkpoint(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _model_slug(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def _run_model(
    model: SweepModel,
    *,
    tasks: list[str],
    output_dir: Path,
    seed: int,
    surrogate_arch: str,
    ensemble_size: int,
    extra_args: list[str],
) -> tuple[int, str]:
    slug = _model_slug(model.model_id)
    model_dir = output_dir / slug
    model_dir.mkdir(parents=True, exist_ok=True)

    compact_json = model_dir / "single_mutant_energy_compact.json"
    full_json = model_dir / "single_mutant_energy_full.json"
    oracle_cache_json = output_dir.parent / "oracle_single_mutant_cache.json"

    cmd = [
        sys.executable,
        "-m",
        "prospero.runners.run_single_mutant_energy_test",
        "--tasks",
        *tasks,
        "--seed",
        str(seed),
        "--surrogate_arch",
        surrogate_arch,
        "--ensemble_size",
        str(ensemble_size),
        "--esm_model_name",
        model.model_id,
        "--output-json",
        str(compact_json),
        "--output-json-full",
        str(full_json),
        "--oracle-cache-json",
        str(oracle_cache_json),
        "--results_dirpath",
        str(model_dir),
    ]
    cmd.extend(extra_args)

    log_path = model_dir / "run.log"
    started = datetime.now(timezone.utc).isoformat()
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{started}] CMD: {' '.join(cmd)}\n")
        log_file.flush()
        proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    return proc.returncode, str(log_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run single-mutation delta test sweep across ESM2+ models with "
            "incremental checkpointing and resume support."
        )
    )
    parser.add_argument(
        "--inventory-json",
        type=Path,
        default=Path("src/prospero/plm/hf_esm_evo_latest_official.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/0423_experiments/esm2plus_single_mutant_sweep"),
    )
    parser.add_argument(
        "--checkpoint-json",
        type=Path,
        default=Path("outputs/0423_experiments/esm2plus_single_mutant_sweep/checkpoint.json"),
    )
    parser.add_argument("--tasks", nargs="+", default=["AAV", "LGK"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--surrogate-arch", type=str, default="frozen_esm_flat_ridge_no_onehot")
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument(
        "--max-models",
        type=int,
        default=None,
        help="Optional cap (after size ordering).",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra arg passed through to run_single_mutant_energy_test (repeatable).",
    )
    args = parser.parse_args()

    models = _load_models(args.inventory_json)
    if args.max_models is not None:
        models = models[: max(0, args.max_models)]

    checkpoint = _load_checkpoint(args.checkpoint_json)
    runs = checkpoint.setdefault("runs", {})
    completed = set(checkpoint.setdefault("completed", []))

    print(f"models_to_run={len(models)}")
    for idx, model in enumerate(models, start=1):
        if model.model_id in completed:
            print(f"[{idx}/{len(models)}] skip {model.model_id} (already completed)")
            continue

        print(f"[{idx}/{len(models)}] run {model.model_id}")
        run_started = datetime.now(timezone.utc).isoformat()
        runs[model.model_id] = {
            "family": model.family,
            "last_modified": model.last_modified,
            "status": "running",
            "started_at_utc": run_started,
        }
        _save_checkpoint(args.checkpoint_json, checkpoint)

        returncode, log_path = _run_model(
            model,
            tasks=list(args.tasks),
            output_dir=args.output_dir,
            seed=args.seed,
            surrogate_arch=args.surrogate_arch,
            ensemble_size=args.ensemble_size,
            extra_args=list(args.extra_arg),
        )

        finished = datetime.now(timezone.utc).isoformat()
        status = "success" if returncode == 0 else "failed"
        runs[model.model_id] = {
            **runs[model.model_id],
            "status": status,
            "returncode": int(returncode),
            "finished_at_utc": finished,
            "log_path": log_path,
        }
        if status == "success":
            completed.add(model.model_id)
            checkpoint["completed"] = sorted(completed)
        _save_checkpoint(args.checkpoint_json, checkpoint)

    checkpoint["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _save_checkpoint(args.checkpoint_json, checkpoint)
    print(f"checkpoint: {args.checkpoint_json}")


if __name__ == "__main__":
    main()
