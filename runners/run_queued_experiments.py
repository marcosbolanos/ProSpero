"""Queue multiple variable-k experiments sequentially with robust logging.

This runner only orchestrates jobs. It does not add new surrogate architectures.
If a configured architecture is not yet supported by ``run_variable_k.py``, that
job will fail fast and the queue will stop.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ALL_TASKS = [
    "AAV",
    "E4B",
    "AMIE",
    "LGK",
    "Pab1",
    "TEM",
    "UBE2I",
    "GFP",
    "D_SHIFT",
    "D_SHIFT_SMALL",
    "D_SHIFT_HARD",
]


@dataclass(frozen=True)
class QueuedJob:
    name: str
    task: str
    surrogate_arch: str
    alphabet: str = "CHARGE"
    extra_args: tuple[str, ...] = ()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_default_jobs() -> list[QueuedJob]:
    jobs: list[QueuedJob] = []

    def add_phase(phase_label: str, alphabet: str) -> None:
        # Group 1: one-hot-only ridge for all tasks (no ESM embeddings).
        # Expected arch placeholder to be implemented in main training loop.
        one_hot_arch = "one_hot_ridge"
        for task in ALL_TASKS:
            jobs.append(
                QueuedJob(
                    name=f"{phase_label}_onehot_ridge_{task.lower()}",
                    task=task,
                    surrogate_arch=one_hot_arch,
                    alphabet=alphabet,
                )
            )

        # Group 2: flattened ESM ridge without one-hot for AAV and LGK.
        # Expected arch placeholder to be implemented in main training loop.
        flat_no_one_hot_arch = "frozen_esm_flat_ridge_no_onehot"
        for task in ("AAV", "LGK"):
            jobs.append(
                QueuedJob(
                    name=f"{phase_label}_flat_esm_ridge_no_onehot_{task.lower()}",
                    task=task,
                    surrogate_arch=flat_no_one_hot_arch,
                    alphabet=alphabet,
                )
            )

    # Phase 1: charge-constrained substitutions (original ProSpero setting).
    add_phase(phase_label="charge", alphabet="CHARGE")
    # Phase 2: free substitutions.
    add_phase(phase_label="random", alphabet="RANDOM")

    return jobs


def _run_job(
    job: QueuedJob,
    args: argparse.Namespace,
    results_root: Path,
    queue_log: Path,
) -> int:
    job_out_dir = results_root / job.name
    job_out_dir.mkdir(parents=True, exist_ok=True)
    launcher_log = job_out_dir / "launcher.log"

    effective_max_workers = args.max_workers
    if job.task.startswith("D_SHIFT"):
        effective_max_workers = args.max_workers_dshift
    if (
        job.task == "LGK"
        and "flat_esm_ridge_no_onehot" in job.surrogate_arch
    ):
        effective_max_workers = args.max_workers_flat_lgk

    cmd = [
        sys.executable,
        "src/prospero/runners/run_variable_k.py",
        str(job_out_dir),
        "--task",
        job.task,
        "--alphabet",
        job.alphabet,
        "--surrogate-arch",
        job.surrogate_arch,
        "--ridge-alpha",
        str(args.ridge_alpha),
        "--max-workers",
        str(effective_max_workers),
        "--n-samples",
        args.n_samples,
        "--seeds",
        args.seeds,
        "--n-iters",
        str(args.n_iters),
        *job.extra_args,
    ]

    run_env = os.environ.copy()
    if args.uv_cache_dir:
        run_env["UV_CACHE_DIR"] = args.uv_cache_dir
    run_env.setdefault("MPLCONFIGDIR", args.mplconfigdir)

    started_at = datetime.now().isoformat(timespec="seconds")
    with queue_log.open("a", encoding="utf-8") as qf:
        qf.write(
            json.dumps(
                {
                    "event": "job_start",
                    "name": job.name,
                    "task": job.task,
                    "surrogate_arch": job.surrogate_arch,
                    "alphabet": job.alphabet,
                    "max_workers": effective_max_workers,
                    "started_at": started_at,
                    "command": cmd,
                    "launcher_log": str(launcher_log),
                }
            )
            + "\n"
        )

    with launcher_log.open("w", encoding="utf-8") as lf:
        lf.write(f"# Started: {started_at}\n")
        lf.write(f"# Command: {shlex.join(cmd)}\n\n")
        lf.flush()
        process = subprocess.run(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=run_env,
            text=True,
            check=False,
        )

    finished_at = datetime.now().isoformat(timespec="seconds")
    with queue_log.open("a", encoding="utf-8") as qf:
        qf.write(
            json.dumps(
                {
                    "event": "job_end",
                    "name": job.name,
                    "task": job.task,
                    "surrogate_arch": job.surrogate_arch,
                    "alphabet": job.alphabet,
                    "finished_at": finished_at,
                    "return_code": process.returncode,
                    "launcher_log": str(launcher_log),
                }
            )
            + "\n"
        )

    return process.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Queue variable-k experiment runs sequentially."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(f"outputs/queued_ridge_sweeps_{_timestamp()}"),
    )
    parser.add_argument("--n-samples", default="8,16,32,64,128")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--n-iters", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument(
        "--max-workers-dshift",
        type=int,
        default=1,
        help="Override max workers for D_SHIFT* tasks to avoid ESMFold OOM.",
    )
    parser.add_argument(
        "--max-workers-flat-lgk",
        type=int,
        default=2,
        help="Override max workers for flat-ESM LGK jobs.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based job index to start from (for resume).",
    )
    parser.add_argument(
        "--skip-dshift",
        action="store_true",
        default=False,
        help="Skip all D_SHIFT* jobs.",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--uv-cache-dir", default=os.environ.get("UV_CACHE_DIR"))
    parser.add_argument("--mplconfigdir", default=".mplconfig")
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print queued jobs and commands, but do not execute them.",
    )
    args = parser.parse_args()

    results_root = args.results_root
    results_root.mkdir(parents=True, exist_ok=True)
    queue_log = results_root / "queue_events.jsonl"

    jobs = build_default_jobs()
    print(f"Queue root: {results_root}")
    print(f"Queue jobs: {len(jobs)}")
    for idx, job in enumerate(jobs, start=1):
        print(
            f"{idx:02d}. {job.name} | task={job.task} | "
            f"arch={job.surrogate_arch} | alphabet={job.alphabet}"
        )

    if args.print_only:
        print("Print-only mode, exiting without running jobs.")
        return

    if args.start_index < 1 or args.start_index > len(jobs):
        raise SystemExit(f"--start-index must be in [1, {len(jobs)}]")

    for idx, job in enumerate(jobs, start=1):
        if idx < args.start_index:
            continue
        if args.skip_dshift and job.task.startswith("D_SHIFT"):
            print(f"[{idx}/{len(jobs)}] Skipping {job.name} (D_SHIFT task)")
            continue
        print(f"[{idx}/{len(jobs)}] Starting {job.name}")
        rc = _run_job(job, args=args, results_root=results_root, queue_log=queue_log)
        if rc != 0:
            print(
                f"Job failed with return code {rc}: {job.name}. Stopping queue.",
                file=sys.stderr,
            )
            raise SystemExit(rc)
        print(f"[{idx}/{len(jobs)}] Finished {job.name}")

    print("Queue completed successfully.")


if __name__ == "__main__":
    main()
