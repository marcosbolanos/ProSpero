"""Run the published ProSpero CNN protocol at several oracle-query budgets."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SeedRun:
    task: str
    seed: int
    query_budget: int
    rounds: int
    output_directory: Path

    @property
    def result_path(self) -> Path:
        return self.output_directory / self.task / f"seed_{self.seed}.pkl"

    @property
    def completion_path(self) -> Path:
        return self.output_directory / self.task / f"seed_{self.seed}.complete.json"

    @property
    def log_path(self) -> Path:
        return self.output_directory / self.task / f"seed_{self.seed}.log"


def comma_separated_integers(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of positive integers."
        )
    return values


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ProSpero CNN campaigns over query budgets and seeds."
    )
    parser.add_argument("results_directory", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--query-budgets", type=comma_separated_integers, default=(8, 128)
    )
    parser.add_argument(
        "--seeds", type=comma_separated_integers, default=(1, 2, 3, 4, 5)
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--maximum-retries", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=float, default=5.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def result_is_complete(run: SeedRun) -> bool:
    if not run.result_path.is_file():
        return False
    try:
        with run.result_path.open("rb") as handle:
            result = pickle.load(handle)
    except (OSError, EOFError, pickle.UnpicklingError):
        return False
    return all(round_index in result for round_index in range(1, run.rounds + 1))


def command(run: SeedRun) -> list[str]:
    return [
        sys.executable,
        "-m",
        "prospero.runners.run_protein",
        "--task",
        run.task,
        "--results-directory",
        str(run.output_directory),
        "--query-budget",
        str(run.query_budget),
        "--rounds",
        str(run.rounds),
        "--seed",
        str(run.seed),
        "--deterministic-algorithms",
    ]


def execute(run: SeedRun, maximum_retries: int, retry_delay_seconds: float) -> Path:
    run.log_path.parent.mkdir(parents=True, exist_ok=True)
    if result_is_complete(run):
        return run.result_path
    for attempt in range(maximum_retries + 1):
        with run.log_path.open("a", encoding="utf-8") as log:
            log.write("$ " + " ".join(command(run)) + "\n")
            completed = subprocess.run(
                command(run), stdout=log, stderr=subprocess.STDOUT, check=False
            )
        if completed.returncode == 0 and result_is_complete(run):
            run.completion_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "task": run.task,
                        "seed": run.seed,
                        "query_budget": run.query_budget,
                        "rounds": run.rounds,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return run.result_path
        if attempt < maximum_retries:
            time.sleep(retry_delay_seconds * (attempt + 1))
    raise RuntimeError(
        f"ProSpero failed for {run.task}, K={run.query_budget}, seed={run.seed}; "
        f"see {run.log_path}."
    )


def main() -> None:
    args = get_parser().parse_args()
    if args.rounds < 1 or args.max_workers < 1:
        raise ValueError("Rounds and max workers must be positive.")
    runs = [
        SeedRun(
            task=args.task,
            seed=seed,
            query_budget=query_budget,
            rounds=args.rounds,
            output_directory=args.results_directory / f"k_{query_budget}",
        )
        for query_budget in args.query_budgets
        for seed in args.seeds
    ]
    pending = [run for run in runs if not (args.resume and result_is_complete(run))]
    print(f"Scheduling {len(pending)} of {len(runs)} seed runs", flush=True)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = {
            executor.submit(
                execute, run, args.maximum_retries, args.retry_delay_seconds
            ): run
            for run in pending
        }
        for future in concurrent.futures.as_completed(futures):
            run = futures[future]
            print(
                f"Complete {run.task} K={run.query_budget} seed={run.seed}: "
                f"{future.result()}",
                flush=True,
            )


if __name__ == "__main__":
    main()
