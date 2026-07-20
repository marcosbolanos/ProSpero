import sys
import os
import json
import threading
import time
import resource
from prospero.experiments_config import ALPHABETS, WT_SEQUENCES


from prospero.utils import get_new_starting_seq, set_seed
from prospero.experiment_tracker import ExperimentTracker
from prospero.inference import ProteinSampler

from prospero.surrogate import (
    Ensemble,
    build_surrogate_model,
)
from prospero.dataset import RegressionDataset
from prospero.landscapes import get_landscape

import argparse
from argparse import ArgumentDefaultsHelpFormatter
import numpy as np
from copy import deepcopy
from datetime import datetime, timezone

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return repr(value)


def _read_proc_status() -> dict[str, str]:
    status = {}
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                status[key.strip()] = value.strip()
    except OSError:
        return {}
    return status


def _resource_snapshot() -> dict[str, object]:
    snapshot: dict[str, object] = {
        "pid": os.getpid(),
        "wall_time": time.time(),
        "ru_maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    proc_status = _read_proc_status()
    if proc_status:
        snapshot["vmrss"] = proc_status.get("VmRSS")
        snapshot["vmhwm"] = proc_status.get("VmHWM")
        snapshot["threads"] = proc_status.get("Threads")
    try:
        import torch

        if torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
            snapshot.update(
                {
                    "cuda_device": device_idx,
                    "cuda_mem_free_mb": round(free_bytes / 1024 / 1024, 2),
                    "cuda_mem_total_mb": round(total_bytes / 1024 / 1024, 2),
                    "cuda_mem_allocated_mb": round(
                        torch.cuda.memory_allocated(device_idx) / 1024 / 1024, 2
                    ),
                    "cuda_mem_reserved_mb": round(
                        torch.cuda.memory_reserved(device_idx) / 1024 / 1024, 2
                    ),
                }
            )
    except Exception as error:
        snapshot["cuda_snapshot_error"] = repr(error)
    return snapshot


class SeedDebugLogger:
    def __init__(
        self,
        path: str,
        *,
        seed: int,
        task: str,
        heartbeat_seconds: float,
    ) -> None:
        self.path = path
        self.seed = seed
        self.task = task
        self.heartbeat_seconds = heartbeat_seconds
        self._lock = threading.Lock()
        self._phase = "initializing"
        self._stop_event = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def event(self, name: str, **fields) -> None:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": name,
            "task": self.task,
            "seed": self.seed,
            "phase": self._phase,
            **_resource_snapshot(),
            **{key: _json_safe(value) for key, value in fields.items()},
        }
        line = json.dumps(payload, sort_keys=True)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")

    def set_phase(self, phase: str, **fields) -> None:
        self._phase = phase
        self.event("phase", phase=phase, **fields)

    def log_exception(self, error: Exception) -> None:
        self.event(
            "exception",
            error_type=type(error).__name__,
            error_message=str(error),
        )

    def start_heartbeat(self) -> None:
        def _heartbeat() -> None:
            while not self._stop_event.wait(self.heartbeat_seconds):
                self.event("heartbeat")

        self._heartbeat_thread = threading.Thread(
            target=_heartbeat,
            name=f"seed-{self.seed}-debug-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1.0)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one ProSpero CNN optimization campaign.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-directory", required=True)
    parser.add_argument("--query-budget", type=int, default=128)
    parser.add_argument("--seed", type=int, choices=range(1, 6), default=1)
    parser.add_argument(
        "--task",
        choices=[task for task in WT_SEQUENCES if not task.startswith("D_SHIFT")],
        required=True,
    )
    parser.add_argument("--deterministic-algorithms", action="store_true")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--resampling-steps", type=int, default=1)
    parser.add_argument("--candidate-batch-size", type=int, default=256)
    parser.add_argument(
        "--substitution-alphabet", default="CHARGE", choices=list(ALPHABETS)
    )
    parser.add_argument("--ucb-scan-coefficient", type=float, default=1.0)
    parser.add_argument("--ucb-guidance-coefficient", type=float, default=0.1)
    parser.add_argument("--scan-multiplier", type=int, default=16)
    parser.add_argument("--minimum-mutations", type=int, default=3)
    parser.add_argument("--maximum-mutations", type=int, default=10)
    parser.add_argument("--maximum-epochs", type=int, default=3000)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--validation-frequency", type=int, default=1)
    parser.add_argument("--surrogate-batch-size", type=int, default=256)
    parser.add_argument("--surrogate-architecture", choices=["cnn"], default="cnn")
    parser.add_argument("--debug-events", action="store_true")
    parser.add_argument("--debug-heartbeat-seconds", type=float, default=15.0)
    return parser


def run_iter(args: argparse.Namespace, logger: logging.Logger) -> None:
    seed = args.seed
    set_seed(seed, args.deterministic_algorithms)
    logger.info(f"Starting seed {seed}")

    save_dir = os.path.join(args.results_directory, args.task)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"seed_{seed}.pkl")
    debug_logger = None
    if getattr(args, "debug_events", False):
        debug_logger = SeedDebugLogger(
            os.path.join(save_dir, f"seed_{seed}.debug.jsonl"),
            seed=seed,
            task=args.task,
            heartbeat_seconds=max(1.0, args.debug_heartbeat_seconds),
        )
        debug_logger.event(
            "seed_start",
            surrogate_architecture=args.surrogate_architecture,
            query_budget=args.query_budget,
            rounds=args.rounds,
            substitution_alphabet=args.substitution_alphabet,
            results_directory=args.results_directory,
        )
        debug_logger.start_heartbeat()

    wt_sequence = WT_SEQUENCES[args.task]
    if debug_logger is not None:
        debug_logger.set_phase("oracle_load_start")
    oracle = get_landscape(args.task)
    if debug_logger is not None:
        debug_logger.set_phase("oracle_load_complete")
    dataset = RegressionDataset(args.task)
    if debug_logger is not None:
        debug_logger.event(
            "dataset_loaded",
            train_size=len(dataset.train),
            valid_size=len(dataset.valid),
        )

    try:
        if debug_logger is not None:
            debug_logger.set_phase("initial_surrogate_build")
        proxy = Ensemble(
            [
                build_surrogate_model(len(wt_sequence), args)
                for _ in range(args.ensemble_size)
            ]
        )
        if debug_logger is not None:
            debug_logger.set_phase("initial_surrogate_train_start")
        logger.info("Training started")
        proxy.train(dataset)
        logger.info("Training finished")
        if debug_logger is not None:
            debug_logger.set_phase("initial_surrogate_train_complete")

        alphabet = ALPHABETS[args.substitution_alphabet]

        if debug_logger is not None:
            debug_logger.set_phase("oadm_model_load_start")
        from evodiff.pretrained import OA_DM_38M  # type: ignore[reportMissingImports]

        model, _, tokenizer_oadm, _ = OA_DM_38M()
        model = model.cuda()
        if debug_logger is not None:
            debug_logger.set_phase("oadm_model_load_complete")
        exp_tracker = ExperimentTracker(
            logger, deepcopy(dataset), wt_sequence, best_percentile=0.95
        )

        starting_sequence = WT_SEQUENCES[args.task]

        for e in range(args.rounds):
            iteration = e + 1
            if debug_logger is not None:
                debug_logger.set_phase(
                    "iteration_start",
                    iteration=iteration,
                    dataset_train_size=len(dataset.train),
                    dataset_valid_size=len(dataset.valid),
                )
            # This class implements algos 2, 3 & 4
            sampler = ProteinSampler(model, tokenizer_oadm, alphabet)
            sequences = list()
            ref_sequences = list(dataset.train) + list(
                dataset.valid
            )  # So we don't regenerate smth that's already in
            # generate new sequences
            generation_round = 0
            while len(sequences) < args.query_budget:
                generation_round += 1
                if debug_logger is not None:
                    debug_logger.event(
                        "generation_round_start",
                        iteration=iteration,
                        generation_round=generation_round,
                        current_sequences=len(sequences),
                    )
                sampler.generate_raa_from_alanine_scan(
                    proxy,
                    starting_sequence,
                    args.candidate_batch_size,
                    args.resampling_steps,
                    args.minimum_mutations,
                    args.maximum_mutations,
                    args.ucb_scan_coefficient,
                    args.scan_multiplier,
                    args.ucb_guidance_coefficient,
                )
                # This method is inherited from parent Sampler class
                top_sequences = sampler.get_top_sequences(
                    args.query_budget, ref_sequences
                )
                sequences += top_sequences
                ref_sequences += sequences  # add sequences to those we've already seen
                if debug_logger is not None:
                    debug_logger.event(
                        "generation_round_complete",
                        iteration=iteration,
                        generation_round=generation_round,
                        new_sequences=len(top_sequences),
                        candidate_pool_size=len(sequences),
                    )

            sequences = sequences[: args.query_budget]
            assert len(sequences) == args.query_budget
            if debug_logger is not None:
                debug_logger.event(
                    "candidate_batch_ready",
                    iteration=iteration,
                    sequence_count=len(sequences),
                )

            # eval candidate sequences
            if debug_logger is not None:
                debug_logger.set_phase("oracle_scoring", iteration=iteration)
            scores = oracle.get_fitness(np.array(sequences)).tolist()
            if debug_logger is not None and scores:
                debug_logger.event(
                    "oracle_scoring_complete",
                    iteration=iteration,
                    score_min=min(scores),
                    score_max=max(scores),
                    score_mean=float(np.mean(scores)),
                )

            # append dataset and retrain the surrogate
            dataset.add((sequences, scores))
            if debug_logger is not None:
                debug_logger.event(
                    "dataset_augmented",
                    iteration=iteration,
                    dataset_train_size=len(dataset.train),
                    dataset_valid_size=len(dataset.valid),
                )
            exp_tracker.calculate_top_n_metrics((sequences, scores), iteration, n=100)
            starting_sequence = get_new_starting_seq(dataset)
            if debug_logger is not None:
                debug_logger.event(
                    "starting_sequence_updated",
                    iteration=iteration,
                    starting_sequence_length=len(starting_sequence),
                    starting_sequence_preview=starting_sequence[:20],
                )

            if debug_logger is not None:
                debug_logger.set_phase("surrogate_rebuild", iteration=iteration)
            proxy = Ensemble(
                [
                    build_surrogate_model(len(wt_sequence), args)
                    for _ in range(args.ensemble_size)
                ]
            )
            exp_tracker.save_results(save_path)
            if debug_logger is not None:
                debug_logger.event(
                    "checkpoint_saved",
                    iteration=iteration,
                    save_path=save_path,
                    save_size_bytes=os.path.getsize(save_path),
                )
            if iteration < args.rounds:
                if debug_logger is not None:
                    debug_logger.set_phase(
                        "iteration_surrogate_train_start",
                        iteration=iteration,
                    )
                proxy.train(dataset)
                if debug_logger is not None:
                    debug_logger.set_phase(
                        "iteration_surrogate_train_complete",
                        iteration=iteration,
                    )
        if debug_logger is not None:
            debug_logger.set_phase(
                "seed_complete",
                save_path=save_path,
                save_size_bytes=os.path.getsize(save_path),
            )
    except Exception as error:
        if debug_logger is not None:
            debug_logger.log_exception(error)
        raise
    finally:
        if debug_logger is not None:
            debug_logger.event("seed_cleanup_start")
        if debug_logger is not None:
            debug_logger.event("seed_cleanup_complete")
            debug_logger.stop()


def main():
    parser = get_parser()
    args = parser.parse_args()
    run_iter(args, logger)


if __name__ == "__main__":
    main()
