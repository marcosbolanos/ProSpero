"""Run the variable-n_queries experiments from Python instead of bash."""

import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
import os
import pickle
import resource
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Optional, Sequence, TextIO

from tqdm import tqdm
from prospero.representations.interplm import DEFAULT_INTERPLM_REPO_ID


class TeeStream:
    """Duplicate writes so console output is mirrored in the log file."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(
            getattr(stream, "isatty", lambda: False)() for stream in self._streams
        )


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
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


class JsonlEventLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, name: str, **fields) -> None:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": name,
            **_resource_snapshot(),
            **{key: _json_safe(value) for key, value in fields.items()},
        }
        line = json.dumps(payload, sort_keys=True)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")


@dataclass(frozen=True)
class SharedESMComponents:
    tokenizer: object
    esm: object
    esm_forward_lock: object | None = None
    esm_batch_worker: object | None = None
    esm_in_memory_cache_pool: object | None = None
    interplm_sae_pool: object | None = None


def _seed_result_path(batch_dir: Path, task: str, seed: int) -> Path:
    return batch_dir / task / f"seed_{seed}.pkl"


def _is_valid_seed_result(path: Path, *, n_iters: int) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with open(path, "rb") as handle:
            result = pickle.load(handle)
    except Exception:
        return False
    if not isinstance(result, dict):
        return False
    expected_keys = set(range(1, n_iters + 1))
    result_keys = set(result.keys())
    return result_keys == expected_keys


def _existing_valid_seed_ids(
    batch_dir: Path,
    task: str,
    seeds: Sequence[int],
    *,
    n_iters: int,
) -> list[int]:
    return [
        seed
        for seed in seeds
        if _is_valid_seed_result(
            _seed_result_path(batch_dir, task, seed),
            n_iters=n_iters,
        )
    ]


class ThreadSafeOracle:
    """Serialize oracle calls so one shared oracle can serve multiple seeds safely."""

    def __init__(self, oracle: object) -> None:
        self._oracle = oracle
        self._lock = threading.Lock()

    def get_fitness(self, sequences):
        with self._lock:
            return self._oracle.get_fitness(sequences)


def parse_int_list(value: str) -> list[int]:
    try:
        return [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid integer list: {value}") from error


def _run_command(process_args: Sequence[str], env: Optional[dict[str, str]]) -> None:
    run_env = dict(env) if env is not None else os.environ.copy()
    # Encourage child Python processes to flush output promptly.
    run_env.setdefault("PYTHONUNBUFFERED", "1")
    run_env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    print(f"[driver] launching: {' '.join(process_args)}", flush=True)

    with subprocess.Popen(
        process_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=run_env,
        text=True,
        bufsize=1,
    ) as process:
        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="")
        return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, process_args)


def run_seed(process_args: Sequence[str], env: dict[str, str]) -> None:
    _run_command(process_args, env)


def _run_seed_with_retries(
    process_args: Sequence[str],
    env: dict[str, str],
    *,
    safe: bool,
    max_seed_retries: int,
    retry_backoff_seconds: float,
) -> None:
    total_attempts = 1 + (max_seed_retries if safe else 0)
    for attempt in range(1, total_attempts + 1):
        try:
            run_seed(process_args, env)
            return
        except Exception:
            if attempt >= total_attempts:
                raise
            wait_seconds = retry_backoff_seconds * attempt
            print(
                f"[safe] seed command failed (attempt {attempt}/{total_attempts}), "
                f"retrying in {wait_seconds:.1f}s: {' '.join(process_args)}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait_seconds)


def prepare_shared_esm_components(args: argparse.Namespace) -> SharedESMComponents:
    import torch
    from transformers import AutoModel, AutoTokenizer
    from prospero.surrogate import SharedESMBatchWorker, SharedInMemoryESMCachePool
    from prospero.representations.interplm import SharedInterPLMSAEPool

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[shared-esm] loading tokenizer from local cache", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.esm_model_name,
            local_files_only=True,
        )
    except Exception:
        print("[shared-esm] local tokenizer cache miss, falling back to hub", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.esm_model_name)

    print("[shared-esm] loading model from local cache", flush=True)
    try:
        esm = AutoModel.from_pretrained(
            args.esm_model_name,
            local_files_only=True,
        ).to(device)
    except Exception:
        print("[shared-esm] local model cache miss, falling back to hub", flush=True)
        esm = AutoModel.from_pretrained(args.esm_model_name).to(device)

    print(f"[shared-esm] shared ESM model loaded on {device}", flush=True)
    esm.eval()
    for param in esm.parameters():
        param.requires_grad = False
    esm_forward_lock = threading.Lock()
    esm_batch_worker = SharedESMBatchWorker(
        tokenizer=tokenizer,
        esm=esm,
        device=device,
        max_batch_sequences=512,
        max_wait_ms=4.0,
    )
    esm_in_memory_cache_pool = SharedInMemoryESMCachePool(storage_dtype=torch.float16)
    interplm_sae_pool = SharedInterPLMSAEPool()
    return SharedESMComponents(
        tokenizer=tokenizer,
        esm=esm,
        esm_forward_lock=esm_forward_lock,
        esm_batch_worker=esm_batch_worker,
        esm_in_memory_cache_pool=esm_in_memory_cache_pool,
        interplm_sae_pool=interplm_sae_pool,
    )


def _build_protein_args(
    runner_args: argparse.Namespace,
    batch_dir: Path,
    n_iters: int,
    n_queries: int,
    seed: int,
) -> argparse.Namespace:
    from prospero.runners.run_protein import get_parser as get_protein_parser

    protein_args = get_protein_parser().parse_args([])
    protein_args.task = runner_args.task
    protein_args.results_dirpath = str(batch_dir)
    protein_args.n_iters = n_iters
    protein_args.alphabet = runner_args.alphabet
    protein_args.min_corruptions = runner_args.min_corruptions
    protein_args.max_corruptions = runner_args.max_corruptions
    protein_args.mask_strategy = runner_args.mask_strategy
    protein_args.sequence_scoring = runner_args.sequence_scoring
    protein_args.mask_count_calibration_rounds = runner_args.mask_count_calibration_rounds
    protein_args.surrogate_arch = runner_args.surrogate_arch
    protein_args.full_deterministic = True
    protein_args.n_queries = n_queries
    protein_args.seed = seed
    protein_args.esm_cnn_projection_dim = runner_args.esm_cnn_projection_dim
    protein_args.esm_cnn_use_layernorm = runner_args.esm_cnn_use_layernorm
    protein_args.esm_cnn_concat_one_hot = runner_args.esm_cnn_concat_one_hot
    protein_args.esm_model_name = runner_args.esm_model_name
    protein_args.esm_max_length = runner_args.esm_max_length
    protein_args.esm_representation_layer = runner_args.esm_representation_layer
    protein_args.esm_truncate_to_representation_layer = (
        runner_args.esm_truncate_to_representation_layer
    )
    protein_args.esm_lora_adapter_path = runner_args.esm_lora_adapter_path
    protein_args.ridge_alpha = runner_args.ridge_alpha
    protein_args.ridge_fit_intercept = runner_args.ridge_fit_intercept
    protein_args.interplm_layer = runner_args.interplm_layer
    protein_args.interplm_repo_id = runner_args.interplm_repo_id
    protein_args.interplm_normalized = runner_args.interplm_normalized
    protein_args.sae_token_chunk_size = runner_args.sae_token_chunk_size
    protein_args.low_rank_positional_rank = getattr(
        runner_args,
        "low_rank_positional_rank",
        16,
    )
    protein_args.low_rank_positional_l2 = getattr(
        runner_args,
        "low_rank_positional_l2",
        1e-4,
    )
    protein_args.low_rank_positional_lr = getattr(
        runner_args,
        "low_rank_positional_lr",
        None,
    )
    protein_args.low_rank_positional_repr_batch_size = getattr(
        runner_args,
        "low_rank_positional_repr_batch_size",
        None,
    )
    protein_args.low_rank_positional_input = getattr(
        runner_args,
        "low_rank_positional_input",
        "sae",
    )
    protein_args.disable_esm_cache = runner_args.disable_esm_cache
    protein_args.debug_events = runner_args.debug_events
    protein_args.debug_heartbeat_seconds = runner_args.debug_heartbeat_seconds
    ensemble_size = getattr(runner_args, "ensemble_size", None)
    proxy_batch_size = getattr(runner_args, "proxy_batch_size", None)
    if ensemble_size is not None:
        protein_args.ensemble_size = ensemble_size
    if proxy_batch_size is not None:
        protein_args.proxy_batch_size = proxy_batch_size
    protein_args.initial_train_top_k = getattr(
        runner_args,
        "initial_train_top_k",
        None,
    )
    if runner_args.surrogate_arch in {
        "interplm_mean_pool_ridge",
        "interplm_low_rank_positional",
        "frozen_esm_layer_cls_ridge",
    }:
        if ensemble_size is None:
            protein_args.ensemble_size = 1
    return protein_args


def _run_seed_in_process(
    protein_args: argparse.Namespace,
    shared_esm_components,
    shared_oracle,
) -> None:
    from prospero.runners.run_protein import (
        logger as protein_logger,
        run_iter as run_protein_iter,
    )

    print(f"[in-process] starting seed {protein_args.seed}", flush=True)
    try:
        run_protein_iter(
            protein_args,
            protein_logger,
            shared_esm_components,
            shared_oracle=shared_oracle,
        )
    except Exception:
        print(
            f"[in-process] seed {protein_args.seed} raised an exception",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        raise
    print(f"[in-process] finished seed {protein_args.seed}", flush=True)


def _run_seed_in_process_with_retries(
    protein_args: argparse.Namespace,
    shared_esm_components,
    shared_oracle,
    *,
    safe: bool,
    max_seed_retries: int,
    retry_backoff_seconds: float,
) -> None:
    total_attempts = 1 + (max_seed_retries if safe else 0)
    for attempt in range(1, total_attempts + 1):
        try:
            _run_seed_in_process(protein_args, shared_esm_components, shared_oracle)
            return
        except Exception:
            if attempt >= total_attempts:
                raise
            # Best-effort cleanup before retrying after a failure.
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            wait_seconds = retry_backoff_seconds * attempt
            print(
                f"[safe] in-process seed {protein_args.seed} failed "
                f"(attempt {attempt}/{total_attempts}), retrying in "
                f"{wait_seconds:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait_seconds)


def main() -> None:
    frozen_esm_surrogate_archs = {
        "interplm_mean_pool_ridge",
        "interplm_low_rank_positional",
        "frozen_esm_mlp",
        "frozen_esm_cnn",
        "frozen_esm_cls_ridge",
        "frozen_esm_layer_cls_ridge",
        "frozen_esm_flat_linear",
        "frozen_esm_flat_ridge",
        "frozen_esm_flat_ridge_no_onehot",
    }
    parser = argparse.ArgumentParser(
        description="Run variable-n_queries seeds with a shared Python driver."
    )
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--task", default="AAV")
    parser.add_argument("--alphabet", default="CHARGE")
    parser.add_argument(
        "--surrogate-arch",
        default="cnn",
        choices=(
            "cnn",
            "one_hot_ridge",
            "interplm_mean_pool_ridge",
            "interplm_low_rank_positional",
            "frozen_esm_mlp",
            "frozen_esm_cnn",
            "frozen_esm_cls_ridge",
            "frozen_esm_layer_cls_ridge",
            "frozen_esm_flat_linear",
            "frozen_esm_flat_ridge",
            "frozen_esm_flat_ridge_no_onehot",
        ),
    )
    parser.add_argument("--esm-cnn-projection-dim", type=int, default=None)
    parser.add_argument("--esm-cnn-use-layernorm", action="store_true", default=False)
    parser.add_argument("--esm-cnn-concat-one-hot", action="store_true", default=False)
    parser.add_argument(
        "--esm-model-name",
        default="facebook/esm2_t6_8M_UR50D",
    )
    parser.add_argument("--esm-max-length", type=int, default=None)
    parser.add_argument("--esm-representation-layer", type=int, default=2)
    parser.add_argument(
        "--esm-truncate-to-representation-layer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--esm-lora-adapter-path", default=None)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--ridge-fit-intercept",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--interplm-layer", type=int, default=2)
    parser.add_argument(
        "--interplm-repo-id",
        default=DEFAULT_INTERPLM_REPO_ID,
    )
    parser.add_argument(
        "--interplm-normalized",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sae-token-chunk-size", type=int, default=1024)
    parser.add_argument("--low-rank-positional-rank", type=int, default=16)
    parser.add_argument("--low-rank-positional-l2", type=float, default=1e-4)
    parser.add_argument("--low-rank-positional-lr", type=float, default=None)
    parser.add_argument("--low-rank-positional-repr-batch-size", type=int, default=None)
    parser.add_argument(
        "--low-rank-positional-input",
        choices=("esm", "sae", "esm_sae_concat"),
        default="sae",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=None,
        help="Override surrogate ensemble size. Defaults to 1 for InterPLM ridge/low-rank surrogate arches.",
    )
    parser.add_argument(
        "--proxy-batch-size",
        type=int,
        default=None,
        help="Override proxy batch size.",
    )
    parser.add_argument("--disable-esm-cache", action="store_true", default=False)
    parser.add_argument("--n-samples", default="8,16,32,64,128")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--n-iters", type=int, default=10)
    parser.add_argument("--min-corruptions", type=int, default=3)
    parser.add_argument("--max-corruptions", type=int, default=10)
    parser.add_argument(
        "--mask-strategy",
        choices=("targeted", "random"),
        default="targeted",
        help="Use standard surrogate-targeted masks or calibrated random masks.",
    )
    parser.add_argument(
        "--sequence-scoring",
        choices=("surrogate", "zero_shot_logp_delta"),
        default="surrogate",
        help="Score generated sequences with surrogate UCB or logP-delta zero-shot score.",
    )
    parser.add_argument(
        "--mask-count-calibration-rounds",
        type=int,
        default=32,
        help="Targeted-mask batches used to calibrate selected mask-count stats for random masking.",
    )
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--safe", action="store_true", default=False)
    parser.add_argument("--resume-missing-seeds", action="store_true", default=False)
    parser.add_argument("--max-seed-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-seconds", type=float, default=5.0)
    parser.add_argument("--debug-events", action="store_true", default=False)
    parser.add_argument("--debug-heartbeat-seconds", type=float, default=15.0)
    parser.add_argument(
        "--initial-train-top-k",
        type=int,
        default=None,
        help=(
            "Restrict each seed's initial training split to the top-k sequences "
            "by training score before the Prospero loop starts."
        ),
    )
    parser.add_argument("--n-queries-base", type=int, default=None)
    parser.add_argument("--uv-cache-dir", default=os.environ.get("UV_CACHE_DIR"))
    parser.add_argument(
        "--share-oracle",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Share a single oracle instance across in-process seed workers. "
            "Defaults to enabled for D_SHIFT* tasks."
        ),
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "run_variable_k.log"
    with open(log_path, "a", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        heartbeat_stop_event = threading.Event()
        driver_debug_logger = (
            JsonlEventLogger(results_dir / "driver.debug.jsonl")
            if args.debug_events
            else None
        )
        try:
            print(f"Logging console output to {log_path}")
            if driver_debug_logger is not None:
                driver_debug_logger.event(
                    "driver_start",
                    argv=sys.argv,
                    results_dir=str(results_dir),
                )

                def _heartbeat() -> None:
                    while not heartbeat_stop_event.wait(
                        max(1.0, args.debug_heartbeat_seconds)
                    ):
                        driver_debug_logger.event("driver_heartbeat")

                heartbeat_thread = threading.Thread(
                    target=_heartbeat,
                    name="run-variable-k-heartbeat",
                    daemon=True,
                )
                heartbeat_thread.start()

                def _signal_handler(signum, _frame) -> None:
                    driver_debug_logger.event(
                        "driver_signal",
                        signal=signum,
                    )

                for sig_name in ("SIGINT", "SIGTERM", "SIGHUP"):
                    sig = getattr(signal, sig_name, None)
                    if sig is not None:
                        signal.signal(sig, _signal_handler)

            n_samples_values = parse_int_list(args.n_samples)
            seeds = parse_int_list(args.seeds)
            max_workers = max(1, args.max_workers)
            n_iters = args.n_iters
            scheduled_seeds_by_n_samples: dict[int, list[int]] = {}
            completed_seeds_by_n_samples: dict[int, list[int]] = {}
            for n_samples in n_samples_values:
                batch_dir = results_dir / f"n_samples_{n_samples}"
                completed_seeds = _existing_valid_seed_ids(
                    batch_dir,
                    args.task,
                    seeds,
                    n_iters=n_iters,
                )
                completed_seeds_by_n_samples[n_samples] = completed_seeds
                if args.resume_missing_seeds:
                    scheduled_seeds_by_n_samples[n_samples] = [
                        seed for seed in seeds if seed not in completed_seeds
                    ]
                else:
                    scheduled_seeds_by_n_samples[n_samples] = list(seeds)
            share_frozen_esm = args.surrogate_arch in frozen_esm_surrogate_archs
            share_oracle = (
                args.task.startswith("D_SHIFT")
                if args.share_oracle is None
                else args.share_oracle
            )
            if share_frozen_esm:
                print(
                    f"[shared-esm] enabled by default for {args.surrogate_arch}",
                    flush=True,
                )
            if share_oracle:
                print(
                    f"[shared-oracle] enabled for task {args.task}",
                    flush=True,
                )
            if driver_debug_logger is not None and share_frozen_esm:
                driver_debug_logger.event(
                    "shared_esm_enabled",
                    surrogate_arch=args.surrogate_arch,
                )
            if driver_debug_logger is not None and share_oracle:
                driver_debug_logger.event(
                    "shared_oracle_enabled",
                    task=args.task,
                )
            if args.safe:
                print(
                    "[safe] enabled: failed seeds will be retried "
                    f"up to {args.max_seed_retries} times",
                    flush=True,
                )
            shared_oracle = None
            if share_oracle:
                from prospero.landscapes import get_landscape

                print(
                    f"[shared-oracle] loading oracle once for task {args.task}",
                    flush=True,
                )
                shared_oracle = ThreadSafeOracle(get_landscape(args.task))
                print("[shared-oracle] ready", flush=True)
            if args.resume_missing_seeds:
                print(
                    "[resume] enabled: valid existing seed checkpoints will be skipped",
                    flush=True,
                )
                if driver_debug_logger is not None:
                    driver_debug_logger.event(
                        "resume_missing_seeds_enabled",
                        completed_seeds_by_n_samples=completed_seeds_by_n_samples,
                    )
            if driver_debug_logger is not None and args.safe:
                driver_debug_logger.event(
                    "safe_enabled",
                    max_seed_retries=args.max_seed_retries,
                    retry_backoff_seconds=args.retry_backoff_seconds,
                )
            if share_oracle and driver_debug_logger is not None:
                driver_debug_logger.event(
                    "shared_oracle_ready",
                    task=args.task,
                )

            total_runs = sum(
                len(scheduled_seeds_by_n_samples[n_samples])
                for n_samples in n_samples_values
            )
            with tqdm(
                total=total_runs,
                desc="variable_k seeds",
                unit="seed",
                leave=True,
            ) as progress:
                for n_samples in n_samples_values:
                    batch_dir = results_dir / f"n_samples_{n_samples}"
                    batch_dir.mkdir(parents=True, exist_ok=True)
                    seeds_to_run = scheduled_seeds_by_n_samples[n_samples]
                    completed_seeds = completed_seeds_by_n_samples[n_samples]
                    print(f"Running {args.task} seeds for n_samples={n_samples}")
                    if args.resume_missing_seeds:
                        print(
                            f"[resume] n_samples={n_samples}: "
                            f"completed={completed_seeds or 'none'} "
                            f"pending={seeds_to_run or 'none'}",
                            flush=True,
                        )
                    if driver_debug_logger is not None:
                        driver_debug_logger.event(
                            "n_samples_start",
                            n_samples=n_samples,
                            batch_dir=str(batch_dir),
                            seeds=seeds,
                            completed_seeds=completed_seeds,
                            scheduled_seeds=seeds_to_run,
                        )

                    cmd_template = [
                        sys.executable,
                        "src/prospero/runners/run_protein.py",
                        "--task",
                        args.task,
                        "--alphabet",
                        args.alphabet,
                        "--results_dirpath",
                        str(batch_dir),
                        "--n_iters",
                        str(n_iters),
                        "--min_corruptions",
                        str(args.min_corruptions),
                        "--max_corruptions",
                        str(args.max_corruptions),
                        "--mask_strategy",
                        args.mask_strategy,
                        "--sequence_scoring",
                        args.sequence_scoring,
                        "--mask_count_calibration_rounds",
                        str(args.mask_count_calibration_rounds),
                        "--surrogate_arch",
                        args.surrogate_arch,
                        "--full_deterministic",
                    ]
                    if args.esm_cnn_projection_dim is not None:
                        cmd_template.extend(
                            [
                                "--esm_cnn_projection_dim",
                                str(args.esm_cnn_projection_dim),
                            ]
                        )
                    if args.esm_cnn_use_layernorm:
                        cmd_template.append("--esm_cnn_use_layernorm")
                    if args.esm_cnn_concat_one_hot:
                        cmd_template.append("--esm_cnn_concat_one_hot")
                    cmd_template.extend(
                        [
                            "--esm_representation_layer",
                            str(args.esm_representation_layer),
                        ]
                    )
                    if not args.esm_truncate_to_representation_layer:
                        cmd_template.append("--no-esm_truncate_to_representation_layer")
                    if args.esm_lora_adapter_path is not None:
                        cmd_template.extend(
                            [
                                "--esm_lora_adapter_path",
                                str(args.esm_lora_adapter_path),
                            ]
                        )
                    n_queries = (
                        args.n_queries_base
                        if args.n_queries_base is not None
                        else n_samples
                    )
                    cmd_template.extend(["--n_queries", str(n_queries)])
                    cmd_template.extend(["--ridge_alpha", str(args.ridge_alpha)])
                    cmd_template.extend(["--interplm_layer", str(args.interplm_layer)])
                    cmd_template.extend(["--interplm_repo_id", args.interplm_repo_id])
                    if not args.interplm_normalized:
                        cmd_template.append("--no-interplm_normalized")
                    cmd_template.extend(
                        ["--sae_token_chunk_size", str(args.sae_token_chunk_size)]
                    )
                    cmd_template.extend(
                        [
                            "--low_rank_positional_rank",
                            str(args.low_rank_positional_rank),
                        ]
                    )
                    cmd_template.extend(
                        [
                            "--low_rank_positional_l2",
                            str(args.low_rank_positional_l2),
                        ]
                    )
                    if args.low_rank_positional_lr is not None:
                        cmd_template.extend(
                            [
                                "--low_rank_positional_lr",
                                str(args.low_rank_positional_lr),
                            ]
                        )
                    if args.low_rank_positional_repr_batch_size is not None:
                        cmd_template.extend(
                            [
                                "--low_rank_positional_repr_batch_size",
                                str(args.low_rank_positional_repr_batch_size),
                            ]
                        )
                    cmd_template.extend(
                        [
                            "--low_rank_positional_input",
                            str(args.low_rank_positional_input),
                        ]
                    )
                    if not args.ridge_fit_intercept:
                        cmd_template.append("--no-ridge_fit_intercept")
                    if args.disable_esm_cache:
                        cmd_template.append("--disable-esm-cache")
                    if args.debug_events:
                        cmd_template.append("--debug-events")
                        cmd_template.extend(
                            [
                                "--debug-heartbeat-seconds",
                                str(args.debug_heartbeat_seconds),
                            ]
                        )
                    if args.initial_train_top_k is not None:
                        cmd_template.extend(
                            [
                                "--initial_train_top_k",
                                str(args.initial_train_top_k),
                            ]
                        )

                    env = os.environ.copy()
                    if args.uv_cache_dir:
                        env["UV_CACHE_DIR"] = args.uv_cache_dir

                    errors: list[tuple[Sequence[str], Exception]] = []
                    seed_cmds = [
                        cmd_template + ["--seed", str(seed)] for seed in seeds_to_run
                    ]
                    use_in_process_runner = share_frozen_esm or share_oracle
                    if not seeds_to_run:
                        print(
                            f"[resume] no pending seeds for n_samples={n_samples}",
                            flush=True,
                        )
                        if driver_debug_logger is not None:
                            driver_debug_logger.event(
                                "n_samples_resume_skip",
                                n_samples=n_samples,
                                completed_seeds=completed_seeds,
                            )
                    elif use_in_process_runner:
                        executor_workers = max_workers
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=executor_workers
                        ) as executor:
                            future_to_cmd = {}
                            shared_esm_components = None
                            if args.uv_cache_dir:
                                os.environ["UV_CACHE_DIR"] = args.uv_cache_dir
                            if share_frozen_esm:
                                print(
                                    f"[shared-esm] preparing shared ESM for {args.surrogate_arch}",
                                    flush=True,
                                )
                                shared_esm_components = prepare_shared_esm_components(args)
                                print("[shared-esm] shared ESM ready", flush=True)
                                if driver_debug_logger is not None:
                                    driver_debug_logger.event(
                                        "shared_esm_prepare_complete",
                                        surrogate_arch=args.surrogate_arch,
                                    )
                            for seed in seeds_to_run:
                                protein_args = _build_protein_args(
                                    runner_args=args,
                                    batch_dir=batch_dir,
                                    n_iters=n_iters,
                                    n_queries=n_queries,
                                    seed=seed,
                                )
                                print(
                                    f"[shared-esm] submitting seed {seed}",
                                    flush=True,
                                )
                                if driver_debug_logger is not None:
                                    driver_debug_logger.event(
                                        "seed_submitted",
                                        seed=seed,
                                        n_samples=n_samples,
                                        mode="in_process",
                                    )
                                future = executor.submit(
                                    _run_seed_in_process_with_retries,
                                    deepcopy(protein_args),
                                    shared_esm_components,
                                    shared_oracle,
                                    safe=args.safe,
                                    max_seed_retries=args.max_seed_retries,
                                    retry_backoff_seconds=args.retry_backoff_seconds,
                                )
                                future_to_cmd[future] = (
                                    f"in-process run_protein seed={seed}",
                                )
                            try:
                                for future in concurrent.futures.as_completed(future_to_cmd):
                                    cmd = future_to_cmd[future]
                                    try:
                                        future.result()
                                        if driver_debug_logger is not None:
                                            driver_debug_logger.event(
                                                "seed_complete",
                                                command=cmd,
                                                n_samples=n_samples,
                                            )
                                    except (
                                        Exception
                                    ) as exc:  # include subprocess.CalledProcessError
                                        errors.append((cmd, exc))
                                        if driver_debug_logger is not None:
                                            driver_debug_logger.event(
                                                "seed_failed",
                                                command=cmd,
                                                n_samples=n_samples,
                                                error_type=type(exc).__name__,
                                                error_message=str(exc),
                                            )
                                    finally:
                                        progress.update(1)
                            finally:
                                if (
                                    shared_esm_components is not None
                                    and shared_esm_components.esm_batch_worker is not None
                                ):
                                    shared_esm_components.esm_batch_worker.close()
                    else:
                        executor_workers = max_workers
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=executor_workers
                        ) as executor:
                            future_to_cmd = {}
                            future_to_cmd = {
                                executor.submit(
                                    _run_seed_with_retries,
                                    tuple(cmd),
                                    env,
                                    safe=args.safe,
                                    max_seed_retries=args.max_seed_retries,
                                    retry_backoff_seconds=args.retry_backoff_seconds,
                                ): tuple(cmd)
                                for cmd in seed_cmds
                            }
                            if driver_debug_logger is not None:
                                for cmd in seed_cmds:
                                    driver_debug_logger.event(
                                        "seed_submitted",
                                        command=cmd,
                                        n_samples=n_samples,
                                        mode="subprocess",
                                    )
                            for future in concurrent.futures.as_completed(future_to_cmd):
                                cmd = future_to_cmd[future]
                                try:
                                    future.result()
                                    if driver_debug_logger is not None:
                                        driver_debug_logger.event(
                                            "seed_complete",
                                            command=cmd,
                                            n_samples=n_samples,
                                        )
                                except (
                                    Exception
                                ) as exc:  # include subprocess.CalledProcessError
                                    errors.append((cmd, exc))
                                    if driver_debug_logger is not None:
                                        driver_debug_logger.event(
                                            "seed_failed",
                                            command=cmd,
                                            n_samples=n_samples,
                                            error_type=type(exc).__name__,
                                            error_message=str(exc),
                                        )
                                finally:
                                    progress.update(1)

                    if errors:
                        for cmd, exc in errors:
                            print(
                                "Seed command failed:", " ".join(cmd), file=sys.stderr
                            )
                            print(exc, file=sys.stderr)
                        raise SystemExit(1)

                    print(f"Completed seeds for n_samples={n_samples}, running ETL")
                    if driver_debug_logger is not None:
                        driver_debug_logger.event("etl_start", n_samples=n_samples)
                    etl_cmd = [
                        sys.executable,
                        "-m",
                        "prospero.runners.etl_results",
                        "--task",
                        args.task,
                        "--results_dirpath",
                        str(batch_dir),
                        "--n_iters",
                        str(n_iters),
                    ]
                    _run_command(etl_cmd, env)
                    if driver_debug_logger is not None:
                        driver_debug_logger.event("etl_complete", n_samples=n_samples)
        finally:
            if driver_debug_logger is not None:
                driver_debug_logger.event("driver_exit")
            heartbeat_stop_event.set()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
