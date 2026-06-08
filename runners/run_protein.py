import sys
import os
import json
import threading
import time
import resource
from prospero.experiments_config import ALPHABETS, WT_SEQUENCES


from prospero.utils import set_seed, get_new_starting_seq, get_new_starting_seq_dshift
from prospero.experiment_tracker import ExperimentTracker
from prospero.inference import ProteinSampler
from prospero.representations.interplm import DEFAULT_INTERPLM_REPO_ID

from prospero.surrogate import (
    Ensemble,
    build_surrogate_model,
    normalize_sequences,
    prepare_shared_esm_components,
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


FROZEN_ESM_SURROGATE_ARCHS = {
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


def _extract_surrogate_artifacts(proxy, args):
    members = []
    for model_idx, model in enumerate(getattr(proxy, "models", [])):
        regressor = getattr(model, "regressor", None)
        if regressor is None or not hasattr(regressor, "coef_"):
            continue

        coef = np.asarray(regressor.coef_)
        intercept = np.asarray(getattr(regressor, "intercept_", 0.0))
        scaler = getattr(model, "scaler", None)

        member_artifact = {
            "model_index": model_idx,
            "surrogate_class": type(model).__name__,
            "regressor_class": type(regressor).__name__,
            "coef": coef.copy(),
            "intercept": intercept.copy(),
        }
        if hasattr(regressor, "alpha"):
            member_artifact["alpha"] = float(regressor.alpha)
        if hasattr(regressor, "fit_intercept"):
            member_artifact["fit_intercept"] = bool(regressor.fit_intercept)
        if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
            member_artifact["scaler_mean"] = np.asarray(scaler.mean_).copy()
            member_artifact["scaler_scale"] = np.asarray(scaler.scale_).copy()
        if hasattr(model, "interplm_layer"):
            member_artifact["interplm_layer"] = int(model.interplm_layer)
        if hasattr(model, "interplm_repo_id"):
            member_artifact["interplm_repo_id"] = str(model.interplm_repo_id)
        if hasattr(model, "interplm_normalized"):
            member_artifact["interplm_normalized"] = bool(model.interplm_normalized)
        members.append(member_artifact)

    if not members:
        return None

    return {
        "surrogate_arch": args.surrogate_arch,
        "members": members,
    }


def _dedupe_preserving_order(sequences):
    seen = set()
    ordered = []
    for sequence in sequences:
        if sequence in seen:
            continue
        seen.add(sequence)
        ordered.append(sequence)
    return ordered


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


def get_parser():
    parser = argparse.ArgumentParser(
        description="Unified Argument Parser for Oracle, Dataset, and Proxy Arguments",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # Experiment arguments
    parser.add_argument("--results_dirpath", type=str)
    parser.add_argument("--n_queries", type=int, default=128)
    parser.add_argument("--seed", type=int, choices=[1, 2, 3, 4, 5], default=1)
    parser.add_argument("--task", type=str, choices=list(WT_SEQUENCES))
    parser.add_argument("--full_deterministic", action="store_true", default=False)
    parser.add_argument("--n_iters", type=int, default=10)

    # Sampler arguments
    parser.add_argument("--resampling_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--alphabet", type=str, default="CHARGE")
    parser.add_argument(
        "--mask_strategy",
        choices=["targeted", "random"],
        default="targeted",
        help="Use standard surrogate-targeted masks or calibrated random masks.",
    )
    parser.add_argument(
        "--sequence_scoring",
        choices=["surrogate", "zero_shot_logp_delta"],
        default="surrogate",
        help=(
            "Score SMC rollouts with the surrogate as usual, or with cumulative "
            "logP(sampled)-logP(original) while still training the surrogate."
        ),
    )
    parser.add_argument(
        "--mask_count_calibration_rounds",
        type=int,
        default=32,
        help=(
            "Number of targeted-mask batches used to calibrate selected mask-count "
            "stats when --mask_strategy=random."
        ),
    )
    parser.add_argument("--kappa_scan", type=float, default=1.0)
    parser.add_argument("--kappa_guidance", type=float, default=0.1)
    parser.add_argument("--n_checks_multiplier", type=int, default=16)
    parser.add_argument("--min_corruptions", type=int, default=3)
    parser.add_argument("--max_corruptions", type=int, default=10)

    # Proxy arguments
    parser.add_argument("--num_model_max_epochs", type=int, default=3000)
    parser.add_argument("--ensemble_size", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--epochs_per_valid", type=int, default=1)
    parser.add_argument("--proxy_batch_size", type=int, default=256)
    parser.add_argument(
        "--surrogate_arch",
        type=str,
        choices=[
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
        ],
        default="cnn",
    )
    parser.add_argument(
        "--esm_model_name",
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
    )
    parser.add_argument("--esm_mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--esm_mlp_dropout", type=float, default=0.25)
    parser.add_argument("--esm_cnn_projection_dim", type=int, default=None)
    parser.add_argument("--esm_cnn_use_layernorm", action="store_true", default=False)
    parser.add_argument("--esm_cnn_concat_one_hot", action="store_true", default=False)
    parser.add_argument(
        "--esm_max_length",
        type=int,
        default=None,
        help="Optional max tokenized sequence length for ESM inputs",
    )
    parser.add_argument(
        "--esm_representation_layer",
        type=int,
        default=2,
        help=(
            "Hidden-state index used by frozen_esm_layer_cls_ridge. "
            "HuggingFace hidden_states[0] is embeddings; this defaults to "
            "the repo's existing layer-2 convention."
        ),
    )
    parser.add_argument(
        "--esm_truncate_to_representation_layer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Truncate HF ESM to the blocks needed for esm_representation_layer.",
    )
    parser.add_argument(
        "--esm_lora_adapter_path",
        type=str,
        default=None,
        help="Optional local LoRA adapter directory for frozen_esm_layer_cls_ridge.",
    )
    parser.add_argument(
        "--ridge_alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength for frozen_esm_flat_ridge.",
    )
    parser.add_argument(
        "--ridge_fit_intercept",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether sklearn linear/ridge surrogate fits an intercept.",
    )
    parser.add_argument("--disable-esm-cache", action="store_true", default=False)
    parser.add_argument(
        "--interplm_layer",
        type=int,
        default=2,
        help="ESM hidden layer used to derive mean-pooled InterPLM features.",
    )
    parser.add_argument(
        "--interplm_repo_id",
        type=str,
        default=DEFAULT_INTERPLM_REPO_ID,
        help="Hugging Face repo containing the InterPLM SAE weights.",
    )
    parser.add_argument(
        "--interplm_normalized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to load normalized InterPLM SAE weights.",
    )
    parser.add_argument(
        "--sae_token_chunk_size",
        type=int,
        default=1024,
        help="Chunk size for pooling SAE activations over residue tokens.",
    )
    parser.add_argument(
        "--low_rank_positional_rank",
        type=int,
        default=16,
        help="Rank for low-rank positional surrogate.",
    )
    parser.add_argument(
        "--low_rank_positional_l2",
        type=float,
        default=1e-4,
        help="L2 coefficient for low-rank positional factors/projection.",
    )
    parser.add_argument(
        "--low_rank_positional_lr",
        type=float,
        default=None,
        help="Optional optimizer LR override for low-rank positional surrogate.",
    )
    parser.add_argument(
        "--low_rank_positional_repr_batch_size",
        type=int,
        default=None,
        help="Representation extraction micro-batch for low-rank positional surrogate.",
    )
    parser.add_argument(
        "--low_rank_positional_input",
        type=str,
        choices=["esm", "sae", "esm_sae_concat"],
        default="sae",
        help="Input representation used by low-rank positional surrogate.",
    )
    parser.add_argument("--debug-events", action="store_true", default=False)
    parser.add_argument("--debug-heartbeat-seconds", type=float, default=15.0)
    parser.add_argument(
        "--initial_train_top_k",
        type=int,
        default=None,
        help=(
            "Restrict the initial training split to the top-k sequences by "
            "training score before the Prospero loop starts. Validation is "
            "left unchanged and collected sequences are added normally."
        ),
    )

    return parser


def run_iter(args, logger, shared_esm_components=None, shared_oracle=None):
    seed = args.seed
    set_seed(seed, args.full_deterministic)
    logger.info(f"Starting seed {seed}")

    save_dir = os.path.join(args.results_dirpath, args.task)
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
            surrogate_arch=args.surrogate_arch,
            n_queries=args.n_queries,
            n_iters=args.n_iters,
            alphabet=args.alphabet,
            mask_strategy=args.mask_strategy,
            sequence_scoring=args.sequence_scoring,
            results_dirpath=args.results_dirpath,
            disable_esm_cache=args.disable_esm_cache,
        )
        debug_logger.start_heartbeat()

    wt_sequence = WT_SEQUENCES[args.task]
    if shared_oracle is None and debug_logger is not None:
        debug_logger.set_phase("oracle_load_start")
    oracle = shared_oracle if shared_oracle is not None else get_landscape(args.task)
    if shared_oracle is None and debug_logger is not None:
        debug_logger.set_phase("oracle_load_complete")
    dataset = RegressionDataset(args.task)
    if args.initial_train_top_k is not None:
        if args.initial_train_top_k <= 0:
            raise ValueError("--initial_train_top_k must be positive")
        if args.initial_train_top_k > len(dataset.train_scores):
            raise ValueError(
                f"--initial_train_top_k={args.initial_train_top_k} exceeds "
                f"initial train size {len(dataset.train_scores)}"
            )
        top_indices = np.argsort(dataset.train_scores)[::-1][: args.initial_train_top_k]
        dataset.train = dataset.train[top_indices]
        dataset.train_scores = dataset.train_scores[top_indices]
        dataset.train_added = len(dataset.train)
    if debug_logger is not None:
        debug_logger.event(
            "dataset_loaded",
            train_size=len(dataset.train),
            valid_size=len(dataset.valid),
            initial_train_top_k=args.initial_train_top_k,
        )
    if args.disable_esm_cache:
        args.cache_allowed_sequences = set()
        args.cache_allowed_sequences_ordered = []
        args.dataset_cache_task = None
        if debug_logger is not None:
            debug_logger.event("esm_cache_disabled")
    else:
        ordered_initial_dataset_sequences = _dedupe_preserving_order(
            normalize_sequences(list(dataset.train) + list(dataset.valid))
        )
        args.cache_allowed_sequences = set(ordered_initial_dataset_sequences)
        args.cache_allowed_sequences_ordered = ordered_initial_dataset_sequences
        args.dataset_cache_task = args.task
        if debug_logger is not None:
            debug_logger.event(
                "esm_cache_ready",
                cache_allowed_sequences=len(ordered_initial_dataset_sequences),
            )

    try:
        if (
            shared_esm_components is None
            and args.surrogate_arch in FROZEN_ESM_SURROGATE_ARCHS
        ):
            if debug_logger is not None:
                debug_logger.set_phase("shared_esm_prepare_start")
            shared_esm_components = prepare_shared_esm_components(args)
            if debug_logger is not None:
                debug_logger.set_phase("shared_esm_prepare_complete")

        if debug_logger is not None:
            debug_logger.set_phase("initial_surrogate_build")
        proxy = Ensemble(
            [
                build_surrogate_model(
                    len(wt_sequence),
                    args,
                    shared_esm_components=shared_esm_components,
                )
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

        alphabet = ALPHABETS[args.alphabet]

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
        mask_count_stats = None
        if args.mask_strategy == "random":
            calibration_sampler = ProteinSampler(model, tokenizer_oadm, alphabet)
            logger.info("Calibrating random mask-count distribution from targeted masking")
            mask_counts = calibration_sampler.collect_targeted_mask_counts(
                starting_sequence,
                proxy,
                args.min_corruptions,
                args.max_corruptions,
                args.batch_size,
                args.n_checks_multiplier,
                args.kappa_scan,
                args.mask_count_calibration_rounds,
            )
            if mask_counts.size == 0:
                raise ValueError("Mask-count calibration produced no counts.")
            mask_count_stats = {
                "mean": float(np.mean(mask_counts)),
                "std": float(np.std(mask_counts, ddof=0)),
                "min": int(np.min(mask_counts)),
                "max": int(np.max(mask_counts)),
                "n": int(mask_counts.size),
                "source": "selected_targeted_masks_calibrated_at_seed_start",
                "calibration_rounds": int(args.mask_count_calibration_rounds),
            }
            logger.info(
                "Mask-count stats: mean=%s std=%s min=%s max=%s n=%s",
                mask_count_stats["mean"],
                mask_count_stats["std"],
                mask_count_stats["min"],
                mask_count_stats["max"],
                mask_count_stats["n"],
            )
            if debug_logger is not None:
                debug_logger.event("mask_count_calibration_complete", **mask_count_stats)

        for e in range(args.n_iters):
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
            while len(sequences) < args.n_queries:  # n_queries is K
                generation_round += 1
                if debug_logger is not None:
                    debug_logger.event(
                        "generation_round_start",
                        iteration=iteration,
                        generation_round=generation_round,
                        current_sequences=len(sequences),
                    )
                # Run the requested masking/scoring ablation.
                if args.mask_strategy == "targeted" and args.sequence_scoring == "surrogate":
                    sampler.generate_raa_from_alanine_scan(
                        proxy,
                        starting_sequence,
                        args.batch_size,
                        args.resampling_steps,
                        args.min_corruptions,
                        args.max_corruptions,
                        args.kappa_scan,
                        args.n_checks_multiplier,
                        args.kappa_guidance,
                    )
                elif args.mask_strategy == "random" and args.sequence_scoring == "surrogate":
                    if mask_count_stats is None:
                        raise RuntimeError("Random masking requires mask-count stats.")
                    sampler.generate_raa_from_random_masks(
                        proxy,
                        starting_sequence,
                        args.batch_size,
                        args.resampling_steps,
                        mask_count_stats["mean"],
                        mask_count_stats["std"],
                        mask_count_stats["min"],
                        mask_count_stats["max"],
                        args.kappa_guidance,
                    )
                elif args.mask_strategy == "targeted" and args.sequence_scoring == "zero_shot_logp_delta":
                    sampler.generate_zero_shot_from_targeted_scan(
                        proxy,
                        starting_sequence,
                        args.batch_size,
                        args.resampling_steps,
                        args.min_corruptions,
                        args.max_corruptions,
                        args.kappa_scan,
                        args.n_checks_multiplier,
                    )
                elif args.mask_strategy == "random" and args.sequence_scoring == "zero_shot_logp_delta":
                    if mask_count_stats is None:
                        raise RuntimeError("Random masking requires mask-count stats.")
                    sampler.generate_zero_shot_from_random_masks(
                        starting_sequence,
                        args.batch_size,
                        args.resampling_steps,
                        mask_count_stats["mean"],
                        mask_count_stats["std"],
                        mask_count_stats["min"],
                        mask_count_stats["max"],
                    )
                else:
                    raise ValueError(
                        f"Unsupported ablation: mask_strategy={args.mask_strategy}, "
                        f"sequence_scoring={args.sequence_scoring}"
                    )
                # This method is inherited from parent Sampler class
                top_sequences = sampler.get_top_sequences(args.n_queries, ref_sequences)
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

            sequences = sequences[: args.n_queries]
            assert len(sequences) == args.n_queries
            if debug_logger is not None:
                debug_logger.event(
                    "candidate_batch_ready",
                    iteration=iteration,
                    sequence_count=len(sequences),
                )

            # eval candidate sequences
            if debug_logger is not None:
                debug_logger.set_phase("oracle_scoring", iteration=iteration)
            if not args.task.startswith("D_SHIFT"):
                scores = oracle.get_fitness(np.array(sequences)).tolist()
            else:
                scores = oracle.get_fitness(sequences).tolist()
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
            surrogate_artifacts = _extract_surrogate_artifacts(proxy, args)
            if surrogate_artifacts is not None:
                surrogate_artifacts["mask_strategy"] = args.mask_strategy
                surrogate_artifacts["sequence_scoring"] = args.sequence_scoring
                if mask_count_stats is not None:
                    surrogate_artifacts["mask_count_stats"] = mask_count_stats
                exp_tracker.attach_surrogate_artifacts(iteration, surrogate_artifacts)
            starting_sequence = (
                get_new_starting_seq(dataset)
                if not args.task.startswith("D_SHIFT")
                else get_new_starting_seq_dshift(dataset, args.task)
            )
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
                    build_surrogate_model(
                        len(wt_sequence),
                        args,
                        shared_esm_components=shared_esm_components,
                    )
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
            if iteration < args.n_iters:
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
