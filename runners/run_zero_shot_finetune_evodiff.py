import argparse
import json
import os
import sys
import time
from argparse import ArgumentDefaultsHelpFormatter
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from prospero.dataset import RegressionDataset
from prospero.debug_trace import JsonlGzTraceWriter
from prospero.evodiff_finetune import (
    bottom_quantile_rank_advantages,
    group_relative_advantages,
    standardized_advantages,
    train_evodiff_reward_weighted_nll,
)
from prospero.experiment_tracker import ExperimentTracker
from prospero.experiments_config import ALPHABETS, WT_SEQUENCES
from prospero.inference import ProteinSampler
from prospero.landscapes import get_landscape
from prospero.utils import get_new_starting_seq, get_new_starting_seq_dshift, set_seed

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Zero-shot ProSpero with online reward-weighted EvoDiff fine-tuning.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_dirpath", required=True)
    parser.add_argument("--task", default="AAV", choices=list(WT_SEQUENCES))
    parser.add_argument("--seed", type=int, choices=[1, 2, 3, 4, 5], default=1)
    parser.add_argument("--n_queries", type=int, default=128)
    parser.add_argument("--n_iters", type=int, default=10)
    parser.add_argument("--full_deterministic", action="store_true", default=False)

    parser.add_argument("--resampling_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--alphabet", type=str, default="CHARGE")
    parser.add_argument(
        "--mask_strategy",
        choices=[
            "calibrated_random",
            "random",
            "middle_entropy",
            "seed_grow",
            "mixed_explore_exploit",
        ],
        default="calibrated_random",
        help=(
            "Zero-shot mask selector. calibrated_random preserves the older "
            "selected-targeted-count calibration route; the other strategies "
            "use a fixed --mask_budget and no training data."
        ),
    )
    parser.add_argument(
        "--mask_budget",
        type=int,
        default=None,
        help="Fixed number of masked positions for random/middle_entropy/seed_grow.",
    )
    parser.add_argument(
        "--entropy_quantile",
        type=float,
        default=0.5,
        help="Quantile H* used by middle-entropy scoring.",
    )
    parser.add_argument(
        "--entropy_sigma",
        type=float,
        default=None,
        help="Optional entropy score bandwidth. Defaults to entropy std.",
    )
    parser.add_argument("--seed_grow_alpha", type=float, default=1.0)
    parser.add_argument("--seed_grow_beta", type=float, default=1.0)
    parser.add_argument(
        "--seed_grow_coupling_tau",
        type=float,
        default=4.0,
        help="Length-scale for sequence-proximity coupling in seed-and-grow.",
    )
    parser.add_argument(
        "--smc_vocab",
        choices=["cluster", "full"],
        default="cluster",
        help="Vocabulary used during the main SMC unmasking step. Rollouts already use full 20-AA.",
    )
    parser.add_argument(
        "--zero_shot_generation_mode",
        choices=["smc_rollout", "no_rollout_sequential"],
        default="smc_rollout",
        help="Use rollout-scored SMC candidates or terminal-only sequential samples without rollouts.",
    )
    parser.add_argument("--min_corruptions", type=int, default=3)
    parser.add_argument("--max_corruptions", type=int, default=10)
    parser.add_argument("--kappa_scan", type=float, default=1.0)
    parser.add_argument("--n_checks_multiplier", type=int, default=16)

    parser.add_argument("--mask_count_mean", type=float, default=None)
    parser.add_argument("--mask_count_std", type=float, default=None)
    parser.add_argument("--mask_count_min", type=int, default=None)
    parser.add_argument("--mask_count_max", type=int, default=None)
    parser.add_argument(
        "--mask_count_calibration_rounds",
        type=int,
        default=32,
        help="Number of targeted-mask batches used to estimate selected mask-count stats.",
    )
    parser.add_argument(
        "--calibration_surrogate_arch",
        default="one_hot_ridge",
        choices=["one_hot_ridge"],
        help="Fast surrogate used only to estimate selected targeted-mask counts when stats are not supplied.",
    )
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument(
        "--ridge_fit_intercept",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--finetune_epochs", type=int, default=5)
    parser.add_argument("--finetune_lr", type=float, default=1e-5)
    parser.add_argument("--lambda_kl", type=float, default=2.0)
    parser.add_argument("--finetune_batch_size", type=int, default=16)
    parser.add_argument(
        "--pre_finetune_train_top_k",
        type=int,
        default=0,
        help=(
            "Before optimization, fine-tune EvoDiff on the top-k sequences from "
            "the initial training split. Disabled when 0."
        ),
    )
    parser.add_argument(
        "--finetune_replay",
        choices=["all", "latest"],
        default="all",
        help="Use all evaluated candidates so far or only the latest oracle batch.",
    )
    parser.add_argument(
        "--finetune_after_final",
        action="store_true",
        default=False,
        help="Also fine-tune after the last oracle evaluation, useful only for diagnostics/checkpointing.",
    )
    parser.add_argument(
        "--save_finetuned_model",
        action="store_true",
        default=False,
        help="Save the final online fine-tuned EvoDiff state dict. Off by default because runs can be large.",
    )
    parser.add_argument(
        "--reward_mode",
        choices=[
            "rank",
            "standardized_advantage",
            "grpo_advantage",
            "bottom_quantile_negative",
        ],
        default="rank",
        help="Fine-tuning sequence weighting objective.",
    )
    parser.add_argument(
        "--advantage_baseline",
        choices=["moving_starting_sequence"],
        default="moving_starting_sequence",
        help="For standardized_advantage: round 1 uses WT, later rounds use the current generation starting sequence.",
    )
    parser.add_argument("--advantage_clip", type=float, default=2.0)
    parser.add_argument("--negative_weight", type=float, default=0.25)
    parser.add_argument("--bottom_quantile", type=float, default=0.25)
    parser.add_argument(
        "--debug_generation_trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write scalar-only SMC/masking/search traces as compressed JSONL.",
    )
    parser.add_argument(
        "--debug_trace_dir",
        default=None,
        help="Optional trace directory. Defaults to <seed output dir>/debug_traces.",
    )

    return parser


def _stats_are_complete(args):
    return all(
        value is not None
        for value in (
            args.mask_count_mean,
            args.mask_count_std,
            args.mask_count_min,
            args.mask_count_max,
        )
    )


def _calibrate_mask_count_stats(args, dataset, sampler, starting_sequence):
    from prospero.surrogate import Ensemble, build_surrogate_model

    logger.info("Calibrating selected targeted-mask count distribution")
    calibration_args = argparse.Namespace(**vars(args))
    calibration_args.surrogate_arch = args.calibration_surrogate_arch
    calibration_args.ensemble_size = 1
    calibration_args.num_model_max_epochs = 3000
    calibration_args.lr = 1e-4
    calibration_args.weight_decay = 1e-4
    calibration_args.patience = 10
    calibration_args.epochs_per_valid = 1
    calibration_args.proxy_batch_size = 256

    proxy = Ensemble(
        [
            build_surrogate_model(
                len(WT_SEQUENCES[args.task]),
                calibration_args,
                shared_esm_components=None,
            )
        ]
    )
    proxy.train(dataset)
    counts = sampler.collect_targeted_mask_counts(
        starting_sequence,
        proxy,
        args.min_corruptions,
        args.max_corruptions,
        args.batch_size,
        args.n_checks_multiplier,
        args.kappa_scan,
        args.mask_count_calibration_rounds,
    )
    if counts.size == 0:
        raise ValueError("Mask-count calibration produced no counts.")
    stats = {
        "mean": float(np.mean(counts)),
        "std": float(np.std(counts, ddof=0)),
        "min": int(np.min(counts)),
        "max": int(np.max(counts)),
        "n": int(counts.size),
        "source": "selected_targeted_masks_calibrated_with_one_hot_ridge",
        "calibration_rounds": int(args.mask_count_calibration_rounds),
    }
    logger.info(
        "Mask-count stats: mean=%s std=%s min=%s max=%s n=%s",
        stats["mean"],
        stats["std"],
        stats["min"],
        stats["max"],
        stats["n"],
    )
    return stats


def _provided_mask_count_stats(args):
    return {
        "mean": float(args.mask_count_mean),
        "std": float(args.mask_count_std),
        "min": int(args.mask_count_min),
        "max": int(args.mask_count_max),
        "source": "cli",
    }


def _score_sequence(oracle, task, sequence):
    if not task.startswith("D_SHIFT"):
        return float(oracle.get_fitness(np.array([sequence]))[0])
    return float(oracle.get_fitness([sequence])[0])


def _write_plots(save_path, metrics_path, plots_dir):
    try:
        import matplotlib.pyplot as plt
        import pickle
    except Exception as exc:  # pragma: no cover - diagnostic plotting only
        logger.warning("Skipping plots; matplotlib/pickle import failed: %s", exc)
        return

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if Path(save_path).exists():
        with open(save_path, "rb") as handle:
            results = pickle.load(handle)
        iters = sorted(k for k in results if isinstance(k, int))
        if iters:
            perf = [float(results[i]["Performance"]) for i in iters]
            best = [float(results[i]["Best score"]) for i in iters]
            median = [float(results[i]["Median performance"]) for i in iters]
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            ax.plot(iters, perf, marker="o", label="mean fitness")
            ax.plot(iters, median, marker="o", label="median fitness")
            ax.plot(iters, best, marker="o", label="best fitness")
            ax.set_xlabel("optimization round")
            ax.set_ylabel("oracle fitness")
            ax.set_title("Zero-shot EvoDiff FT fitness over time")
            ax.legend()
            fig.savefig(plots_dir / "fitness_curve.png", dpi=180)
            plt.close(fig)

    metric_events = []
    metrics_path = Path(metrics_path)
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                if event.get("event") == "evodiff_finetune_epoch":
                    metric_events.append(event)
    if metric_events:
        x = np.arange(1, len(metric_events) + 1)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        axes[0, 0].plot(x, [m["weighted_nll"] for m in metric_events], label="weighted NLL")
        axes[0, 0].plot(x, [m["kl"] for m in metric_events], label="KL")
        axes[0, 0].plot(x, [m["loss"] for m in metric_events], label="total")
        axes[0, 0].set_title("loss terms")
        axes[0, 0].legend()
        axes[0, 1].plot(x, [m["grad_norm"] for m in metric_events], label="mean grad norm")
        axes[0, 1].plot(x, [m["max_grad_norm"] for m in metric_events], label="max grad norm")
        axes[0, 1].set_title("gradient norms")
        axes[0, 1].legend()
        axes[1, 0].plot(x, [m["update_norm"] for m in metric_events], label="update norm")
        axes[1, 0].plot(x, [m["relative_update_norm"] for m in metric_events], label="relative update")
        axes[1, 0].set_title("distance from frozen base")
        axes[1, 0].legend()
        axes[1, 1].plot(x, [m["seconds"] for m in metric_events], label="epoch seconds")
        axes[1, 1].set_title("fine-tuning wall time")
        axes[1, 1].legend()
        for ax in axes.flat:
            ax.set_xlabel("fine-tune epoch event")
        fig.savefig(plots_dir / "finetune_diagnostics.png", dpi=180)
        plt.close(fig)


def run_iter(args, logger):
    set_seed(args.seed, args.full_deterministic)
    logger.info("Starting zero-shot seed %s", args.seed)

    save_dir = os.path.join(args.results_dirpath, args.task)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"seed_{args.seed}.pkl")
    metadata_path = os.path.join(save_dir, f"seed_{args.seed}.zero_shot_metadata.json")
    finetune_dir = os.path.join(save_dir, f"seed_{args.seed}.evodiff_finetune")
    finetune_metrics_path = os.path.join(finetune_dir, "evodiff_finetune_metrics.jsonl")
    plots_dir = os.path.join(finetune_dir, "plots")

    wt_sequence = WT_SEQUENCES[args.task]
    oracle = get_landscape(args.task)
    dataset = RegressionDataset(args.task)
    alphabet = ALPHABETS[args.alphabet]

    from evodiff.pretrained import OA_DM_38M  # type: ignore[reportMissingImports]

    model, _, tokenizer_oadm, _ = OA_DM_38M()
    base_model, _, _, _ = OA_DM_38M()
    model = model.cuda()
    base_model = base_model.cuda()
    model.eval()
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad_(False)

    sampler = ProteinSampler(model, tokenizer_oadm, alphabet)
    trace_writer = None
    if args.debug_generation_trace:
        trace_dir = args.debug_trace_dir or os.path.join(save_dir, "debug_traces")
        trace_path = os.path.join(trace_dir, f"seed_{args.seed}.events.jsonl.gz")
        trace_writer = JsonlGzTraceWriter(trace_path)
        sampler.set_trace_writer(trace_writer)
        logger.info("Writing generation trace: %s", trace_path)
    starting_sequence = wt_sequence
    if args.mask_strategy == "calibrated_random":
        mask_count_stats = (
            _provided_mask_count_stats(args)
            if _stats_are_complete(args)
            else _calibrate_mask_count_stats(args, dataset, sampler, starting_sequence)
        )
    else:
        if args.mask_budget is None:
            raise ValueError("--mask_budget is required for fixed-budget mask strategies.")
        if args.mask_budget <= 0:
            raise ValueError("--mask_budget must be positive.")
        mask_count_stats = None

    entropy_metadata = None
    if args.mask_strategy in {"middle_entropy", "seed_grow", "mixed_explore_exploit"}:
        tokenized_seq = tokenizer_oadm.tokenize([starting_sequence])
        maskable_tokens = np.array(list(sampler.token_to_cluster))
        maskable_ids = np.nonzero(np.isin(tokenized_seq, maskable_tokens))[0]
        entropies = sampler.compute_masked_position_entropies(
            starting_sequence,
            maskable_ids,
        )
        _, entropy_metadata = sampler.middle_entropy_scores(
            entropies,
            quantile=args.entropy_quantile,
            sigma=args.entropy_sigma,
        )
    metadata = {
        "task": args.task,
        "seed": args.seed,
        "n_queries": args.n_queries,
        "n_iters": args.n_iters,
        "alphabet": args.alphabet,
        "mask_strategy": args.mask_strategy,
        "mask_budget": args.mask_budget,
        "score": "sum(logP(sampled_residue)-logP(original_residue))",
        "constrained_unmasking_distribution": (
            "full 20-AA softmax" if args.smc_vocab == "full" else "alphabet cluster softmax"
        ),
        "smc_vocab": args.smc_vocab,
        "zero_shot_generation_mode": args.zero_shot_generation_mode,
        "rollout_distribution": (
            "full 20-AA softmax"
            if args.zero_shot_generation_mode == "smc_rollout"
            else "disabled"
        ),
        "resampling": (
            "zero_shot_score_times_inverse_perplexity"
            if args.zero_shot_generation_mode == "smc_rollout"
            else "disabled"
        ),
        "debug_generation_trace": bool(args.debug_generation_trace),
        "mask_count_stats": mask_count_stats,
        "entropy_metadata": entropy_metadata,
        "seed_grow": {
            "base_score": "middle_entropy",
            "coupling": "sequence_proximity_exp_distance",
            "alpha": args.seed_grow_alpha,
            "beta": args.seed_grow_beta,
            "coupling_tau": args.seed_grow_coupling_tau,
        }
        if args.mask_strategy in {"seed_grow", "mixed_explore_exploit"}
        else None,
        "mixed_explore_exploit": {
            "exploit": "half of K from positive online position reward, seed_grow fallback before feedback",
            "explore": "one middle-entropy position plus remaining anti-collapse random positions",
            "position_reward": "round-centered oracle fitness assigned to mutated absolute positions",
            "anti_collapse": "random channel weighted by 1/sqrt(1 + position_mask_count)",
        }
        if args.mask_strategy == "mixed_explore_exploit"
        else None,
        "evodiff_finetuning": {
            "objective": "reward_weighted_masked_token_nll_plus_kl_current_to_frozen_base",
            "reward_mode": args.reward_mode,
            "advantage_baseline": args.advantage_baseline,
            "advantage_clip": args.advantage_clip,
            "negative_weight": args.negative_weight,
            "rank_weighting": "normalized oracle-fitness rank, lowest=0 highest=1",
            "standardized_advantage": (
                "adv=clip((oracle_score - baseline_fitness) / std(current_round_scores), "
                "-advantage_clip, advantage_clip); positive adv reinforces, negative adv penalizes with negative_weight"
            ),
            "grpo_advantage": (
                "adv=clip((oracle_score - mean(current_round_scores)) / std(current_round_scores), "
                "-advantage_clip, advantage_clip); normalized within each oracle query batch"
            ),
            "bottom_quantile_negative": (
                "positive normalized rank weights, but bottom_quantile of each oracle batch is assigned -1 "
                "and penalized with negative_weight"
            ),
            "kl": "KL(p_theta(.|x_t,t) || p_base(.|x_t,t)) over AA tokens at masked positions",
            "lambda_kl": args.lambda_kl,
            "epochs_per_round": args.finetune_epochs,
            "lr": args.finetune_lr,
            "batch_size": args.finetune_batch_size,
            "replay": args.finetune_replay,
            "training_mask_budget": args.mask_budget,
            "training_corruption": "uniform random fixed-budget masks",
            "pre_finetune_train_top_k": args.pre_finetune_train_top_k,
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    exp_tracker = ExperimentTracker(
        logger,
        deepcopy(dataset),
        wt_sequence,
        best_percentile=0.95,
    )
    replay_sequences = []
    replay_scores = []
    replay_advantages = []
    replay_reward_metadata = []

    if args.pre_finetune_train_top_k > 0:
        if args.mask_budget is None:
            raise ValueError("--pre_finetune_train_top_k requires --mask_budget.")
        k = min(int(args.pre_finetune_train_top_k), len(dataset.train_scores))
        if k <= 0:
            raise ValueError("--pre_finetune_train_top_k must be positive when enabled.")
        top_indices = np.argsort(np.asarray(dataset.train_scores, dtype=float))[::-1][:k]
        pretrain_sequences = [
            "".join(str(token) for token in np.asarray(dataset.train[idx]).tolist())
            for idx in top_indices
        ]
        pretrain_scores = [float(dataset.train_scores[idx]) for idx in top_indices]
        logger.info(
            "Pre-fine-tuning EvoDiff on top %s initial train sequences for %s epochs",
            k,
            args.finetune_epochs,
        )
        pretrain_start = time.perf_counter()
        pretrain_metrics = train_evodiff_reward_weighted_nll(
            model,
            base_model,
            tokenizer_oadm,
            alphabet,
            pretrain_sequences,
            pretrain_scores,
            finetune_dir,
            round_idx=0,
            epochs=args.finetune_epochs,
            lr=args.finetune_lr,
            lambda_kl=args.lambda_kl,
            batch_size=args.finetune_batch_size,
            mask_budget=args.mask_budget,
        )
        pretrain_seconds = time.perf_counter() - pretrain_start
        metadata["evodiff_finetuning"]["pre_finetune"] = {
            "source": "initial_train_split",
            "top_k": int(k),
            "seconds": float(pretrain_seconds),
            "score_min": float(np.min(pretrain_scores)),
            "score_max": float(np.max(pretrain_scores)),
            "last_epoch": pretrain_metrics[-1] if pretrain_metrics else None,
        }
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        logger.info(
            "Pre-fine-tuned EvoDiff in %.2f seconds; last metrics=%s",
            pretrain_seconds,
            pretrain_metrics[-1] if pretrain_metrics else None,
        )
        _write_plots(save_path, finetune_metrics_path, plots_dir)

    try:
        for e in range(args.n_iters):
            iteration = e + 1
            model.eval()
            round_starting_sequence = starting_sequence
            round_baseline_fitness = _score_sequence(oracle, args.task, round_starting_sequence)
            logger.info("Zero-shot iteration %s/%s", iteration, args.n_iters)
            sequences = []
            ref_sequences = list(dataset.train) + list(dataset.valid)
            generation_round = 0
            while len(sequences) < args.n_queries:
                generation_round += 1
                sampler.set_trace_context(
                    task=args.task,
                    seed=args.seed,
                    n_queries=args.n_queries,
                    optimization_round=iteration,
                    generation_round=generation_round,
                    method="ft_kl2" if float(args.lambda_kl) == 2.0 else f"ft_kl{args.lambda_kl}",
                    lambda_kl=float(args.lambda_kl),
                    mask_strategy=args.mask_strategy,
                    mask_budget=args.mask_budget,
                )
                logger.info(
                    "Generation round %s, current candidates %s/%s",
                    generation_round,
                    len(sequences),
                    args.n_queries,
                )
                if args.mask_strategy == "calibrated_random":
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
                    sampler.generate_zero_shot_from_fixed_masks(
                        starting_sequence,
                        args.batch_size,
                        args.resampling_steps,
                        args.mask_budget,
                        args.mask_strategy,
                        entropy_quantile=args.entropy_quantile,
                        entropy_sigma=args.entropy_sigma,
                        seed_grow_alpha=args.seed_grow_alpha,
                        seed_grow_beta=args.seed_grow_beta,
                        coupling_tau=args.seed_grow_coupling_tau,
                        smc_vocab=args.smc_vocab,
                        generation_mode=args.zero_shot_generation_mode,
                    )
                top_sequences = sampler.get_top_sequences(args.n_queries, ref_sequences)
                sequences += top_sequences
                ref_sequences += sequences

            sequences = sequences[: args.n_queries]
            if len(sequences) != args.n_queries:
                raise RuntimeError(
                    f"Generated {len(sequences)} sequences, expected {args.n_queries}."
                )

            if not args.task.startswith("D_SHIFT"):
                scores = oracle.get_fitness(np.array(sequences)).tolist()
            else:
                scores = oracle.get_fitness(sequences).tolist()
            round_advantages = None
            round_reward_metadata = None
            if args.reward_mode == "standardized_advantage":
                round_advantages, round_reward_metadata = standardized_advantages(
                    scores,
                    baseline=round_baseline_fitness,
                    scale_scores=scores,
                    clip=args.advantage_clip,
                )
                round_reward_metadata.update(
                    {
                        "baseline_mode": args.advantage_baseline,
                        "baseline_sequence": round_starting_sequence,
                        "negative_weight": float(args.negative_weight),
                        "advantage_clip": float(args.advantage_clip),
                        "round": int(iteration),
                    }
                )
            elif args.reward_mode == "grpo_advantage":
                round_advantages, round_reward_metadata = group_relative_advantages(
                    scores,
                    clip=args.advantage_clip,
                )
                round_reward_metadata.update(
                    {
                        "negative_weight": float(args.negative_weight),
                        "advantage_clip": float(args.advantage_clip),
                        "round": int(iteration),
                    }
                )
            elif args.reward_mode == "bottom_quantile_negative":
                round_advantages, round_reward_metadata = bottom_quantile_rank_advantages(
                    scores,
                    bottom_quantile=args.bottom_quantile,
                )
                round_reward_metadata.update(
                    {
                        "negative_weight": float(args.negative_weight),
                        "bottom_quantile": float(args.bottom_quantile),
                        "round": int(iteration),
                    }
                )
            for query_idx, (sequence, score) in enumerate(zip(sequences, scores), start=1):
                sampler.trace_event(
                    "oracle_query",
                    optimization_round=iteration,
                    query_rank=int(query_idx),
                    sequence=sequence,
                    oracle_score=float(score),
                    baseline_fitness=float(round_baseline_fitness),
                    advantage=float(round_advantages[query_idx - 1]) if round_advantages is not None else None,
                )
            if args.mask_strategy == "mixed_explore_exploit":
                sampler.update_mask_position_rewards(
                    round_starting_sequence,
                    sequences,
                    scores,
                )
            dataset.add((sequences, scores))
            replay_sequences.extend(sequences)
            replay_scores.extend(scores)
            if round_advantages is not None:
                replay_advantages.extend([float(value) for value in round_advantages])
                replay_reward_metadata.append(round_reward_metadata)
            exp_tracker.calculate_top_n_metrics((sequences, scores), iteration, n=100)
            exp_tracker.exp_results[iteration]["Zero-shot"] = metadata
            if round_reward_metadata is not None:
                exp_tracker.exp_results[iteration]["Reward"] = round_reward_metadata
            exp_tracker.save_results(save_path)
            _write_plots(save_path, finetune_metrics_path, plots_dir)

            starting_sequence = (
                get_new_starting_seq(dataset)
                if not args.task.startswith("D_SHIFT")
                else get_new_starting_seq_dshift(dataset, args.task)
            )

            should_finetune = iteration < args.n_iters or args.finetune_after_final
            if should_finetune:
                if args.finetune_replay == "latest":
                    train_sequences = sequences
                    train_scores = scores
                    train_weights = round_advantages
                    train_reward_metadata = round_reward_metadata
                else:
                    train_sequences = replay_sequences
                    train_scores = replay_scores
                    train_weights = (
                        np.asarray(replay_advantages, dtype=np.float32)
                        if args.reward_mode
                        in {
                            "standardized_advantage",
                            "grpo_advantage",
                            "bottom_quantile_negative",
                        }
                        else None
                    )
                    train_reward_metadata = (
                        {
                            "replay_rounds": replay_reward_metadata,
                            "n_replay_advantages": int(len(replay_advantages)),
                        }
                        if args.reward_mode
                        in {
                            "standardized_advantage",
                            "grpo_advantage",
                            "bottom_quantile_negative",
                        }
                        else None
                    )
                logger.info(
                    "Fine-tuning EvoDiff after iteration %s on %s sequences for %s epochs",
                    iteration,
                    len(train_sequences),
                    args.finetune_epochs,
                )
                finetune_start = time.perf_counter()
                metrics = train_evodiff_reward_weighted_nll(
                    model,
                    base_model,
                    tokenizer_oadm,
                    alphabet,
                    train_sequences,
                    train_scores,
                    finetune_dir,
                    round_idx=iteration,
                    epochs=args.finetune_epochs,
                    lr=args.finetune_lr,
                    lambda_kl=args.lambda_kl,
                    batch_size=args.finetune_batch_size,
                    mask_budget=args.mask_budget,
                    reward_mode=args.reward_mode,
                    sequence_weights=train_weights,
                    negative_weight=args.negative_weight,
                    reward_metadata=train_reward_metadata,
                )
                finetune_seconds = time.perf_counter() - finetune_start
                logger.info(
                    "Fine-tuned EvoDiff after iteration %s in %.2f seconds; last metrics=%s",
                    iteration,
                    finetune_seconds,
                    metrics[-1] if metrics else None,
                )
                exp_tracker.exp_results[iteration]["EvoDiff fine-tune"] = {
                    "seconds": finetune_seconds,
                    "n_sequences": len(train_sequences),
                    "epochs": args.finetune_epochs,
                    "last_epoch": metrics[-1] if metrics else None,
                }
                exp_tracker.save_results(save_path)
                _write_plots(save_path, finetune_metrics_path, plots_dir)
    finally:
        if trace_writer is not None:
            trace_writer.close()
            summary_path = os.path.join(os.path.dirname(trace_writer.path), f"seed_{args.seed}.trace_summary.json")
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(trace_writer.summary(), handle, indent=2, sort_keys=True)

    if args.save_finetuned_model:
        model_path = os.path.join(finetune_dir, "final_evodiff_state_dict.pt")
        torch.save(model.state_dict(), model_path)
        logger.info("Saved final fine-tuned EvoDiff state dict: %s", model_path)
    logger.info("Zero-shot seed %s complete: %s", args.seed, save_path)


def main():
    args = get_parser().parse_args()
    run_iter(args, logger)


if __name__ == "__main__":
    main()
