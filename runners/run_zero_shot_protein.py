import argparse
import json
import os
import sys
import time
from argparse import ArgumentDefaultsHelpFormatter
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from prospero.dataset import RegressionDataset
from prospero.debug_trace import JsonlGzTraceWriter
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
        description="Zero-shot ProSpero ablation without a trainable surrogate.",
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
    parser.add_argument("--finetune_evodiff", action="store_true", default=False)
    parser.add_argument("--finetune_epochs", type=int, default=5)
    parser.add_argument("--finetune_lr", type=float, default=1e-5)
    parser.add_argument("--lambda_kl", type=float, default=2.0)
    parser.add_argument("--finetune_batch_size", type=int, default=1)
    parser.add_argument("--finetune_replay", choices=["latest", "all"], default="latest")
    parser.add_argument("--finetune_after_final", action="store_true", default=False)
    parser.add_argument("--reward_mode", choices=["grpo_advantage"], default="grpo_advantage")
    parser.add_argument("--negative_weight", type=float, default=0.25)
    parser.add_argument("--advantage_clip", type=float, default=2.0)

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


def grpo_advantage_weights(scores, clip=2.0):
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("Cannot compute GRPO rewards for an empty score list.")
    center = float(np.mean(scores))
    scale = float(np.std(scores))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    weights = (scores - center) / scale
    if clip is not None:
        weights = np.clip(weights, -float(clip), float(clip))
    return weights.astype(np.float32), {
        "reward_mode": "grpo_advantage",
        "baseline": center,
        "baseline_mode": "group_mean",
        "scale": scale,
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "weight_mean": float(np.mean(weights)),
        "weight_std": float(np.std(weights)),
        "frac_negative": float(np.mean(weights < 0)),
    }


def global_grad_norm(parameters):
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total += float(torch.sum(grad * grad).item())
    return total**0.5


@torch.no_grad()
def parameter_delta_norm(model, base_model):
    delta_sq = 0.0
    base_sq = 0.0
    for param, base_param in zip(model.parameters(), base_model.parameters()):
        diff = param.detach() - base_param.detach()
        delta_sq += float(torch.sum(diff * diff).item())
        base_sq += float(torch.sum(base_param.detach() * base_param.detach()).item())
    delta = delta_sq**0.5
    base = base_sq**0.5
    return delta, delta / max(base, 1e-12)


def make_evodiff_finetune_batch(tokenizer, sequences, weights, mask_budget, device):
    input_rows = np.stack([tokenizer.tokenize([seq]) for seq in sequences])
    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
    target_ids = input_ids.clone()
    masked_input_ids = input_ids.clone()
    row_ids = []
    pos_ids = []
    target_tokens = []
    mask_weights = []
    for row_idx, seq in enumerate(sequences):
        budget = min(max(1, int(mask_budget)), len(seq))
        positions = np.random.choice(np.arange(len(seq)), budget, replace=False)
        for pos in positions:
            masked_input_ids[row_idx, int(pos)] = tokenizer.mask_id
            row_ids.append(row_idx)
            pos_ids.append(int(pos))
            target_tokens.append(int(target_ids[row_idx, int(pos)].item()))
            mask_weights.append(float(weights[row_idx]))
    return {
        "input_ids": masked_input_ids,
        "row_ids": torch.tensor(row_ids, dtype=torch.long, device=device),
        "pos_ids": torch.tensor(pos_ids, dtype=torch.long, device=device),
        "target_ids": torch.tensor(target_tokens, dtype=torch.long, device=device),
        "weights": torch.tensor(mask_weights, dtype=torch.float32, device=device),
    }


def finetune_evodiff_on_sequences(
    model,
    base_model,
    tokenizer,
    sequences,
    scores,
    output_dir,
    round_idx,
    args,
):
    if len(sequences) != len(scores):
        raise ValueError("sequences and scores must have the same length.")
    if not sequences:
        return []
    if args.reward_mode != "grpo_advantage":
        raise ValueError(f"Unsupported reward_mode={args.reward_mode!r}")

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "evodiff_finetune_metrics.jsonl")
    weights, reward_metadata = grpo_advantage_weights(scores, clip=args.advantage_clip)
    order = np.arange(len(sequences))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr)
    metrics = []
    train_start = time.perf_counter()
    device = next(model.parameters()).device

    for epoch in range(1, args.finetune_epochs + 1):
        model.train()
        np.random.shuffle(order)
        epoch_start = time.perf_counter()
        step_metrics = []
        for start in range(0, len(order), args.finetune_batch_size):
            ids = order[start : start + args.finetune_batch_size]
            batch = make_evodiff_finetune_batch(
                tokenizer,
                [sequences[i] for i in ids],
                weights[ids],
                args.mask_budget,
                device,
            )
            timestep = torch.zeros(batch["input_ids"].shape[0], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], timestep)
            with torch.no_grad():
                base_logits = base_model(batch["input_ids"], timestep)

            masked_logits = logits[batch["row_ids"], batch["pos_ids"]]
            masked_base_logits = base_logits[batch["row_ids"], batch["pos_ids"]]
            nll = F.cross_entropy(masked_logits, batch["target_ids"], reduction="none")

            log_probs = F.log_softmax(masked_logits[:, :20], dim=-1)
            base_log_probs = F.log_softmax(masked_base_logits[:, :20], dim=-1)
            probs = log_probs.exp()
            kl = (probs * (log_probs - base_log_probs)).sum(dim=-1)

            pos_weight = batch["weights"].clamp_min(0.0)
            neg_weight = (-batch["weights"]).clamp_min(0.0)
            signed_weight = pos_weight - float(args.negative_weight) * neg_weight
            nll_loss = (nll * signed_weight).mean()
            kl_loss = kl.mean()
            loss = nll_loss + float(args.lambda_kl) * kl_loss
            loss.backward()
            grad_norm = global_grad_norm(model.parameters())
            optimizer.step()
            step_metrics.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "weighted_nll": float(nll_loss.detach().cpu()),
                    "kl": float(kl_loss.detach().cpu()),
                    "grad_norm": float(grad_norm),
                    "mean_weight": float(batch["weights"].mean().detach().cpu()),
                    "effective_positive_weight": float(pos_weight.mean().detach().cpu()),
                    "effective_negative_weight": float(neg_weight.mean().detach().cpu()),
                }
            )

        update_norm, relative_update_norm = parameter_delta_norm(model, base_model)
        metric = {
            "event": "evodiff_finetune_epoch",
            "round": int(round_idx),
            "epoch": int(epoch),
            "epochs": int(args.finetune_epochs),
            "n_sequences": int(len(sequences)),
            "batch_size": int(args.finetune_batch_size),
            "mask_budget": int(args.mask_budget),
            "lr": float(args.finetune_lr),
            "lambda_kl": float(args.lambda_kl),
            "reward_mode": args.reward_mode,
            "negative_weight": float(args.negative_weight),
            "reward_metadata": reward_metadata,
            "seconds": float(time.perf_counter() - epoch_start),
            "loss": float(np.mean([m["loss"] for m in step_metrics])),
            "weighted_nll": float(np.mean([m["weighted_nll"] for m in step_metrics])),
            "kl": float(np.mean([m["kl"] for m in step_metrics])),
            "grad_norm": float(np.mean([m["grad_norm"] for m in step_metrics])),
            "max_grad_norm": float(np.max([m["grad_norm"] for m in step_metrics])),
            "mean_weight": float(np.mean([m["mean_weight"] for m in step_metrics])),
            "effective_positive_weight": float(np.mean([m["effective_positive_weight"] for m in step_metrics])),
            "effective_negative_weight": float(np.mean([m["effective_negative_weight"] for m in step_metrics])),
            "update_norm": float(update_norm),
            "relative_update_norm": float(relative_update_norm),
        }
        metrics.append(metric)
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric, sort_keys=True) + "\n")

    with open(metrics_path, "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event": "evodiff_finetune_round_complete",
                    "round": int(round_idx),
                    "epochs": int(args.finetune_epochs),
                    "n_sequences": int(len(sequences)),
                    "total_seconds": float(time.perf_counter() - train_start),
                },
                sort_keys=True,
            )
            + "\n"
        )
    model.eval()
    return metrics


def run_iter(args, logger):
    set_seed(args.seed, args.full_deterministic)
    logger.info("Starting zero-shot seed %s", args.seed)

    save_dir = os.path.join(args.results_dirpath, args.task)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"seed_{args.seed}.pkl")
    metadata_path = os.path.join(save_dir, f"seed_{args.seed}.zero_shot_metadata.json")

    wt_sequence = WT_SEQUENCES[args.task]
    oracle = get_landscape(args.task)
    dataset = RegressionDataset(args.task)
    alphabet = ALPHABETS[args.alphabet]

    from evodiff.pretrained import OA_DM_38M  # type: ignore[reportMissingImports]

    model, _, tokenizer_oadm, _ = OA_DM_38M()
    model = model.cuda()
    model.eval()
    base_model = None
    if args.finetune_evodiff:
        base_model, _, _, _ = OA_DM_38M()
        base_model = base_model.cuda()
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
            raise ValueError(
                "--mask_budget is required for random, middle_entropy, and seed_grow."
            )
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
        "constrained_unmasking_distribution": "alphabet cluster softmax",
        "rollout_distribution": "full 20-AA softmax",
        "resampling": "zero_shot_score_times_inverse_perplexity",
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
        if args.mask_strategy == "seed_grow"
        else None,
        "evodiff_finetuning": {
            "enabled": bool(args.finetune_evodiff),
            "objective": "grpo_advantage_weighted_masked_token_nll_plus_kl_current_to_frozen_base",
            "training_corruption": "uniform random fixed-budget masks",
            "mask_budget": args.mask_budget,
            "epochs_per_round": args.finetune_epochs,
            "lr": args.finetune_lr,
            "lambda_kl": args.lambda_kl,
            "batch_size": args.finetune_batch_size,
            "replay": args.finetune_replay,
            "reward_mode": args.reward_mode,
            "negative_weight": args.negative_weight,
            "advantage_clip": args.advantage_clip,
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
    finetune_dir = os.path.join(save_dir, f"seed_{args.seed}.evodiff_finetune")
    try:
        for e in range(args.n_iters):
            iteration = e + 1
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
                    method="no_ft",
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
            for query_idx, (sequence, score) in enumerate(zip(sequences, scores), start=1):
                sampler.trace_event(
                    "oracle_query",
                    optimization_round=iteration,
                    query_rank=int(query_idx),
                    sequence=sequence,
                    oracle_score=float(score),
                )
            if args.mask_strategy == "mixed_explore_exploit":
                sampler.update_mask_position_rewards(starting_sequence, sequences, scores)
            dataset.add((sequences, scores))
            replay_sequences.extend(sequences)
            replay_scores.extend(scores)
            exp_tracker.calculate_top_n_metrics((sequences, scores), iteration, n=100)
            exp_tracker.exp_results[iteration]["Zero-shot"] = metadata
            exp_tracker.save_results(save_path)

            starting_sequence = (
                get_new_starting_seq(dataset)
                if not args.task.startswith("D_SHIFT")
                else get_new_starting_seq_dshift(dataset, args.task)
            )
            if args.finetune_evodiff and (iteration < args.n_iters or args.finetune_after_final):
                if args.mask_budget is None:
                    raise ValueError("--mask_budget is required for EvoDiff fine-tuning.")
                train_sequences = sequences if args.finetune_replay == "latest" else replay_sequences
                train_scores = scores if args.finetune_replay == "latest" else replay_scores
                ft_start = time.perf_counter()
                ft_metrics = finetune_evodiff_on_sequences(
                    model,
                    base_model,
                    tokenizer_oadm,
                    train_sequences,
                    train_scores,
                    finetune_dir,
                    round_idx=iteration,
                    args=args,
                )
                exp_tracker.exp_results[iteration]["EvoDiff fine-tune"] = {
                    "seconds": float(time.perf_counter() - ft_start),
                    "n_sequences": len(train_sequences),
                    "epochs": args.finetune_epochs,
                    "last_epoch": ft_metrics[-1] if ft_metrics else None,
                }
                exp_tracker.save_results(save_path)
    finally:
        if trace_writer is not None:
            trace_writer.close()
            summary_path = os.path.join(os.path.dirname(trace_writer.path), f"seed_{args.seed}.trace_summary.json")
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(trace_writer.summary(), handle, indent=2, sort_keys=True)

    logger.info("Zero-shot seed %s complete: %s", args.seed, save_path)


def main():
    args = get_parser().parse_args()
    run_iter(args, logger)


if __name__ == "__main__":
    main()
