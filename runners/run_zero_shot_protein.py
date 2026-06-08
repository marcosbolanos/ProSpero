import argparse
import json
import os
import sys
from argparse import ArgumentDefaultsHelpFormatter
from copy import deepcopy

import numpy as np

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
        choices=["calibrated_random", "random", "middle_entropy", "seed_grow"],
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
    if args.mask_strategy in {"middle_entropy", "seed_grow"}:
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
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    exp_tracker = ExperimentTracker(
        logger,
        deepcopy(dataset),
        wt_sequence,
        best_percentile=0.95,
    )

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
            dataset.add((sequences, scores))
            exp_tracker.calculate_top_n_metrics((sequences, scores), iteration, n=100)
            exp_tracker.exp_results[iteration]["Zero-shot"] = metadata
            exp_tracker.save_results(save_path)

            starting_sequence = (
                get_new_starting_seq(dataset)
                if not args.task.startswith("D_SHIFT")
                else get_new_starting_seq_dshift(dataset, args.task)
            )
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
