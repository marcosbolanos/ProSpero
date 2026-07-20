#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prospero.dataset import RegressionDataset
from prospero.experiment_tracker import ExperimentTracker
from prospero.experiments_config import ALPHABETS, WT_SEQUENCES
from prospero.landscapes import get_landscape
from prospero.runners.run_zero_shot_prosst import (
    AA20,
    CampaignState,
    JsonlGzWriter,
    ProSSTGenerator,
    known_wt_fitness,
)
from prospero.runners.run_zero_shot_protein import finetune_evodiff_on_sequences
from prospero.utils import set_seed


LOGGER = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Final 0shotProt optimization protocol with EvoDiff OA_DM_38M."
    )
    parser.add_argument("--results_dirpath", required=True)
    parser.add_argument("--task", default="AAV", choices=sorted(WT_SEQUENCES))
    parser.add_argument("--seed", type=int, choices=[1, 2, 3, 4, 5], default=1)
    parser.add_argument("--n_queries", type=int, default=128)
    parser.add_argument("--n_iters", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mask_budget", type=int, default=4)
    parser.add_argument("--entropy_quantile", type=float, default=0.5)
    parser.add_argument("--entropy_sigma", type=float, default=None)
    parser.add_argument("--entropy_chunk_size", type=int, default=64)
    parser.add_argument("--seed_grow_alpha", type=float, default=1.0)
    parser.add_argument("--seed_grow_beta", type=float, default=1.0)
    parser.add_argument("--seed_grow_coupling_tau", type=float, default=4.0)
    parser.add_argument("--alphabet", default="CHARGE", choices=list(ALPHABETS))
    parser.add_argument(
        "--decoding_vocab",
        choices=["restricted", "unrestricted"],
        default="restricted",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--debug_generation_trace",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--finetune_evodiff", action="store_true", default=False)
    parser.add_argument("--finetune_epochs", type=int, default=5)
    parser.add_argument("--finetune_lr", type=float, default=3e-5)
    parser.add_argument("--lambda_kl", type=float, default=2.0)
    parser.add_argument("--finetune_batch_size", type=int, default=1)
    parser.add_argument("--advantage_clip", type=float, default=2.0)
    parser.add_argument("--negative_weight", type=float, default=0.25)
    parser.add_argument("--full_deterministic", action="store_true", default=False)
    return parser


class EvoDiffGenerator(ProSSTGenerator):
    """EvoDiff adapter for the same generation and ranking protocol as ProSST."""

    def __init__(self, args):
        from evodiff.pretrained import OA_DM_38M  # type: ignore[reportMissingImports]

        self.args = args
        self.task = args.task
        self.device = torch.device(args.device)
        self.model, _, self.tokenizer, _ = OA_DM_38M()
        self.model = self.model.eval().to(self.device)
        self.base_model = None
        if args.finetune_evodiff:
            self.base_model, _, _, _ = OA_DM_38M()
            self.base_model = self.base_model.eval().to(self.device)
            for parameter in self.base_model.parameters():
                parameter.requires_grad_(False)

        self.aa_to_id = {
            aa: int(np.asarray(self.tokenizer.tokenize([aa])).reshape(-1)[0])
            for aa in AA20
        }
        self.id_to_aa = {token_id: aa for aa, token_id in self.aa_to_id.items()}
        self.full_ids = torch.tensor(
            [self.aa_to_id[aa] for aa in AA20],
            dtype=torch.long,
            device=self.device,
        )
        self.aa_index = {aa: index for index, aa in enumerate(AA20)}
        self.alphabet = ALPHABETS[args.alphabet]
        self.trace_writer = None
        self.trace_context = {}
        self.mask_position_reward_sum = {}
        self.mask_position_reward_count = {}
        self.mask_position_sample_count = {}
        self._mms_cache = None
        self.covered_length = len(WT_SEQUENCES[self.task])

    def _tokenize(self, sequences: list[str]) -> torch.Tensor:
        rows = [np.asarray(self.tokenizer.tokenize([sequence]), dtype=np.int64) for sequence in sequences]
        return torch.tensor(np.stack(rows), dtype=torch.long, device=self.device)

    def _model_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        timestep = torch.zeros(input_ids.shape[0], dtype=torch.long, device=self.device)
        return self.model(input_ids, timestep)

    @torch.inference_mode()
    def logits_for_sequences(self, seqs):
        return self._model_logits(self._tokenize(list(seqs)))

    @torch.inference_mode()
    def compute_position_entropies(self, sequence, chunk_size=None):
        if chunk_size is None:
            chunk_size = int(self.args.entropy_chunk_size)
        tokens = self._tokenize([sequence])[0]
        rows = tokens.repeat(len(sequence), 1)
        positions = torch.arange(len(sequence), device=self.device)
        rows[positions, positions] = int(self.tokenizer.mask_id)
        entropies = []
        for start in range(0, len(sequence), chunk_size):
            stop = min(start + chunk_size, len(sequence))
            logits = self._model_logits(rows[start:stop])
            local = torch.arange(stop - start, device=self.device)
            selected = logits[local, positions[start:stop]][:, self.full_ids]
            log_probs = F.log_softmax(selected, dim=-1)
            entropies.append((-(log_probs.exp() * log_probs).sum(dim=-1)).cpu())
        return torch.cat(entropies).numpy()

    @torch.inference_mode()
    def generate_batch(self, starting_sequence, batch_size):
        masks, _ = self.sample_masks(starting_sequence, batch_size)
        sequences = [list(starting_sequence) for _ in range(batch_size)]
        sequential_scores = np.zeros(batch_size, dtype=np.float64)
        log_likelihoods = np.zeros(batch_size, dtype=np.float64)
        max_steps = max(len(mask) for mask in masks)

        for step in range(max_steps):
            active = [index for index, mask in enumerate(masks) if step < len(mask)]
            positions = [int(masks[index][step]) for index in active]
            rows = self._tokenize(["".join(sequences[index]) for index in active])
            for local, position in enumerate(positions):
                rows[local, position] = int(self.tokenizer.mask_id)
            logits = self._model_logits(rows)
            for local, (particle, position) in enumerate(zip(active, positions)):
                original = starting_sequence[position]
                sampled, log_delta, sampled_ll, logp_sampled, logp_original = self._sample_from_logits(
                    logits[local, position], original
                )
                sequences[particle][position] = sampled
                sequential_scores[particle] += float(log_delta.item())
                log_likelihoods[particle] += float(sampled_ll.item())
                self.trace_event(
                    "decode_step",
                    particle=particle,
                    step=step + 1,
                    position=position,
                    distribution=self.args.decoding_vocab,
                    sampled=sampled,
                    original=original,
                    logp_sampled=float(logp_sampled.item()),
                    logp_original=float(logp_original.item()),
                    log_delta=float(log_delta.item()),
                    full_vocab_logp_sampled=float(sampled_ll.item()),
                )

        candidates = ["".join(sequence) for sequence in sequences]
        mms_scores = self.marginal_mutation_scores(starting_sequence, candidates)
        for index, (sequence, mms, sequential, ll) in enumerate(
            zip(candidates, mms_scores, sequential_scores, log_likelihoods)
        ):
            self.trace_event(
                "candidate",
                stage="completed_sequence",
                decode_steps=max_steps,
                candidate=index,
                sequence=sequence,
                zero_shot_score=float(mms),
                mms_score=float(mms),
                sequential_decode_score=float(sequential),
                log_likelihood=float(ll),
                inv_perplexity=float(np.exp(ll / max(1, self.args.mask_budget))),
            )
        return candidates, mms_scores

    def finetune_evodiff(self, sequences, scores, output_dir, round_idx):
        if self.base_model is None:
            raise ValueError("Fine-tuning requires --finetune_evodiff.")
        metrics = finetune_evodiff_on_sequences(
            self.model,
            self.base_model,
            self.tokenizer,
            sequences,
            scores,
            output_dir,
            round_idx,
            self.args,
        )
        self._mms_cache = None
        return metrics


def run_seed(args) -> None:
    set_seed(args.seed, args.full_deterministic)
    save_dir = Path(args.results_dirpath) / args.task
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"seed_{args.seed}.pkl"
    completion_path = save_dir / f"seed_{args.seed}.complete.json"
    metadata_path = save_dir / f"seed_{args.seed}.evodiff_zero_shot_metadata.json"
    trace_writer = (
        JsonlGzWriter(save_dir / "debug_traces" / f"seed_{args.seed}.events.jsonl.gz")
        if args.debug_generation_trace
        else None
    )

    oracle = get_landscape(args.task)
    reference_dataset = RegressionDataset(args.task)
    wild_type = WT_SEQUENCES[args.task]
    campaign = CampaignState(wild_type, known_wt_fitness(reference_dataset, wild_type))
    tracker = ExperimentTracker(
        LOGGER,
        deepcopy(reference_dataset),
        wild_type,
        best_percentile=0.95,
    )
    generator = EvoDiffGenerator(args)
    if trace_writer is not None:
        generator.set_trace_writer(trace_writer)

    metadata = {
        "task": args.task,
        "model": "EvoDiff OA_DM_38M",
        "generation": {
            "mode": "sequential_masked_decoding_without_resampling",
            "mask_strategy": "mixed_position_entropy_anticollapse",
            "mask_budget": args.mask_budget,
            "decoding_vocab": args.decoding_vocab,
            "score": "incumbent-relative marginal mutation score from fixed unmasked incumbent logits",
            "alphabet": args.alphabet,
            "batch_size": args.batch_size,
        },
        "evodiff_finetuning": {
            "enabled": bool(args.finetune_evodiff),
            "objective": (
                "advantage_weighted_masked_finetuning_plus_kl_to_frozen_base"
                if args.finetune_evodiff
                else None
            ),
            "training_corruption": "uniform random fixed-budget masks",
            "mask_budget": args.mask_budget,
            "epochs_per_round": args.finetune_epochs,
            "lr": args.finetune_lr,
            "lambda_kl": args.lambda_kl,
            "batch_size": args.finetune_batch_size,
            "replay": "all_online_queries",
            "negative_weight": args.negative_weight,
            "advantage_clip": args.advantage_clip,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    replay_sequences: list[str] = []
    replay_scores: list[float] = []
    finetune_dir = save_dir / f"seed_{args.seed}.evodiff_finetune"
    try:
        for iteration in range(1, args.n_iters + 1):
            candidate_pool: dict[str, tuple[str, float, str]] = {}
            excluded = campaign.excluded_sequences
            planned_batches = math.ceil(max(args.n_queries, args.batch_size) / args.batch_size)
            generation_round = 0
            while generation_round < planned_batches or len(candidate_pool) < args.n_queries:
                generation_round += 1
                incumbent = campaign.incumbent_sequence
                generator.set_trace_context(
                    task=args.task,
                    seed=args.seed,
                    n_queries=args.n_queries,
                    optimization_round=iteration,
                    generation_round=generation_round,
                    method="evodiff_zero_shot",
                    mask_strategy="mixed_position_entropy_anticollapse",
                    mask_budget=args.mask_budget,
                )
                batch_sequences, batch_scores = generator.generate_batch(incumbent, args.batch_size)
                for sequence, score in zip(batch_sequences, batch_scores):
                    if sequence in excluded:
                        continue
                    record = (sequence, float(score), incumbent)
                    previous = candidate_pool.get(sequence)
                    if previous is None or record[1] > previous[1]:
                        candidate_pool[sequence] = record

            selected = sorted(candidate_pool.values(), key=lambda record: record[1], reverse=True)[: args.n_queries]
            sequences = [record[0] for record in selected]
            mms_scores = [record[1] for record in selected]
            generator.trace_event(
                "candidate_pool_summary",
                planned_batches=planned_batches,
                actual_batches=generation_round,
                generated_candidates=generation_round * args.batch_size,
                unique_eligible_candidates=len(candidate_pool),
            )
            for rank, record in enumerate(selected, start=1):
                generator.trace_event(
                    "candidate_selected_for_query",
                    selected_rank=rank,
                    sequence=record[0],
                    zero_shot_score=record[1],
                )

            scores = oracle.get_fitness(np.asarray(sequences)).tolist()
            for rank, (sequence, score) in enumerate(zip(sequences, scores), start=1):
                generator.trace_event(
                    "oracle_query",
                    optimization_round=iteration,
                    query_rank=rank,
                    sequence=sequence,
                    oracle_score=float(score),
                )
            generator.update_mask_position_rewards(campaign.incumbent_sequence, sequences, scores)
            replay_sequences.extend(sequences)
            replay_scores.extend(float(score) for score in scores)
            tracker.calculate_top_n_metrics((sequences, scores), iteration, n=100)
            tracker.exp_results[iteration]["Zero-shot"] = deepcopy(metadata)
            tracker.exp_results[iteration]["Zero-shot"]["start_sequences"] = [campaign.incumbent_sequence]
            tracker.exp_results[iteration]["Zero-shot"]["selection"] = {
                "generated_candidates": generation_round * args.batch_size,
                "unique_eligible_candidates": len(candidate_pool),
                "selected_mms": mms_scores,
            }
            tracker.save_results(save_path)
            campaign.observe(sequences, scores)

            if args.finetune_evodiff and iteration < args.n_iters:
                start = time.perf_counter()
                metrics = generator.finetune_evodiff(
                    replay_sequences,
                    replay_scores,
                    finetune_dir,
                    iteration,
                )
                tracker.exp_results[iteration]["EvoDiff fine-tune"] = {
                    "seconds": float(time.perf_counter() - start),
                    "n_sequences": len(replay_sequences),
                    "epochs": args.finetune_epochs,
                    "last_epoch": metrics[-1] if metrics else None,
                }
                tracker.save_results(save_path)
    finally:
        if trace_writer is not None:
            trace_writer.close()
            summary = trace_writer.path.with_suffix("").with_suffix(".trace_summary.json")
            summary.write_text(json.dumps({"event_counts": trace_writer.counts}, indent=2, sort_keys=True))

    completion_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "task": args.task,
                "seed": args.seed,
                "rounds": args.n_iters,
                "queries_per_round": args.n_queries,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_seed(get_parser().parse_args())


if __name__ == "__main__":
    main()
