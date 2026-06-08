#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
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
from prospero.utils import get_new_starting_seq, set_seed


AA20 = list("ACDEFGHIKLMNPQRSTVWY")
STRUCTURE_TOKENS_DIR = Path("outputs/prosst_structure_tokens")
class StructureMapping:
    def __init__(self, structure_token_name, offset, covered_length, identity, note=""):
        self.structure_token_name = structure_token_name
        self.offset = offset
        self.covered_length = covered_length
        self.identity = identity
        self.note = note


TASK_MAPPINGS = {
    "AAV": StructureMapping("CAPSD_AAV2S_Sinai_2021", 450, 90, 1.0),
    "GFP": StructureMapping("GFP_AEQVI_Sarkisyan_2016", 0, 238, 1.0),
    "AMIE": StructureMapping("AMIE_PSEAE_Wrenbeck_2017", 0, 341, 0.9971),
    "TEM": StructureMapping("BLAT_ECOLX_Firnberg_2014", 0, 286, 0.9965),
    "UBE2I": StructureMapping("UBC9_HUMAN_Weile_2017", 0, 159, 0.9937),
    "Pab1": StructureMapping("PABP_YEAST_Melamed_2013", 125, 75, 0.9733),
    "E4B": StructureMapping("UBE4B_MOUSE_Starita_2013", 1071, 102, 0.9510),
    "LGK": StructureMapping(
        "LGK_LIPST_Klesmith_2015",
        0,
        439,
        0.9732,
        "ProSpero LGK has 8 C-terminal residues without precomputed structure tokens; those positions are fixed.",
    ),
}
SUPPORTED_TASKS = sorted(TASK_MAPPINGS)


def get_parser():
    p = argparse.ArgumentParser(description="ProSpero-style optimization with ProSST logits.")
    p.add_argument("--results_dirpath", required=True)
    p.add_argument("--task", default="AAV", choices=SUPPORTED_TASKS)
    p.add_argument("--seed", type=int, choices=[1, 2, 3, 4, 5], default=1)
    p.add_argument("--n_queries", type=int, default=128)
    p.add_argument("--n_iters", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--mask_budget", type=int, default=4)
    p.add_argument("--mask_strategy", choices=["random", "middle_entropy", "seed_grow", "mixed_explore_exploit"], default="mixed_explore_exploit")
    p.add_argument("--entropy_quantile", type=float, default=0.5)
    p.add_argument("--entropy_sigma", type=float, default=None)
    p.add_argument("--entropy_chunk_size", type=int, default=16)
    p.add_argument("--seed_grow_alpha", type=float, default=1.0)
    p.add_argument("--seed_grow_beta", type=float, default=1.0)
    p.add_argument("--seed_grow_coupling_tau", type=float, default=4.0)
    p.add_argument("--alphabet", default="CHARGE", choices=list(ALPHABETS))
    p.add_argument("--smc_vocab", choices=["cluster", "full"], default="cluster")
    p.add_argument(
        "--non_cluster_logit_penalty",
        type=float,
        default=0.0,
        help=(
            "When --smc_vocab full, subtract this from logits for amino acids "
            "outside the original residue's alphabet cluster before sampling."
        ),
    )
    p.add_argument("--model_path", default="AI4Protein/ProSST-2048")
    p.add_argument("--structure_tokens_dir", default=str(STRUCTURE_TOKENS_DIR))
    p.add_argument("--structure_vocab_size", default="2048")
    p.add_argument("--device", default="cuda")
    p.add_argument("--debug_generation_trace", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--finetune_prosst", action="store_true", default=False)
    p.add_argument("--finetune_epochs", type=int, default=5)
    p.add_argument("--finetune_lr", type=float, default=1e-5)
    p.add_argument("--lambda_kl", type=float, default=2.0)
    p.add_argument("--finetune_batch_size", type=int, default=16)
    p.add_argument("--finetune_replay", choices=["all", "latest"], default="all")
    p.add_argument("--finetune_after_final", action="store_true", default=False)
    p.add_argument(
        "--reward_mode",
        choices=["rank", "grpo_advantage", "standardized_advantage", "bottom_quantile_negative"],
        default="rank",
    )
    p.add_argument("--advantage_clip", type=float, default=2.0)
    p.add_argument("--negative_weight", type=float, default=0.25)
    p.add_argument("--bottom_quantile", type=float, default=0.25)
    p.add_argument("--full_deterministic", action="store_true", default=False)
    return p


class JsonlGzWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = gzip.open(self.path, "at", encoding="utf-8")
        self.counts = {}

    def write(self, record):
        self.counts[record.get("event", "unknown")] = self.counts.get(record.get("event", "unknown"), 0) + 1
        self.handle.write(json.dumps(record, sort_keys=True) + "\n")

    def close(self):
        self.handle.close()


def read_fasta_sequence(path):
    seq = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith(">"):
                seq.append(line)
    return "".join(seq)


def read_structure_tokens(path):
    return [int(tok) for tok in read_fasta_sequence(path).split(",") if tok]


def tokenize_structure_tokens(tokens, device):
    shifted = [token + 3 for token in tokens]
    return torch.tensor([[1, *shifted, 2]], dtype=torch.long, device=device)


def normalized_rank_weights(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("Cannot rank an empty score list.")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=float)
    ranks[order] = np.arange(scores.size, dtype=float)
    if scores.size == 1:
        return np.ones(1, dtype=np.float32)
    return (ranks / float(scores.size - 1)).astype(np.float32)


def reward_weights(scores, mode, baseline=None, clip=2.0, bottom_quantile=0.25):
    scores = np.asarray(scores, dtype=float)
    if mode == "rank":
        weights = normalized_rank_weights(scores)
        metadata = {
            "reward_mode": "rank",
            "weight_min": float(np.min(weights)),
            "weight_max": float(np.max(weights)),
            "weight_mean": float(np.mean(weights)),
        }
        return weights, metadata
    if scores.size == 0:
        raise ValueError("Cannot compute rewards for an empty score list.")
    if mode == "grpo_advantage":
        center = float(np.mean(scores))
        scale = float(np.std(scores))
        baseline_mode = "group_mean"
    elif mode == "standardized_advantage":
        center = float(baseline)
        scale = float(np.std(scores))
        baseline_mode = "moving_starting_sequence"
    elif mode == "bottom_quantile_negative":
        weights = normalized_rank_weights(scores).astype(np.float32)
        threshold = float(np.quantile(scores, bottom_quantile))
        weights[scores <= threshold] = -1.0
        return weights, {
            "reward_mode": mode,
            "baseline": threshold,
            "baseline_mode": f"bottom_quantile_{bottom_quantile:g}",
            "bottom_quantile": float(bottom_quantile),
            "weight_min": float(np.min(weights)),
            "weight_max": float(np.max(weights)),
            "weight_mean": float(np.mean(weights)),
            "frac_negative": float(np.mean(weights < 0)),
        }
    else:
        raise ValueError(f"Unsupported reward mode: {mode}")

    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    weights = (scores - center) / scale
    if clip is not None:
        weights = np.clip(weights, -float(clip), float(clip))
    return weights.astype(np.float32), {
        "reward_mode": mode,
        "baseline": center,
        "baseline_mode": baseline_mode,
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


class ProSSTGenerator:
    def __init__(self, args):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.args = args
        self.task = args.task
        self.device = torch.device(args.device)
        self.model = AutoModelForMaskedLM.from_pretrained(args.model_path, trust_remote_code=True).eval().to(self.device)
        self.base_model = None
        if args.finetune_prosst:
            self.base_model = AutoModelForMaskedLM.from_pretrained(args.model_path, trust_remote_code=True).eval().to(self.device)
            for param in self.base_model.parameters():
                param.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        self.vocab = self.tokenizer.get_vocab()
        self.aa_to_id = {aa: self.vocab[aa] for aa in AA20}
        self.id_to_aa = {idx: aa for aa, idx in self.aa_to_id.items()}
        self.full_ids = torch.tensor([self.aa_to_id[aa] for aa in AA20], dtype=torch.long, device=self.device)
        self.alphabet = ALPHABETS[args.alphabet]
        self.trace_writer = None
        self.trace_context = {}
        self.mask_position_reward_sum = {}
        self.mask_position_reward_count = {}
        self.mask_position_sample_count = {}

        mapping = TASK_MAPPINGS[self.task]
        self.mapping = mapping
        wt = WT_SEQUENCES[self.task]
        self.full_wt_length = len(wt)
        self.covered_length = min(mapping.covered_length, len(wt))
        structure_path = (
            Path(args.structure_tokens_dir)
            / "structure_sequence"
            / args.structure_vocab_size
            / f"{mapping.structure_token_name}.fasta"
        )
        structure_tokens = read_structure_tokens(structure_path)
        self.structure_tokens = structure_tokens[mapping.offset : mapping.offset + self.covered_length]
        if len(self.structure_tokens) != self.covered_length:
            raise ValueError(f"{self.task} structure token crop length mismatch.")
        self.ss_input_ids_1 = tokenize_structure_tokens(self.structure_tokens, self.device)

    def set_trace_writer(self, writer):
        self.trace_writer = writer

    def set_trace_context(self, **kwargs):
        self.trace_context = {k: v for k, v in kwargs.items() if v is not None}

    def trace_event(self, event, **payload):
        if self.trace_writer is not None:
            self.trace_writer.write({"event": event, **self.trace_context, **payload})

    def _tokenize_batch(self, seqs):
        out = self.tokenizer(seqs, return_tensors="pt", padding=False)
        return out["input_ids"].to(self.device), out["attention_mask"].to(self.device)

    @torch.inference_mode()
    def logits_for_sequences(self, seqs):
        covered = [seq[: self.covered_length] for seq in seqs]
        input_ids, attention_mask = self._tokenize_batch(covered)
        return self.logits_for_input_ids(input_ids, attention_mask)

    @torch.inference_mode()
    def logits_for_input_ids(self, input_ids, attention_mask):
        ss = self.ss_input_ids_1.repeat(input_ids.shape[0], 1)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, ss_input_ids=ss)
        return out.logits[:, 1:-1, :]

    @torch.inference_mode()
    def compute_position_entropies(self, sequence, chunk_size=None):
        if chunk_size is None:
            chunk_size = int(self.args.entropy_chunk_size)
        sequence = sequence[: self.covered_length]
        input_ids, attention_mask = self._tokenize_batch([sequence])
        rows = input_ids.repeat(len(sequence), 1)
        masks = attention_mask.repeat(len(sequence), 1)
        for pos in range(len(sequence)):
            rows[pos, pos + 1] = self.tokenizer.mask_token_id
        entropies = []
        aa_cols = self.full_ids
        for start in range(0, rows.shape[0], chunk_size):
            logits = self.logits_for_input_ids(rows[start : start + chunk_size], masks[start : start + chunk_size])
            positions = torch.arange(start, min(start + chunk_size, rows.shape[0]), device=self.device)
            local = torch.arange(len(positions), device=self.device)
            selected = logits[local, positions][:, aa_cols]
            log_probs = F.log_softmax(selected, dim=1)
            probs = log_probs.exp()
            entropies.append((-(probs * log_probs).sum(dim=1)).detach().cpu())
        return torch.cat(entropies).numpy()

    def middle_entropy_scores(self, entropies):
        h_star = float(np.quantile(entropies, self.args.entropy_quantile))
        sigma = self.args.entropy_sigma
        if sigma is None:
            sigma = float(np.std(entropies))
            if sigma <= 1e-8:
                sigma = 1.0
        scores = np.exp(-((entropies - h_star) ** 2) / (2 * sigma**2)) + 1e-12
        scores = scores / scores.sum()
        return scores, {
            "entropy_quantile": float(self.args.entropy_quantile),
            "entropy_h_star": h_star,
            "entropy_sigma": float(sigma),
            "entropy_min": float(np.min(entropies)),
            "entropy_max": float(np.max(entropies)),
            "entropy_mean": float(np.mean(entropies)),
        }

    def sample_seed_and_grow_mask(self, positions, base_scores, mask_budget):
        selected = [int(np.random.choice(positions, p=base_scores))]
        positions = np.asarray(positions)
        base_scores = np.asarray(base_scores)
        while len(selected) < mask_budget:
            remaining_mask = ~np.isin(positions, selected)
            remaining_ids = positions[remaining_mask]
            remaining_base = base_scores[remaining_mask]
            distances = np.abs(remaining_ids[:, None] - np.asarray(selected)[None, :])
            coupling = np.exp(-distances / max(self.args.seed_grow_coupling_tau, 1e-6)).max(axis=1)
            scores = self.args.seed_grow_alpha * remaining_base + self.args.seed_grow_beta * coupling
            scores = np.maximum(scores, 1e-12)
            scores = scores / scores.sum()
            selected.append(int(np.random.choice(remaining_ids, p=scores)))
        return np.array(selected, dtype=int)

    def sample_mixed_mask(self, positions, base_scores, mask_budget):
        positions = np.asarray(positions)
        selected = []

        def remaining():
            return ~np.isin(positions, selected)

        def choose(ids, scores):
            scores = np.maximum(np.asarray(scores, dtype=float), 1e-12)
            scores = scores / scores.sum()
            return int(np.random.choice(ids, p=scores))

        exploit_budget = max(1, mask_budget // 2)
        entropy_budget = 1 if mask_budget - exploit_budget > 0 else 0
        rewards = np.asarray(
            [
                self.mask_position_reward_sum.get(int(pos), 0.0)
                / max(1, self.mask_position_reward_count.get(int(pos), 0))
                for pos in positions
            ],
            dtype=float,
        )
        positive = np.maximum(rewards, 0.0)
        if positive.sum() <= 1e-12:
            selected.extend(self.sample_seed_and_grow_mask(positions, base_scores, min(exploit_budget, mask_budget)).tolist())
        else:
            for _ in range(min(exploit_budget, mask_budget)):
                rem = remaining()
                selected.append(choose(positions[rem], positive[rem] if positive[rem].sum() > 1e-12 else base_scores[rem]))
        for _ in range(entropy_budget):
            rem = remaining()
            if len(selected) < mask_budget:
                selected.append(choose(positions[rem], base_scores[rem]))
        while len(selected) < mask_budget:
            rem = remaining()
            ids = positions[rem]
            counts = np.asarray([self.mask_position_sample_count.get(int(pos), 0) for pos in ids], dtype=float)
            selected.append(choose(ids, 1.0 / np.sqrt(1.0 + counts)))
        return np.array(selected, dtype=int)

    def sample_masks(self, sequence, batch_size):
        covered_sequence = sequence[: self.covered_length]
        mask_budget = min(max(1, int(self.args.mask_budget)), len(covered_sequence))
        positions = np.arange(len(covered_sequence))
        entropy_metadata = None
        base_scores = np.ones(len(covered_sequence), dtype=float) / len(covered_sequence)
        if self.args.mask_strategy in {"middle_entropy", "seed_grow", "mixed_explore_exploit"}:
            entropies = self.compute_position_entropies(sequence)
            base_scores, entropy_metadata = self.middle_entropy_scores(entropies)
        masks = []
        for particle in range(batch_size):
            if self.args.mask_strategy == "random":
                sampled = np.random.choice(positions, mask_budget, replace=False)
            elif self.args.mask_strategy == "middle_entropy":
                sampled = np.random.choice(positions, mask_budget, replace=False, p=base_scores)
            elif self.args.mask_strategy == "seed_grow":
                sampled = self.sample_seed_and_grow_mask(positions, base_scores, mask_budget)
            elif self.args.mask_strategy == "mixed_explore_exploit":
                sampled = self.sample_mixed_mask(positions, base_scores, mask_budget)
            else:
                raise ValueError(self.args.mask_strategy)
            sampled = np.array(sorted(sampled), dtype=int)
            for pos in sampled:
                self.mask_position_sample_count[int(pos)] = self.mask_position_sample_count.get(int(pos), 0) + 1
            self.trace_event(
                "mask_selected",
                particle=particle,
                strategy=self.args.mask_strategy,
                starting_sequence=sequence,
                mask_positions=sampled.tolist(),
                mask_residues=[covered_sequence[pos] for pos in sampled],
                mask_size=len(sampled),
                entropy_metadata=entropy_metadata,
            )
            masks.append(sampled)
        return masks, entropy_metadata

    def _distribution_ids(self, original_aa):
        if self.args.smc_vocab == "full":
            return self.full_ids
        return torch.tensor([self.aa_to_id[aa] for aa in self.alphabet[original_aa]], dtype=torch.long, device=self.device)

    def _sample_from_logits(self, logits, original_aa):
        dist_ids = self._distribution_ids(original_aa)
        dist_logits = logits[dist_ids].clone()
        if self.args.smc_vocab == "full" and self.args.non_cluster_logit_penalty > 0:
            cluster = set(self.alphabet[original_aa])
            penalty = torch.tensor(
                [
                    0.0 if self.id_to_aa[int(token.item())] in cluster else float(self.args.non_cluster_logit_penalty)
                    for token in dist_ids
                ],
                dtype=dist_logits.dtype,
                device=dist_logits.device,
            )
            dist_logits = dist_logits - penalty
        dist_log_probs = F.log_softmax(dist_logits, dim=0)
        sampled_idx = torch.multinomial(dist_log_probs.exp(), num_samples=1)
        sampled_id = dist_ids[sampled_idx].flatten()[0]
        original_id = torch.tensor(self.aa_to_id[original_aa], dtype=torch.long, device=self.device)
        original_idx = torch.nonzero(dist_ids == original_id, as_tuple=False).flatten()
        if original_idx.numel() == 0:
            raise ValueError(f"Original AA {original_aa} absent from sampling distribution")
        logp_sampled = dist_log_probs[sampled_idx].flatten()[0]
        logp_original = dist_log_probs[original_idx[:1]].flatten()[0]
        full_log_probs = F.log_softmax(logits[self.full_ids], dim=0)
        aa = self.id_to_aa[int(sampled_id.item())]
        return aa, logp_sampled - logp_original, full_log_probs[AA20.index(aa)], logp_sampled, logp_original

    @torch.inference_mode()
    def generate_batch(self, starting_sequence, batch_size):
        masks, entropy_metadata = self.sample_masks(starting_sequence, batch_size)
        covered_start = starting_sequence[: self.covered_length]
        fixed_tail = starting_sequence[self.covered_length :]
        seqs = [list(covered_start) for _ in range(batch_size)]
        scores = np.zeros(batch_size, dtype=np.float64)
        lls = np.zeros(batch_size, dtype=np.float64)
        max_steps = max(len(mask) for mask in masks)
        for step in range(max_steps):
            masked_input_rows = []
            masked_attention_rows = []
            active = []
            active_pos = []
            for idx, mask in enumerate(masks):
                if step >= len(mask):
                    continue
                pos = int(mask[step])
                input_ids, attention_mask = self._tokenize_batch(["".join(seqs[idx])])
                input_ids = input_ids[0]
                attention_mask = attention_mask[0]
                input_ids[pos + 1] = self.tokenizer.mask_token_id
                masked_input_rows.append(input_ids)
                masked_attention_rows.append(attention_mask)
                active.append(idx)
                active_pos.append(pos)
            logits = self.logits_for_input_ids(
                torch.stack(masked_input_rows, dim=0),
                torch.stack(masked_attention_rows, dim=0),
            )
            for local_idx, (particle, pos) in enumerate(zip(active, active_pos)):
                original_aa = covered_start[pos]
                sampled_aa, log_delta, sampled_ll, logp_sampled, logp_original = self._sample_from_logits(
                    logits[local_idx, pos],
                    original_aa,
                )
                seqs[particle][pos] = sampled_aa
                scores[particle] += float(log_delta.item())
                lls[particle] += float(sampled_ll.item())
                self.trace_event(
                    "smc_step",
                    particle=int(particle),
                    step=int(step + 1),
                    position=int(pos),
                    distribution="full_20aa" if self.args.smc_vocab == "full" else "constrained_cluster",
                    non_cluster_logit_penalty=float(self.args.non_cluster_logit_penalty),
                    sampled=sampled_aa,
                    original=original_aa,
                    logp_sampled=float(logp_sampled.item()),
                    logp_original=float(logp_original.item()),
                    log_delta=float(log_delta.item()),
                    full_vocab_logp_sampled=float(sampled_ll.item()),
                )
        out = ["".join(seq) + fixed_tail for seq in seqs]
        for idx, (seq, score, ll) in enumerate(zip(out, scores, lls)):
            self.trace_event(
                "candidate",
                stage="terminal_no_rollout",
                smc_step=max_steps,
                candidate=int(idx),
                sequence=seq,
                zero_shot_score=float(score),
                log_likelihood=float(ll),
                inv_perplexity=float(np.exp(ll / max(1, self.args.mask_budget))),
            )
        return out, scores

    def get_top_sequences(self, candidates, candidate_scores, n_queries, ref_sequences):
        ref = set("".join(str(x) for x in seq) for seq in ref_sequences)
        best = {}
        for seq, score in zip(candidates, candidate_scores):
            if seq in ref:
                continue
            if seq not in best or float(score) > best[seq]:
                best[seq] = float(score)
        selected = sorted(best, key=best.get, reverse=True)[:n_queries]
        for rank, seq in enumerate(selected, start=1):
            self.trace_event("candidate_selected_for_query", selected_rank=rank, sequence=seq, zero_shot_score=best[seq])
        return selected

    def update_mask_position_rewards(self, starting_sequence, sequences, scores):
        baseline = float(np.mean(scores))
        for sequence, score in zip(sequences, scores):
            advantage = float(score) - baseline
            for pos, (before, after) in enumerate(zip(starting_sequence, sequence)):
                if before == after:
                    continue
                self.mask_position_reward_sum[pos] = self.mask_position_reward_sum.get(pos, 0.0) + advantage
                self.mask_position_reward_count[pos] = self.mask_position_reward_count.get(pos, 0) + 1

    def _make_finetune_batch(self, sequences, weights, mask_budget):
        covered_sequences = [seq[: self.covered_length] for seq in sequences]
        input_ids, attention_mask = self._tokenize_batch(covered_sequences)
        target_input_ids = input_ids.clone()
        masked_input_ids = input_ids.clone()
        row_ids = []
        pos_ids = []
        target_ids = []
        mask_weights = []
        for row_idx, seq in enumerate(covered_sequences):
            budget = min(max(1, int(mask_budget)), len(seq))
            positions = np.random.choice(np.arange(len(seq)), budget, replace=False)
            for pos in positions:
                token_pos = int(pos) + 1
                masked_input_ids[row_idx, token_pos] = self.tokenizer.mask_token_id
                row_ids.append(row_idx)
                pos_ids.append(token_pos)
                target_ids.append(int(target_input_ids[row_idx, token_pos].item()))
                mask_weights.append(float(weights[row_idx]))
        return {
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "ss_input_ids": self.ss_input_ids_1.repeat(len(sequences), 1),
            "row_ids": torch.tensor(row_ids, dtype=torch.long, device=self.device),
            "pos_ids": torch.tensor(pos_ids, dtype=torch.long, device=self.device),
            "target_ids": torch.tensor(target_ids, dtype=torch.long, device=self.device),
            "weights": torch.tensor(mask_weights, dtype=torch.float32, device=self.device),
        }

    def finetune_on_sequences(
        self,
        sequences,
        scores,
        output_dir,
        round_idx,
        epochs,
        lr,
        lambda_kl,
        batch_size,
        mask_budget,
        reward_mode="rank",
        baseline=None,
        negative_weight=0.25,
        bottom_quantile=0.25,
        advantage_clip=2.0,
    ):
        if self.base_model is None:
            raise ValueError("base_model is only loaded when --finetune_prosst is set.")
        if len(sequences) != len(scores):
            raise ValueError("sequences and scores must have the same length.")
        if not sequences:
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "prosst_finetune_metrics.jsonl"
        weights, reward_metadata = reward_weights(
            scores,
            reward_mode,
            baseline=baseline,
            clip=advantage_clip,
            bottom_quantile=bottom_quantile,
        )
        order = np.arange(len(sequences))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        metrics = []
        train_start = time.perf_counter()
        aa_cols = self.full_ids

        for epoch in range(1, epochs + 1):
            self.model.train()
            np.random.shuffle(order)
            epoch_start = time.perf_counter()
            step_metrics = []
            for start in range(0, len(order), batch_size):
                ids = order[start : start + batch_size]
                batch = self._make_finetune_batch(
                    [sequences[i] for i in ids],
                    weights[ids],
                    mask_budget,
                )
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    ss_input_ids=batch["ss_input_ids"],
                ).logits
                with torch.no_grad():
                    base_logits = self.base_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        ss_input_ids=batch["ss_input_ids"],
                    ).logits

                masked_logits = logits[batch["row_ids"], batch["pos_ids"]]
                masked_base_logits = base_logits[batch["row_ids"], batch["pos_ids"]]
                nll = F.cross_entropy(masked_logits, batch["target_ids"], reduction="none")

                log_probs = F.log_softmax(masked_logits[:, aa_cols], dim=-1)
                base_log_probs = F.log_softmax(masked_base_logits[:, aa_cols], dim=-1)
                probs = log_probs.exp()
                kl = (probs * (log_probs - base_log_probs)).sum(dim=-1)

                if reward_mode == "rank":
                    weight = batch["weights"].clamp_min(0.0)
                    denom = weight.sum().clamp_min(1e-12)
                    nll_loss = (nll * weight).sum() / denom
                    effective_positive_weight = float(weight.mean().detach().cpu())
                    effective_negative_weight = 0.0
                else:
                    pos_weight = batch["weights"].clamp_min(0.0)
                    neg_weight = (-batch["weights"]).clamp_min(0.0)
                    signed_weight = pos_weight - float(negative_weight) * neg_weight
                    nll_loss = (nll * signed_weight).mean()
                    weight = batch["weights"]
                    effective_positive_weight = float(pos_weight.mean().detach().cpu())
                    effective_negative_weight = float(neg_weight.mean().detach().cpu())
                kl_loss = kl.mean()
                loss = nll_loss + float(lambda_kl) * kl_loss
                loss.backward()
                grad_norm = global_grad_norm(self.model.parameters())
                optimizer.step()
                step_metrics.append(
                    {
                        "loss": float(loss.detach().cpu()),
                        "weighted_nll": float(nll_loss.detach().cpu()),
                        "kl": float(kl_loss.detach().cpu()),
                        "grad_norm": float(grad_norm),
                        "mean_weight": float(weight.mean().detach().cpu()),
                        "effective_positive_weight": effective_positive_weight,
                        "effective_negative_weight": effective_negative_weight,
                    }
                )

            update_norm, relative_update_norm = parameter_delta_norm(self.model, self.base_model)
            metric = {
                "event": "prosst_finetune_epoch",
                "round": int(round_idx),
                "epoch": int(epoch),
                "epochs": int(epochs),
                "n_sequences": int(len(sequences)),
                "batch_size": int(batch_size),
                "mask_budget": int(mask_budget),
                "lr": float(lr),
                "lambda_kl": float(lambda_kl),
                "reward_mode": reward_mode,
                "negative_weight": float(negative_weight),
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
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metric, sort_keys=True) + "\n")

        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "prosst_finetune_round_complete",
                        "round": int(round_idx),
                        "epochs": int(epochs),
                        "n_sequences": int(len(sequences)),
                        "total_seconds": float(time.perf_counter() - train_start),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        self.model.eval()
        return metrics


def run_seed(args):
    set_seed(args.seed, args.full_deterministic)
    save_dir = Path(args.results_dirpath) / args.task
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"seed_{args.seed}.pkl"
    metadata_path = save_dir / f"seed_{args.seed}.prosst_zero_shot_metadata.json"
    trace_writer = None
    if args.debug_generation_trace:
        trace_writer = JsonlGzWriter(save_dir / "debug_traces" / f"seed_{args.seed}.events.jsonl.gz")

    oracle = get_landscape(args.task)
    dataset = RegressionDataset(args.task)
    tracker = ExperimentTracker(sys.modules[__name__], deepcopy(dataset), WT_SEQUENCES[args.task], best_percentile=0.95)
    generator = ProSSTGenerator(args)
    if trace_writer is not None:
        generator.set_trace_writer(trace_writer)

    metadata = {
        "task": args.task,
        "model": args.model_path,
        "structure": {
            "source": "ProSST precomputed structure tokens",
            "structure_token_name": generator.mapping.structure_token_name,
            "offset": generator.mapping.offset,
            "covered_length": generator.covered_length,
            "identity": generator.mapping.identity,
            "note": generator.mapping.note,
        },
        "generation": {
            "mode": "no_rollout_sequential",
            "mask_strategy": args.mask_strategy,
            "mask_budget": args.mask_budget,
            "smc_vocab": args.smc_vocab,
            "non_cluster_logit_penalty": args.non_cluster_logit_penalty,
            "score": "sum logP(sampled residue)-logP(original residue) under ProSST logits",
            "alphabet": args.alphabet,
            "batch_size": args.batch_size,
        },
        "prosst_finetuning": {
            "enabled": bool(args.finetune_prosst),
            "objective": "rank_weighted_masked_token_nll_plus_kl_current_to_frozen_base",
            "training_corruption": "uniform random fixed-budget masks",
            "mask_budget": args.mask_budget,
            "epochs_per_round": args.finetune_epochs,
            "lr": args.finetune_lr,
            "lambda_kl": args.lambda_kl,
            "batch_size": args.finetune_batch_size,
            "replay": args.finetune_replay,
            "reward_mode": args.reward_mode,
            "negative_weight": args.negative_weight,
            "bottom_quantile": args.bottom_quantile,
            "advantage_clip": args.advantage_clip,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    starting_sequence = WT_SEQUENCES[args.task]
    replay_sequences = []
    replay_scores = []
    finetune_dir = save_dir / f"seed_{args.seed}.prosst_finetune"
    try:
        for iteration in range(1, args.n_iters + 1):
            round_starting_sequence = starting_sequence
            round_baseline_score = (
                float(oracle.get_fitness(np.array([round_starting_sequence]))[0])
                if args.reward_mode == "standardized_advantage"
                else None
            )
            generator.set_trace_context(
                task=args.task,
                seed=args.seed,
                n_queries=args.n_queries,
                optimization_round=iteration,
                method="prosst_zero_shot",
                mask_strategy=args.mask_strategy,
                mask_budget=args.mask_budget,
            )
            candidates = []
            candidate_scores = []
            ref_sequences = list(dataset.train) + list(dataset.valid)
            generation_round = 0
            while len(candidates) < args.n_queries:
                generation_round += 1
                generator.set_trace_context(
                    task=args.task,
                    seed=args.seed,
                    n_queries=args.n_queries,
                    optimization_round=iteration,
                    generation_round=generation_round,
                    method="prosst_zero_shot",
                    mask_strategy=args.mask_strategy,
                    mask_budget=args.mask_budget,
                )
                batch_seqs, batch_scores = generator.generate_batch(starting_sequence, args.batch_size)
                selected = generator.get_top_sequences(batch_seqs, batch_scores, args.n_queries, ref_sequences + candidates)
                selected_scores = {seq: score for seq, score in zip(batch_seqs, batch_scores)}
                candidates.extend(selected)
                candidate_scores.extend([selected_scores[seq] for seq in selected])
            sequences = candidates[: args.n_queries]
            scores = oracle.get_fitness(np.array(sequences)).tolist()
            for query_idx, (sequence, score) in enumerate(zip(sequences, scores), start=1):
                generator.trace_event("oracle_query", optimization_round=iteration, query_rank=query_idx, sequence=sequence, oracle_score=float(score))
            if args.mask_strategy == "mixed_explore_exploit":
                generator.update_mask_position_rewards(starting_sequence, sequences, scores)
            dataset.add((sequences, scores))
            replay_sequences.extend(sequences)
            replay_scores.extend(scores)
            tracker.calculate_top_n_metrics((sequences, scores), iteration, n=100)
            tracker.exp_results[iteration]["Zero-shot"] = metadata
            tracker.save_results(save_path)
            starting_sequence = get_new_starting_seq(dataset)
            if args.finetune_prosst and (iteration < args.n_iters or args.finetune_after_final):
                if args.finetune_replay == "latest":
                    train_sequences = sequences
                    train_scores = scores
                else:
                    train_sequences = replay_sequences
                    train_scores = replay_scores
                ft_start = time.perf_counter()
                ft_metrics = generator.finetune_on_sequences(
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
                    baseline=round_baseline_score,
                    negative_weight=args.negative_weight,
                    bottom_quantile=args.bottom_quantile,
                    advantage_clip=args.advantage_clip,
                )
                tracker.exp_results[iteration]["ProSST fine-tune"] = {
                    "seconds": float(time.perf_counter() - ft_start),
                    "n_sequences": len(train_sequences),
                    "epochs": args.finetune_epochs,
                    "last_epoch": ft_metrics[-1] if ft_metrics else None,
                }
                tracker.save_results(save_path)
    finally:
        if trace_writer is not None:
            trace_writer.close()
            summary_path = trace_writer.path.with_suffix("").with_suffix(".trace_summary.json")
            summary_path.write_text(json.dumps({"event_counts": trace_writer.counts}, indent=2, sort_keys=True), encoding="utf-8")
    print(f"complete task={args.task} seed={args.seed} path={save_path}", flush=True)


def info(msg):
    print(msg, flush=True)


def main():
    args = get_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    run_seed(args)


if __name__ == "__main__":
    main()
