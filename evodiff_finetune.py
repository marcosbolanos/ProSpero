import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


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


def standardized_advantages(
    scores,
    baseline,
    scale_scores=None,
    clip=2.0,
):
    scores = np.asarray(scores, dtype=float)
    scale_source = scores if scale_scores is None else np.asarray(scale_scores, dtype=float)
    scale = float(np.std(scale_source))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    advantages = (scores - float(baseline)) / scale
    if clip is not None:
        advantages = np.clip(advantages, -float(clip), float(clip))
    return advantages.astype(np.float32), {
        "baseline": float(baseline),
        "scale": float(scale),
        "advantage_min": float(np.min(advantages)) if advantages.size else 0.0,
        "advantage_max": float(np.max(advantages)) if advantages.size else 0.0,
        "advantage_mean": float(np.mean(advantages)) if advantages.size else 0.0,
        "advantage_std": float(np.std(advantages)) if advantages.size else 0.0,
        "frac_positive_advantage": float(np.mean(advantages > 0)) if advantages.size else 0.0,
        "frac_negative_advantage": float(np.mean(advantages < 0)) if advantages.size else 0.0,
    }


def group_relative_advantages(scores, clip=2.0):
    """GRPO-style per-round reward normalization."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("Cannot normalize an empty score list.")
    mean = float(np.mean(scores))
    scale = float(np.std(scores))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    advantages = (scores - mean) / scale
    if clip is not None:
        advantages = np.clip(advantages, -float(clip), float(clip))
    return advantages.astype(np.float32), {
        "baseline": mean,
        "baseline_mode": "group_mean",
        "scale": scale,
        "advantage_min": float(np.min(advantages)),
        "advantage_max": float(np.max(advantages)),
        "advantage_mean": float(np.mean(advantages)),
        "advantage_std": float(np.std(advantages)),
        "frac_positive_advantage": float(np.mean(advantages > 0)),
        "frac_negative_advantage": float(np.mean(advantages < 0)),
    }


def bottom_quantile_rank_advantages(scores, bottom_quantile=0.25):
    """
    Rank-positive rewards with explicit negatives only for the bottom quantile.
    This keeps the successful rank-FT imitation signal while suppressing the
    worst candidates from each oracle batch.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("Cannot rank an empty score list.")
    weights = normalized_rank_weights(scores).astype(np.float32)
    threshold = float(np.quantile(scores, bottom_quantile))
    advantages = weights.copy()
    bottom_mask = scores <= threshold
    if np.any(bottom_mask):
        advantages[bottom_mask] = -1.0
    return advantages.astype(np.float32), {
        "baseline": threshold,
        "baseline_mode": f"bottom_quantile_{bottom_quantile:g}",
        "bottom_quantile": float(bottom_quantile),
        "advantage_min": float(np.min(advantages)),
        "advantage_max": float(np.max(advantages)),
        "advantage_mean": float(np.mean(advantages)),
        "advantage_std": float(np.std(advantages)),
        "frac_positive_advantage": float(np.mean(advantages > 0)),
        "frac_negative_advantage": float(np.mean(advantages < 0)),
    }


def _tokenize_sequences(tokenizer, sequences):
    return np.asarray([tokenizer.tokenize([seq]) for seq in sequences], dtype=np.int64)


def _sample_mask_positions(tokenized, maskable_tokens, mask_budget):
    positions = []
    for row in tokenized:
        maskable = np.nonzero(np.isin(row, maskable_tokens))[0]
        if maskable.size == 0:
            raise ValueError("Encountered a sequence with no maskable residues.")
        budget = int(min(max(1, mask_budget), maskable.size))
        positions.append(np.random.choice(maskable, budget, replace=False))
    return positions


def _make_batch(tokenizer, sequences, weights, maskable_tokens, mask_budget, device):
    tokenized = _tokenize_sequences(tokenizer, sequences)
    positions = _sample_mask_positions(tokenized, maskable_tokens, mask_budget)
    masked = tokenized.copy()
    row_ids = []
    col_ids = []
    targets = []
    per_mask_weights = []
    for row_idx, pos in enumerate(positions):
        masked[row_idx, pos] = tokenizer.mask_id
        row_ids.extend([row_idx] * len(pos))
        col_ids.extend(pos.tolist())
        targets.extend(tokenized[row_idx, pos].tolist())
        per_mask_weights.extend([weights[row_idx]] * len(pos))

    return {
        "masked": torch.tensor(masked, dtype=torch.long, device=device),
        "row_ids": torch.tensor(row_ids, dtype=torch.long, device=device),
        "col_ids": torch.tensor(col_ids, dtype=torch.long, device=device),
        "targets": torch.tensor(targets, dtype=torch.long, device=device),
        "weights": torch.tensor(per_mask_weights, dtype=torch.float32, device=device),
    }


def _global_grad_norm(parameters):
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total += float(torch.sum(grad * grad).item())
    return math.sqrt(total)


@torch.no_grad()
def _parameter_delta_norm(model, base_model):
    delta_sq = 0.0
    base_sq = 0.0
    for param, base_param in zip(model.parameters(), base_model.parameters()):
        diff = param.detach() - base_param.detach()
        delta_sq += float(torch.sum(diff * diff).item())
        base_sq += float(torch.sum(base_param.detach() * base_param.detach()).item())
    delta = math.sqrt(delta_sq)
    base = math.sqrt(base_sq)
    return delta, delta / max(base, 1e-12)


def _weighted_mean(values, weights):
    weights = weights.clamp_min(0.0)
    denom = weights.sum()
    if float(denom.detach().cpu()) <= 1e-12:
        return values.mean()
    return (values * weights).sum() / denom


def train_evodiff_reward_weighted_nll(
    model,
    base_model,
    tokenizer,
    alphabet,
    sequences,
    scores,
    output_dir,
    round_idx,
    epochs=5,
    lr=1e-5,
    lambda_kl=2.0,
    batch_size=16,
    mask_budget=4,
    log_prefix="evodiff_finetune",
    reward_mode="rank",
    sequence_weights=None,
    negative_weight=0.25,
    reward_metadata=None,
):
    """
    Fine-tune EvoDiff on evaluated sequences with reward-weighted masked-token
    NLL plus KL(current || frozen_base) on the same masked contexts.
    """
    if len(sequences) != len(scores):
        raise ValueError("sequences and scores must have the same length.")
    if not sequences:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "evodiff_finetune_metrics.jsonl"

    device = next(model.parameters()).device
    maskable_tokens = np.array(
        [tokenizer.tokenize([aa]) for aa in alphabet.keys()],
        dtype=np.int64,
    )
    if sequence_weights is None:
        weights = normalized_rank_weights(scores)
        reward_mode = "rank"
    else:
        weights = np.asarray(sequence_weights, dtype=np.float32)
        if len(weights) != len(sequences):
            raise ValueError("sequence_weights must match sequences length.")
    order = np.arange(len(sequences))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad_(False)

    epoch_metrics = []
    train_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(order)
        epoch_start = time.perf_counter()
        step_metrics = []
        for start in range(0, len(order), batch_size):
            batch_ids = order[start : start + batch_size]
            batch = _make_batch(
                tokenizer,
                [sequences[i] for i in batch_ids],
                weights[batch_ids],
                maskable_tokens,
                mask_budget,
                device,
            )
            timestep = torch.zeros(
                batch["masked"].shape[0],
                dtype=torch.long,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["masked"], timestep)
            with torch.no_grad():
                base_logits = base_model(batch["masked"], timestep)

            masked_logits = logits[
                batch["row_ids"],
                batch["col_ids"],
                :20,
            ]
            masked_base_logits = base_logits[
                batch["row_ids"],
                batch["col_ids"],
                :20,
            ]
            targets = batch["targets"]
            weights_mask = batch["weights"]

            nll = F.cross_entropy(masked_logits, targets, reduction="none")
            log_probs = F.log_softmax(masked_logits, dim=-1)
            base_log_probs = F.log_softmax(masked_base_logits, dim=-1)
            probs = log_probs.exp()
            kl = (probs * (log_probs - base_log_probs)).sum(dim=-1)

            if reward_mode in {
                "standardized_advantage",
                "grpo_advantage",
                "bottom_quantile_negative",
            }:
                pos_weights = weights_mask.clamp_min(0.0)
                neg_weights = (-weights_mask).clamp_min(0.0)
                signed_weights = pos_weights - float(negative_weight) * neg_weights
                nll_loss = (signed_weights * nll).mean()
                effective_positive_weight = float(pos_weights.mean().detach().cpu())
                effective_negative_weight = float(neg_weights.mean().detach().cpu())
            else:
                nll_loss = _weighted_mean(nll, weights_mask)
                effective_positive_weight = float(weights_mask.clamp_min(0.0).mean().detach().cpu())
                effective_negative_weight = 0.0
            kl_loss = kl.mean()
            loss = nll_loss + lambda_kl * kl_loss
            loss.backward()
            grad_norm = _global_grad_norm(model.parameters())
            optimizer.step()

            step_metrics.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "weighted_nll": float(nll_loss.detach().cpu()),
                    "kl": float(kl_loss.detach().cpu()),
                    "grad_norm": grad_norm,
                    "effective_positive_weight": effective_positive_weight,
                    "effective_negative_weight": effective_negative_weight,
                }
            )

        update_norm, relative_update_norm = _parameter_delta_norm(model, base_model)
        epoch_seconds = time.perf_counter() - epoch_start
        metric = {
            "event": "evodiff_finetune_epoch",
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
            "seconds": float(epoch_seconds),
            "loss": float(np.mean([m["loss"] for m in step_metrics])),
            "weighted_nll": float(np.mean([m["weighted_nll"] for m in step_metrics])),
            "kl": float(np.mean([m["kl"] for m in step_metrics])),
            "grad_norm": float(np.mean([m["grad_norm"] for m in step_metrics])),
            "max_grad_norm": float(np.max([m["grad_norm"] for m in step_metrics])),
            "effective_positive_weight": float(np.mean([m["effective_positive_weight"] for m in step_metrics])),
            "effective_negative_weight": float(np.mean([m["effective_negative_weight"] for m in step_metrics])),
            "update_norm": float(update_norm),
            "relative_update_norm": float(relative_update_norm),
        }
        epoch_metrics.append(metric)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric, sort_keys=True) + "\n")

    total_metric = {
        "event": "evodiff_finetune_round_complete",
        "round": int(round_idx),
        "epochs": int(epochs),
        "n_sequences": int(len(sequences)),
        "total_seconds": float(time.perf_counter() - train_start),
    }
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(total_metric, sort_keys=True) + "\n")
    model.eval()
    return epoch_metrics
