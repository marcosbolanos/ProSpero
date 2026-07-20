from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from prospero.optimization.core import (
    gradient_norm,
    model_update_norm,
    standardized_advantages,
)
from prospero.optimization.types import OnlineAdaptationConfig


def make_training_batch(
    tokenizer: Any,
    sequences: Sequence[str],
    advantages: np.ndarray,
    mask_budget: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    input_rows = np.stack([tokenizer.tokenize([sequence]) for sequence in sequences])
    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
    target_ids = input_ids.clone()
    masked_input_ids = input_ids.clone()
    row_ids: list[int] = []
    position_ids: list[int] = []
    target_tokens: list[int] = []
    mask_weights: list[float] = []
    for row_index, sequence in enumerate(sequences):
        budget = min(max(1, int(mask_budget)), len(sequence))
        positions = np.random.choice(np.arange(len(sequence)), budget, replace=False)
        for position in positions:
            position = int(position)
            masked_input_ids[row_index, position] = tokenizer.mask_id
            row_ids.append(row_index)
            position_ids.append(position)
            target_tokens.append(int(target_ids[row_index, position].item()))
            mask_weights.append(float(advantages[row_index]))
    return {
        "input_ids": masked_input_ids,
        "row_ids": torch.tensor(row_ids, dtype=torch.long, device=device),
        "pos_ids": torch.tensor(position_ids, dtype=torch.long, device=device),
        "target_ids": torch.tensor(target_tokens, dtype=torch.long, device=device),
        "weights": torch.tensor(mask_weights, dtype=torch.float32, device=device),
    }


def adapt_evodiff(
    model: torch.nn.Module,
    reference_model: torch.nn.Module,
    tokenizer: Any,
    sequences: Sequence[str],
    fitnesses: Sequence[float],
    output_directory: Path,
    round_index: int,
    config: OnlineAdaptationConfig,
    mask_budget: int,
) -> list[dict[str, Any]]:
    if len(sequences) != len(fitnesses):
        raise ValueError("Sequences and fitnesses must have the same length.")
    if not sequences:
        return []

    output_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = output_directory / "evodiff_adaptation_metrics.jsonl"
    advantages, reward_metadata = standardized_advantages(
        fitnesses,
        maximum_absolute_advantage=config.maximum_absolute_advantage,
    )
    order = np.arange(len(sequences))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    metrics: list[dict[str, Any]] = []
    training_start = time.perf_counter()
    device = next(model.parameters()).device

    for epoch in range(1, config.epochs + 1):
        model.train()
        np.random.shuffle(order)
        epoch_start = time.perf_counter()
        step_metrics = []
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            batch = make_training_batch(
                tokenizer,
                [sequences[int(index)] for index in indices],
                advantages[indices],
                mask_budget,
                device,
            )
            timestep = torch.zeros(
                batch["input_ids"].shape[0], dtype=torch.long, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], timestep)
            with torch.no_grad():
                reference_logits = reference_model(batch["input_ids"], timestep)

            masked_logits = logits[batch["row_ids"], batch["pos_ids"]]
            masked_reference_logits = reference_logits[
                batch["row_ids"], batch["pos_ids"]
            ]
            negative_log_likelihood = F.cross_entropy(
                masked_logits, batch["target_ids"], reduction="none"
            )
            log_probabilities = F.log_softmax(masked_logits[:, :20], dim=-1)
            reference_log_probabilities = F.log_softmax(
                masked_reference_logits[:, :20], dim=-1
            )
            probabilities = log_probabilities.exp()
            kl_divergence = (
                probabilities * (log_probabilities - reference_log_probabilities)
            ).sum(dim=-1)

            positive_weight = batch["weights"].clamp_min(0.0)
            negative_weight = (-batch["weights"]).clamp_min(0.0)
            signed_weight = (
                positive_weight - config.negative_advantage_scale * negative_weight
            )
            reconstruction_loss = (negative_log_likelihood * signed_weight).mean()
            kl_loss = kl_divergence.mean()
            loss = reconstruction_loss + config.kl_coefficient * kl_loss
            loss.backward()
            current_gradient_norm = gradient_norm(model.parameters())
            optimizer.step()
            step_metrics.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "weighted_nll": float(reconstruction_loss.detach().cpu()),
                    "kl": float(kl_loss.detach().cpu()),
                    "grad_norm": current_gradient_norm,
                    "mean_weight": float(batch["weights"].mean().detach().cpu()),
                    "effective_positive_weight": float(
                        positive_weight.mean().detach().cpu()
                    ),
                    "effective_negative_weight": float(
                        negative_weight.mean().detach().cpu()
                    ),
                }
            )

        update_norm, relative_update_norm = model_update_norm(model, reference_model)
        metric = {
            "event": "evodiff_adaptation_epoch",
            "round": int(round_index),
            "epoch": int(epoch),
            "epochs": config.epochs,
            "n_sequences": int(len(sequences)),
            "batch_size": config.batch_size,
            "mask_budget": mask_budget,
            "lr": config.learning_rate,
            "lambda_kl": config.kl_coefficient,
            "objective": "advantage_weighted_masked_online_adaptation",
            "negative_weight": config.negative_advantage_scale,
            "reward_metadata": reward_metadata,
            "seconds": float(time.perf_counter() - epoch_start),
            "loss": float(np.mean([item["loss"] for item in step_metrics])),
            "weighted_nll": float(
                np.mean([item["weighted_nll"] for item in step_metrics])
            ),
            "kl": float(np.mean([item["kl"] for item in step_metrics])),
            "grad_norm": float(np.mean([item["grad_norm"] for item in step_metrics])),
            "max_grad_norm": float(
                np.max([item["grad_norm"] for item in step_metrics])
            ),
            "mean_weight": float(
                np.mean([item["mean_weight"] for item in step_metrics])
            ),
            "effective_positive_weight": float(
                np.mean([item["effective_positive_weight"] for item in step_metrics])
            ),
            "effective_negative_weight": float(
                np.mean([item["effective_negative_weight"] for item in step_metrics])
            ),
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
                    "event": "evodiff_adaptation_round_complete",
                    "round": int(round_index),
                    "epochs": config.epochs,
                    "n_sequences": int(len(sequences)),
                    "total_seconds": float(time.perf_counter() - training_start),
                },
                sort_keys=True,
            )
            + "\n"
        )
    model.eval()
    return metrics
