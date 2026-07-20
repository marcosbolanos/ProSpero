from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Iterable, Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from prospero.optimization.types import DecodingVocabulary, ModelConfig


AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")


class ReferenceDataset(Protocol):
    train: Any
    valid: Any
    train_scores: Any
    valid_scores: Any


@dataclass
class CampaignState:
    """The wild-type baseline and observations available to optimization."""

    incumbent_sequence: str
    incumbent_fitness: float
    queried_sequences: set[str] = field(default_factory=set)
    baseline_sequence: str = field(init=False)

    def __post_init__(self) -> None:
        self.baseline_sequence = self.incumbent_sequence

    @property
    def excluded_sequences(self) -> set[str]:
        return self.queried_sequences | {self.baseline_sequence}

    def observe(self, sequences: Sequence[str], fitnesses: Sequence[float]) -> None:
        if not sequences:
            raise ValueError("Cannot update a campaign without observations.")
        if len(sequences) != len(fitnesses):
            raise ValueError("Sequences and fitnesses must have the same length.")
        self.queried_sequences.update(sequences)
        best_index = int(np.argmax(fitnesses))
        best_fitness = float(fitnesses[best_index])
        if best_fitness > self.incumbent_fitness:
            self.incumbent_sequence = sequences[best_index]
            self.incumbent_fitness = best_fitness


def load_wild_type_fitness(
    reference_dataset: ReferenceDataset, wild_type: str
) -> float:
    """Read only the known wild-type baseline from the benchmark data."""

    sequences = np.concatenate(
        (reference_dataset.train, reference_dataset.valid), axis=0
    )
    fitnesses = np.concatenate(
        (reference_dataset.train_scores, reference_dataset.valid_scores), axis=0
    )
    matches = np.asarray(
        ["".join(map(str, sequence)) == wild_type for sequence in sequences]
    )
    if matches.sum() != 1:
        raise ValueError(
            f"Expected exactly one wild-type baseline, found {int(matches.sum())}."
        )
    return float(fitnesses[matches][0])


class TraceWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = gzip.open(self.path, "at", encoding="utf-8")
        self.counts: dict[str, int] = {}

    def write(self, record: dict[str, Any]) -> None:
        event = str(record.get("event", "unknown"))
        self.counts[event] = self.counts.get(event, 0) + 1
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")

    def close(self) -> None:
        self._handle.close()


def standardized_advantages(
    fitnesses: Sequence[float], maximum_absolute_advantage: float | None = 2.0
) -> tuple[np.ndarray, dict[str, float | str]]:
    values = np.asarray(fitnesses, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot compute advantages for an empty fitness list.")
    center = float(np.mean(values))
    scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    advantages = (values - center) / scale
    if maximum_absolute_advantage is not None:
        advantages = np.clip(
            advantages,
            -float(maximum_absolute_advantage),
            float(maximum_absolute_advantage),
        )
    return advantages.astype(np.float32), {
        "reward_mode": "advantage_weighted",
        "baseline": center,
        "baseline_mode": "group_mean",
        "scale": scale,
        "weight_min": float(np.min(advantages)),
        "weight_max": float(np.max(advantages)),
        "weight_mean": float(np.mean(advantages)),
        "weight_std": float(np.std(advantages)),
        "frac_negative": float(np.mean(advantages < 0)),
    }


def gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        squared_norm += float(torch.sum(gradient * gradient).item())
    return squared_norm**0.5


@torch.no_grad()
def model_update_norm(
    model: torch.nn.Module, reference_model: torch.nn.Module
) -> tuple[float, float]:
    squared_update_norm = 0.0
    squared_reference_norm = 0.0
    for parameter, reference_parameter in zip(
        model.parameters(), reference_model.parameters()
    ):
        difference = parameter.detach() - reference_parameter.detach()
        squared_update_norm += float(torch.sum(difference * difference).item())
        squared_reference_norm += float(
            torch.sum(
                reference_parameter.detach() * reference_parameter.detach()
            ).item()
        )
    update_norm = squared_update_norm**0.5
    reference_norm = squared_reference_norm**0.5
    return update_norm, update_norm / max(reference_norm, 1e-12)


class SequenceGenerator:
    """Mask selection and MMS ranking shared by masked protein models."""

    config: ModelConfig
    device: torch.device
    covered_length: int
    full_ids: torch.Tensor
    aa_to_id: dict[str, int]
    id_to_aa: dict[int, str]
    aa_index: dict[str, int]
    alphabet: Mapping[str, Sequence[str]]

    def _initialize_search_state(self) -> None:
        self.trace_writer: TraceWriter | None = None
        self.trace_context: dict[str, Any] = {}
        self.mask_position_reward_sum: dict[int, float] = {}
        self.mask_position_reward_count: dict[int, int] = {}
        self.mask_position_sample_count: dict[int, int] = {}
        self._mms_cache: tuple[str, np.ndarray] | None = None

    def set_trace_writer(self, writer: TraceWriter) -> None:
        self.trace_writer = writer

    def set_trace_context(self, **context: Any) -> None:
        self.trace_context = {
            key: value for key, value in context.items() if value is not None
        }

    def trace_event(self, event: str, **payload: Any) -> None:
        if self.trace_writer is not None:
            self.trace_writer.write({"event": event, **self.trace_context, **payload})

    def logits_for_sequences(self, sequences: Sequence[str]) -> torch.Tensor:
        raise NotImplementedError

    def compute_position_entropies(
        self, sequence: str, chunk_size: int | None = None
    ) -> np.ndarray:
        raise NotImplementedError

    def middle_entropy_scores(
        self, entropies: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float]]:
        preferred_entropy = float(
            np.quantile(entropies, self.config.masking.entropy_quantile)
        )
        bandwidth = self.config.masking.entropy_bandwidth
        if bandwidth is None:
            bandwidth = float(np.std(entropies))
            if bandwidth <= 1e-8:
                bandwidth = 1.0
        scores = (
            np.exp(-((entropies - preferred_entropy) ** 2) / (2 * bandwidth**2)) + 1e-12
        )
        scores = scores / scores.sum()
        return scores, {
            "entropy_quantile": float(self.config.masking.entropy_quantile),
            "entropy_h_star": preferred_entropy,
            "entropy_sigma": float(bandwidth),
            "entropy_min": float(np.min(entropies)),
            "entropy_max": float(np.max(entropies)),
            "entropy_mean": float(np.mean(entropies)),
        }

    def sample_seed_and_grow_mask(
        self,
        positions: np.ndarray,
        base_scores: np.ndarray,
        mask_budget: int,
    ) -> np.ndarray:
        selected = [int(np.random.choice(positions, p=base_scores))]
        positions = np.asarray(positions)
        base_scores = np.asarray(base_scores)
        while len(selected) < mask_budget:
            remaining = ~np.isin(positions, selected)
            remaining_positions = positions[remaining]
            remaining_base_scores = base_scores[remaining]
            distances = np.abs(
                remaining_positions[:, None] - np.asarray(selected)[None, :]
            )
            coupling = np.exp(
                -distances / max(self.config.masking.seed_grow_coupling_length, 1e-6)
            ).max(axis=1)
            scores = (
                self.config.masking.seed_grow_alpha * remaining_base_scores
                + self.config.masking.seed_grow_beta * coupling
            )
            scores = np.maximum(scores, 1e-12)
            scores = scores / scores.sum()
            selected.append(int(np.random.choice(remaining_positions, p=scores)))
        return np.asarray(selected, dtype=int)

    def sample_mixed_mask(
        self,
        positions: np.ndarray,
        entropy_scores: np.ndarray,
        mask_budget: int,
    ) -> np.ndarray:
        positions = np.asarray(positions)
        selected: list[int] = []

        def remaining() -> np.ndarray:
            return ~np.isin(positions, selected)

        def choose(position_ids: np.ndarray, scores: np.ndarray) -> int:
            probabilities = np.maximum(np.asarray(scores, dtype=float), 1e-12)
            probabilities = probabilities / probabilities.sum()
            return int(np.random.choice(position_ids, p=probabilities))

        exploitation_budget = max(1, mask_budget // 2)
        entropy_budget = 1 if mask_budget - exploitation_budget > 0 else 0
        rewards = np.asarray(
            [
                self.mask_position_reward_sum.get(int(position), 0.0)
                / max(1, self.mask_position_reward_count.get(int(position), 0))
                for position in positions
            ],
            dtype=float,
        )
        positive_rewards = np.maximum(rewards, 0.0)
        if positive_rewards.sum() <= 1e-12:
            selected.extend(
                self.sample_seed_and_grow_mask(
                    positions,
                    entropy_scores,
                    min(exploitation_budget, mask_budget),
                ).tolist()
            )
        else:
            for _ in range(min(exploitation_budget, mask_budget)):
                available = remaining()
                scores = positive_rewards[available]
                if scores.sum() <= 1e-12:
                    scores = entropy_scores[available]
                selected.append(choose(positions[available], scores))
        for _ in range(entropy_budget):
            available = remaining()
            if len(selected) < mask_budget:
                selected.append(choose(positions[available], entropy_scores[available]))
        while len(selected) < mask_budget:
            available = remaining()
            available_positions = positions[available]
            counts = np.asarray(
                [
                    self.mask_position_sample_count.get(int(position), 0)
                    for position in available_positions
                ],
                dtype=float,
            )
            selected.append(choose(available_positions, 1.0 / np.sqrt(1.0 + counts)))
        return np.asarray(selected, dtype=int)

    def sample_masks(
        self, sequence: str, batch_size: int
    ) -> tuple[list[np.ndarray], dict[str, float]]:
        covered_sequence = sequence[: self.covered_length]
        mask_budget = min(
            max(1, int(self.config.masking.budget)), len(covered_sequence)
        )
        positions = np.arange(len(covered_sequence))
        entropies = self.compute_position_entropies(sequence)
        entropy_scores, entropy_metadata = self.middle_entropy_scores(entropies)
        masks = []
        for particle in range(batch_size):
            sampled = self.sample_mixed_mask(positions, entropy_scores, mask_budget)
            sampled = np.asarray(sorted(sampled), dtype=int)
            for position in sampled:
                position = int(position)
                self.mask_position_sample_count[position] = (
                    self.mask_position_sample_count.get(position, 0) + 1
                )
            self.trace_event(
                "mask_selected",
                particle=particle,
                strategy="mixed_position_entropy_anticollapse",
                starting_sequence=sequence,
                mask_positions=sampled.tolist(),
                mask_residues=[covered_sequence[position] for position in sampled],
                mask_size=len(sampled),
                entropy_metadata=entropy_metadata,
            )
            masks.append(sampled)
        return masks, entropy_metadata

    def _distribution_ids(self, original_amino_acid: str) -> torch.Tensor:
        if self.config.decoding_vocabulary is DecodingVocabulary.UNRESTRICTED:
            return self.full_ids
        return torch.tensor(
            [
                self.aa_to_id[amino_acid]
                for amino_acid in self.alphabet[original_amino_acid]
            ],
            dtype=torch.long,
            device=self.device,
        )

    def _sample_from_logits(
        self, logits: torch.Tensor, original_amino_acid: str
    ) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution_ids = self._distribution_ids(original_amino_acid)
        distribution_logits = logits[distribution_ids]
        distribution_log_probabilities = F.log_softmax(distribution_logits, dim=0)
        sampled_index = torch.multinomial(
            distribution_log_probabilities.exp(), num_samples=1
        )
        sampled_id = distribution_ids[sampled_index].flatten()[0]
        original_id = torch.tensor(
            self.aa_to_id[original_amino_acid],
            dtype=torch.long,
            device=self.device,
        )
        original_index = torch.nonzero(
            distribution_ids == original_id, as_tuple=False
        ).flatten()
        if original_index.numel() == 0:
            raise ValueError(
                f"Original amino acid {original_amino_acid} is absent from "
                "the decoding distribution."
            )
        sampled_log_probability = distribution_log_probabilities[
            sampled_index
        ].flatten()[0]
        original_log_probability = distribution_log_probabilities[
            original_index[:1]
        ].flatten()[0]
        full_log_probabilities = F.log_softmax(logits[self.full_ids], dim=0)
        sampled_amino_acid = self.id_to_aa[int(sampled_id.item())]
        return (
            sampled_amino_acid,
            sampled_log_probability - original_log_probability,
            full_log_probabilities[AMINO_ACIDS.index(sampled_amino_acid)],
            sampled_log_probability,
            original_log_probability,
        )

    @torch.inference_mode()
    def marginal_mutation_scores(
        self, incumbent: str, candidates: Sequence[str]
    ) -> np.ndarray:
        """Score mutations using one unmasked incumbent context."""

        covered_incumbent = incumbent[: self.covered_length]
        if self._mms_cache is None or self._mms_cache[0] != covered_incumbent:
            logits = self.logits_for_sequences([covered_incumbent])[0]
            log_probabilities = F.log_softmax(logits[:, self.full_ids], dim=-1)
            self._mms_cache = (
                covered_incumbent,
                log_probabilities.detach().cpu().numpy(),
            )

        reference_log_probabilities = self._mms_cache[1]
        original_ids = np.asarray(
            [self.aa_index[amino_acid] for amino_acid in covered_incumbent],
            dtype=np.int64,
        )
        positions = np.arange(len(covered_incumbent))
        scores = np.zeros(len(candidates), dtype=np.float64)
        for index, sequence in enumerate(candidates):
            covered_candidate = sequence[: self.covered_length]
            if len(covered_candidate) != len(covered_incumbent):
                raise ValueError(
                    "MMS requires candidates and incumbents with equal covered length."
                )
            candidate_ids = np.asarray(
                [self.aa_index[amino_acid] for amino_acid in covered_candidate],
                dtype=np.int64,
            )
            mutated = candidate_ids != original_ids
            scores[index] = np.sum(
                reference_log_probabilities[positions[mutated], candidate_ids[mutated]]
                - reference_log_probabilities[
                    positions[mutated], original_ids[mutated]
                ],
                dtype=np.float64,
            )
        return scores

    def update_mask_position_rewards(
        self,
        incumbent: str,
        sequences: Sequence[str],
        fitnesses: Sequence[float],
    ) -> None:
        baseline = float(np.mean(fitnesses))
        for sequence, fitness in zip(sequences, fitnesses):
            advantage = float(fitness) - baseline
            for position, (before, after) in enumerate(zip(incumbent, sequence)):
                if before == after:
                    continue
                self.mask_position_reward_sum[position] = (
                    self.mask_position_reward_sum.get(position, 0.0) + advantage
                )
                self.mask_position_reward_count[position] = (
                    self.mask_position_reward_count.get(position, 0) + 1
                )
