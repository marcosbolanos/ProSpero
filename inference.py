from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as functional
from tqdm import tqdm


class ProteinSampler:
    """Targeted masking and biologically constrained SMC from ProSpero."""

    padding_position = -1

    def __init__(self, model, tokenizer, substitution_alphabet) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.substitution_alphabet = substitution_alphabet
        self.token_clusters = self._token_clusters()
        self.completed_sequence_scores: dict[str, float] = {}

    def _token_clusters(self) -> dict[int, np.ndarray]:
        amino_acid_tokens = {
            amino_acid: token
            for token, amino_acid in enumerate(self.tokenizer.all_aas[:20])
        }
        return {
            amino_acid_tokens[original]: self.tokenizer.tokenize([substitutions])
            for original, substitutions in self.substitution_alphabet.items()
        }

    def top_sequences(self, count: int, excluded_sequences) -> list[str]:
        excluded = set(excluded_sequences)
        ranked = sorted(
            self.completed_sequence_scores,
            key=self.completed_sequence_scores.__getitem__,
            reverse=True,
        )
        selected = [sequence for sequence in ranked if sequence not in excluded][:count]
        self.completed_sequence_scores.clear()
        return selected

    # The original runner historically calls this method name.
    get_top_sequences = top_sequences

    @staticmethod
    def _is_resampling_step(steps_left: int, interval: int | list[int]) -> bool:
        if isinstance(interval, list):
            return steps_left in interval
        return steps_left % interval == 0

    def _resampling_indices(self, scores: torch.Tensor) -> torch.Tensor:
        weights = scores.detach().cpu()
        weights -= weights.min()
        weights += 1e-8
        weights /= weights.sum()
        return torch.multinomial(weights, len(weights), replacement=True).to(
            self.device
        )

    def targeted_masks(
        self,
        sequence: str,
        surrogate,
        minimum_mutations: int,
        maximum_mutations: int,
        batch_size: int,
        scan_multiplier: int,
        ucb_coefficient: float,
    ) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Select alanine-scan variants by surrogate UCB and mask their edits."""
        tokenized_sequence = self.tokenizer.tokenize([sequence])
        maskable_positions = np.flatnonzero(
            np.isin(tokenized_sequence, np.asarray(list(self.token_clusters)))
        )
        variants = []
        mutated_positions = []
        for _ in range(batch_size * scan_multiplier):
            mutation_count = np.random.randint(minimum_mutations, maximum_mutations + 1)
            mutation_count = min(mutation_count, len(maskable_positions))
            positions = np.random.choice(
                maskable_positions, mutation_count, replace=False
            )
            variant = np.asarray(list(sequence))
            variant[positions] = "A"
            variants.append("".join(variant))
            mutated_positions.append(positions)

        mean, standard_deviation = surrogate.forward_with_uncertainty(variants)
        acquisition = (
            (mean + ucb_coefficient * standard_deviation).detach().cpu().numpy()
        )
        selected_indices = np.argsort(acquisition)[::-1][:batch_size]
        masked_batch = np.tile(tokenized_sequence, (batch_size, 1))
        selected_positions = [mutated_positions[index] for index in selected_indices]
        token_choices = []
        ordered_positions = []
        for row, positions in zip(masked_batch, selected_positions):
            positions = np.asarray(
                sorted(
                    positions,
                    key=lambda position: len(self.token_clusters[row[position]]),
                )
            )
            token_choices.append(
                {
                    position: torch.as_tensor(
                        self.token_clusters[row[position]], device=self.device
                    )
                    for position in positions
                }
            )
            row[positions] = self.tokenizer.mask_id
            ordered_positions.append(positions)

        maximum_length = max(map(len, ordered_positions))
        padded_positions = np.asarray(
            [
                np.pad(
                    positions,
                    (0, maximum_length - len(positions)),
                    constant_values=self.padding_position,
                )
                for positions in ordered_positions
            ]
        )
        return (
            torch.as_tensor(masked_batch, device=self.device),
            padded_positions,
            np.asarray(token_choices, dtype=object),
        )

    # The paper and legacy runner use this descriptive name.
    shotgun_alanine_scan = targeted_masks

    @torch.no_grad()
    def generate_raa_from_alanine_scan(
        self,
        guide,
        starting_sequence: str,
        batch_size: int,
        resampling_steps: int | list[int],
        min_corruptions: int,
        max_corruptions: int,
        kappa_scan: float,
        n_checks_multiplier: int,
        kappa_guidance: float,
    ) -> None:
        particles, positions, token_choices = self.targeted_masks(
            starting_sequence,
            guide,
            min_corruptions,
            max_corruptions,
            batch_size,
            n_checks_multiplier,
            kappa_scan,
        )
        steps = positions.shape[1]
        log_likelihood = torch.zeros(batch_size, device=self.device)
        prediction_counts = torch.as_tensor(
            (positions != self.padding_position).sum(axis=1), device=self.device
        )

        for step in tqdm(range(steps)):
            steps_left = steps - step
            active = np.flatnonzero(positions[:, step] != self.padding_position)
            if not len(active):
                break
            timestep = torch.zeros(len(active), dtype=torch.long, device=self.device)
            logits = self.model(particles, timestep)[active, positions[active, step]]
            sampled_tokens = []
            for row_logits, choices, position in zip(
                logits, token_choices[active], positions[active, step]
            ):
                allowed_tokens = choices[position]
                probabilities = functional.softmax(row_logits[allowed_tokens], dim=0)
                sampled_index = torch.multinomial(probabilities, 1)
                sampled_tokens.append(allowed_tokens[sampled_index])
            sampled_tokens = torch.cat(sampled_tokens)
            particles[active, positions[active, step]] = sampled_tokens
            log_probabilities = functional.log_softmax(logits[:, :20], dim=1)
            log_likelihood[active] += log_probabilities[
                torch.arange(len(active), device=self.device), sampled_tokens
            ]

            if not self._is_resampling_step(steps_left, resampling_steps):
                continue
            completed, completed_log_likelihood = self.rollout(
                particles,
                positions[:, step + 1 :],
                token_choices,
                log_likelihood,
            )
            inverse_perplexity = 1 / torch.exp(
                -completed_log_likelihood / prediction_counts
            )
            scores = guide.get_ucb(completed, kappa_guidance)
            if steps_left < 10:
                self.completed_sequence_scores.update(zip(completed, scores))
            parent_indices = self._resampling_indices(scores * inverse_perplexity)
            particles = particles[parent_indices]
            log_likelihood = log_likelihood[parent_indices]
            prediction_counts = prediction_counts[parent_indices]
            parent_indices_numpy = parent_indices.cpu().numpy()
            positions = positions[parent_indices_numpy]
            token_choices = token_choices[parent_indices_numpy]

    @torch.no_grad()
    def rollout(
        self,
        particles: torch.Tensor,
        remaining_positions: np.ndarray,
        token_choices: np.ndarray,
        log_likelihood: torch.Tensor,
    ) -> tuple[list[str], torch.Tensor]:
        particles = deepcopy(particles)
        log_likelihood = deepcopy(log_likelihood)
        if remaining_positions.shape[1] == 0:
            return [self.tokenizer.untokenize(row) for row in particles], log_likelihood

        for step in range(remaining_positions.shape[1]):
            active = np.flatnonzero(
                remaining_positions[:, step] != self.padding_position
            )
            if not len(active):
                break
            timestep = torch.zeros(len(active), dtype=torch.long, device=self.device)
            logits = self.model(particles, timestep)[
                active, remaining_positions[active, step]
            ]
            sampled_tokens = []
            for row_logits, choices, position in zip(
                logits, token_choices[active], remaining_positions[active, step]
            ):
                allowed_tokens = choices[position]
                probabilities = functional.softmax(row_logits[allowed_tokens], dim=0)
                sampled_tokens.append(
                    allowed_tokens[torch.multinomial(probabilities, 1)]
                )
            sampled_tokens = torch.cat(sampled_tokens)
            particles[active, remaining_positions[active, step]] = sampled_tokens
            log_probabilities = functional.log_softmax(logits[:, :20], dim=1)
            log_likelihood[active] += log_probabilities[
                torch.arange(len(active), device=self.device), sampled_tokens
            ]
        return [self.tokenizer.untokenize(row) for row in particles], log_likelihood

    unroll_from_alanine_scan = rollout
