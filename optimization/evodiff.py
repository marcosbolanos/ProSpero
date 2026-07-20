from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from prospero.experiments_config import ALPHABETS, WT_SEQUENCES
from prospero.optimization.core import AMINO_ACIDS, SequenceGenerator
from prospero.optimization.evodiff_adaptation import adapt_evodiff
from prospero.optimization.types import EvoDiffConfig


class EvoDiffModel(SequenceGenerator):
    """EvoDiff sequence generation and online adaptation."""

    def __init__(self, config: EvoDiffConfig):
        from evodiff.pretrained import OA_DM_38M  # type: ignore[reportMissingImports]

        self.config = config
        self.task = config.task
        self.device = torch.device(config.device)
        self.model, _, self.tokenizer, _ = OA_DM_38M()
        self.model = self.model.eval().to(self.device)
        self.base_model = None
        if config.adaptation is not None:
            self.base_model, _, _, _ = OA_DM_38M()
            self.base_model = self.base_model.eval().to(self.device)
            for parameter in self.base_model.parameters():
                parameter.requires_grad_(False)

        self.aa_to_id = {
            aa: int(np.asarray(self.tokenizer.tokenize([aa])).reshape(-1)[0])
            for aa in AMINO_ACIDS
        }
        self.id_to_aa = {token_id: aa for aa, token_id in self.aa_to_id.items()}
        self.full_ids = torch.tensor(
            [self.aa_to_id[aa] for aa in AMINO_ACIDS],
            dtype=torch.long,
            device=self.device,
        )
        self.aa_index = {aa: index for index, aa in enumerate(AMINO_ACIDS)}
        self.alphabet = ALPHABETS[config.substitution_alphabet]
        self.adaptation_enabled = config.adaptation is not None
        self._initialize_search_state()
        self.covered_length = len(WT_SEQUENCES[self.task])

    def _tokenize(self, sequences: list[str]) -> torch.Tensor:
        rows = [
            np.asarray(self.tokenizer.tokenize([sequence]), dtype=np.int64)
            for sequence in sequences
        ]
        return torch.tensor(np.stack(rows), dtype=torch.long, device=self.device)

    def _model_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        timestep = torch.zeros(input_ids.shape[0], dtype=torch.long, device=self.device)
        return self.model(input_ids, timestep)

    @torch.inference_mode()
    def logits_for_sequences(self, sequences: Sequence[str]) -> torch.Tensor:
        return self._model_logits(self._tokenize(list(sequences)))

    @torch.inference_mode()
    def compute_position_entropies(self, sequence, chunk_size=None):
        if chunk_size is None:
            chunk_size = self.config.masking.entropy_chunk_size
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
    def generate_batch(
        self, incumbent: str, batch_size: int
    ) -> tuple[list[str], np.ndarray]:
        masks, _ = self.sample_masks(incumbent, batch_size)
        sequences = [list(incumbent) for _ in range(batch_size)]
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
                original = incumbent[position]
                sampled, log_delta, sampled_ll, logp_sampled, logp_original = (
                    self._sample_from_logits(logits[local, position], original)
                )
                sequences[particle][position] = sampled
                sequential_scores[particle] += float(log_delta.item())
                log_likelihoods[particle] += float(sampled_ll.item())
                self.trace_event(
                    "decode_step",
                    particle=particle,
                    step=step + 1,
                    position=position,
                    distribution=self.config.decoding_vocabulary.value,
                    sampled=sampled,
                    original=original,
                    logp_sampled=float(logp_sampled.item()),
                    logp_original=float(logp_original.item()),
                    log_delta=float(log_delta.item()),
                    full_vocab_logp_sampled=float(sampled_ll.item()),
                )

        candidates = ["".join(sequence) for sequence in sequences]
        mms_scores = self.marginal_mutation_scores(incumbent, candidates)
        for index, (sequence, mms, sequential, ll) in enumerate(
            zip(candidates, mms_scores, sequential_scores, log_likelihoods)
        ):
            self.trace_event(
                "candidate",
                stage="completed_sequence",
                decode_steps=max_steps,
                candidate=index,
                sequence=sequence,
                mms_score=float(mms),
                sequential_decode_score=float(sequential),
                log_likelihood=float(ll),
                inv_perplexity=float(np.exp(ll / max(1, self.config.masking.budget))),
            )
        return candidates, mms_scores

    def adapt(
        self,
        sequences: Sequence[str],
        fitnesses: Sequence[float],
        output_directory: Path,
        round_index: int,
    ) -> list[dict[str, Any]]:
        if self.base_model is None:
            raise ValueError("Online adaptation is disabled.")
        adaptation = self.config.adaptation
        if adaptation is None:
            raise ValueError("Online adaptation is disabled.")
        metrics = adapt_evodiff(
            self.model,
            self.base_model,
            self.tokenizer,
            sequences,
            fitnesses,
            output_directory,
            round_index,
            adaptation,
            self.config.masking.budget,
        )
        self._mms_cache = None
        return metrics
