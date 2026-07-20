from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from prospero.experiments_config import ALPHABETS, WT_SEQUENCES
from prospero.optimization.core import (
    AMINO_ACIDS,
    SequenceGenerator,
    gradient_norm,
    model_update_norm,
    standardized_advantages,
)
from prospero.optimization.types import ProSSTConfig


STRUCTURE_TOKENS_DIR = Path("outputs/prosst_structure_tokens")


@dataclass(frozen=True)
class StructureMapping:
    structure_token_name: str
    offset: int
    covered_length: int
    identity: float
    note: str = ""


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


class ProSSTModel(SequenceGenerator):
    def __init__(self, config: ProSSTConfig):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.config = config
        self.task = config.task
        self.device = torch.device(config.device)
        self.model = (
            AutoModelForMaskedLM.from_pretrained(
                config.model_name_or_path, trust_remote_code=True
            )
            .eval()
            .to(self.device)
        )
        self.base_model = None
        if config.adaptation is not None:
            self.base_model = (
                AutoModelForMaskedLM.from_pretrained(
                    config.model_name_or_path, trust_remote_code=True
                )
                .eval()
                .to(self.device)
            )
            for param in self.base_model.parameters():
                param.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path, trust_remote_code=True
        )
        self.vocab = self.tokenizer.get_vocab()
        self.aa_to_id = {aa: self.vocab[aa] for aa in AMINO_ACIDS}
        self.id_to_aa = {idx: aa for aa, idx in self.aa_to_id.items()}
        self.full_ids = torch.tensor(
            [self.aa_to_id[aa] for aa in AMINO_ACIDS],
            dtype=torch.long,
            device=self.device,
        )
        self.aa_index = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
        self.alphabet = ALPHABETS[config.substitution_alphabet]
        self.adaptation_enabled = config.adaptation is not None
        self._initialize_search_state()

        mapping = TASK_MAPPINGS[self.task]
        self.mapping = mapping
        wt = WT_SEQUENCES[self.task]
        self.full_wt_length = len(wt)
        self.covered_length = min(mapping.covered_length, len(wt))
        structure_path = (
            config.structure_tokens_directory
            / "structure_sequence"
            / config.structure_vocabulary_size
            / f"{mapping.structure_token_name}.fasta"
        )
        structure_tokens = read_structure_tokens(structure_path)
        self.structure_tokens = structure_tokens[
            mapping.offset : mapping.offset + self.covered_length
        ]
        if len(self.structure_tokens) != self.covered_length:
            raise ValueError(f"{self.task} structure token crop length mismatch.")
        self.ss_input_ids_1 = tokenize_structure_tokens(
            self.structure_tokens, self.device
        )

    def _tokenize_batch(self, seqs):
        out = self.tokenizer(seqs, return_tensors="pt", padding=False)
        return out["input_ids"].to(self.device), out["attention_mask"].to(self.device)

    @torch.inference_mode()
    def logits_for_sequences(self, sequences: Sequence[str]) -> torch.Tensor:
        covered = [sequence[: self.covered_length] for sequence in sequences]
        input_ids, attention_mask = self._tokenize_batch(covered)
        return self.logits_for_input_ids(input_ids, attention_mask)

    @torch.inference_mode()
    def logits_for_input_ids(self, input_ids, attention_mask):
        ss = self.ss_input_ids_1.repeat(input_ids.shape[0], 1)
        out = self.model(
            input_ids=input_ids, attention_mask=attention_mask, ss_input_ids=ss
        )
        return out.logits[:, 1:-1, :]

    @torch.inference_mode()
    def compute_position_entropies(self, sequence, chunk_size=None):
        if chunk_size is None:
            chunk_size = self.config.masking.entropy_chunk_size
        sequence = sequence[: self.covered_length]
        input_ids, attention_mask = self._tokenize_batch([sequence])
        rows = input_ids.repeat(len(sequence), 1)
        masks = attention_mask.repeat(len(sequence), 1)
        for pos in range(len(sequence)):
            rows[pos, pos + 1] = self.tokenizer.mask_token_id
        entropies = []
        aa_cols = self.full_ids
        for start in range(0, rows.shape[0], chunk_size):
            logits = self.logits_for_input_ids(
                rows[start : start + chunk_size], masks[start : start + chunk_size]
            )
            positions = torch.arange(
                start, min(start + chunk_size, rows.shape[0]), device=self.device
            )
            local = torch.arange(len(positions), device=self.device)
            selected = logits[local, positions][:, aa_cols]
            log_probs = F.log_softmax(selected, dim=1)
            probs = log_probs.exp()
            entropies.append((-(probs * log_probs).sum(dim=1)).detach().cpu())
        return torch.cat(entropies).numpy()

    def generate_batch(
        self, incumbent: str, batch_size: int
    ) -> tuple[list[str], np.ndarray]:
        masks, _ = self.sample_masks(incumbent, batch_size)
        covered_start = incumbent[: self.covered_length]
        unmodeled_suffix = incumbent[self.covered_length :]
        seqs = [list(covered_start) for _ in range(batch_size)]
        sequential_scores = np.zeros(batch_size, dtype=np.float64)
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
                sampled_aa, log_delta, sampled_ll, logp_sampled, logp_original = (
                    self._sample_from_logits(
                        logits[local_idx, pos],
                        original_aa,
                    )
                )
                seqs[particle][pos] = sampled_aa
                sequential_scores[particle] += float(log_delta.item())
                lls[particle] += float(sampled_ll.item())
                self.trace_event(
                    "decode_step",
                    particle=int(particle),
                    step=int(step + 1),
                    position=int(pos),
                    distribution=self.config.decoding_vocabulary.value,
                    sampled=sampled_aa,
                    original=original_aa,
                    logp_sampled=float(logp_sampled.item()),
                    logp_original=float(logp_original.item()),
                    log_delta=float(log_delta.item()),
                    full_vocab_logp_sampled=float(sampled_ll.item()),
                )
        out = ["".join(seq) + unmodeled_suffix for seq in seqs]
        mms_scores = self.marginal_mutation_scores(incumbent, out)
        for idx, (seq, mms_score, sequential_score, ll) in enumerate(
            zip(out, mms_scores, sequential_scores, lls)
        ):
            self.trace_event(
                "candidate",
                stage="completed_sequence",
                decode_steps=max_steps,
                candidate=int(idx),
                sequence=seq,
                mms_score=float(mms_score),
                sequential_decode_score=float(sequential_score),
                log_likelihood=float(ll),
                inv_perplexity=float(np.exp(ll / max(1, self.config.masking.budget))),
            )
        return out, mms_scores

    def _make_training_batch(self, sequences, weights, mask_budget):
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
            "target_ids": torch.tensor(
                target_ids, dtype=torch.long, device=self.device
            ),
            "weights": torch.tensor(
                mask_weights, dtype=torch.float32, device=self.device
            ),
        }

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
        if len(sequences) != len(fitnesses):
            raise ValueError("Sequences and fitnesses must have the same length.")
        if not sequences:
            return []

        output_directory.mkdir(parents=True, exist_ok=True)
        metrics_path = output_directory / "prosst_adaptation_metrics.jsonl"
        weights, reward_metadata = standardized_advantages(
            fitnesses,
            maximum_absolute_advantage=adaptation.maximum_absolute_advantage,
        )
        order = np.arange(len(sequences))
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=adaptation.learning_rate
        )
        metrics = []
        train_start = time.perf_counter()
        aa_cols = self.full_ids
        for epoch in range(1, adaptation.epochs + 1):
            self.model.train()
            np.random.shuffle(order)
            epoch_start = time.perf_counter()
            step_metrics = []
            batch_iter = []
            for start in range(0, len(order), adaptation.batch_size):
                ids = order[start : start + adaptation.batch_size]
                batch_iter.append(
                    self._make_training_batch(
                        [sequences[i] for i in ids],
                        weights[ids],
                        self.config.masking.budget,
                    )
                )
            for batch in batch_iter:
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
                nll = F.cross_entropy(
                    masked_logits, batch["target_ids"], reduction="none"
                )
                log_probs = F.log_softmax(masked_logits[:, aa_cols], dim=-1)
                base_log_probs = F.log_softmax(masked_base_logits[:, aa_cols], dim=-1)
                probs = log_probs.exp()
                kl = (probs * (log_probs - base_log_probs)).sum(dim=-1)

                pos_weight = batch["weights"].clamp_min(0.0)
                neg_weight = (-batch["weights"]).clamp_min(0.0)
                signed_weight = (
                    pos_weight - adaptation.negative_advantage_scale * neg_weight
                )
                nll_loss = (nll * signed_weight).mean()
                weight = batch["weights"]
                effective_positive_weight = float(pos_weight.mean().detach().cpu())
                effective_negative_weight = float(neg_weight.mean().detach().cpu())
                kl_loss = kl.mean()
                loss = nll_loss + adaptation.kl_coefficient * kl_loss
                loss.backward()
                grad_norm = gradient_norm(self.model.parameters())
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

            update_norm, relative_update_norm = model_update_norm(
                self.model, self.base_model
            )
            metric = {
                "event": "prosst_adaptation_epoch",
                "round": int(round_index),
                "epoch": int(epoch),
                "epochs": adaptation.epochs,
                "n_sequences": int(len(sequences)),
                "batch_size": adaptation.batch_size,
                "mask_budget": self.config.masking.budget,
                "lr": adaptation.learning_rate,
                "lambda_kl": adaptation.kl_coefficient,
                "objective": "advantage_weighted_masked_online_adaptation",
                "negative_weight": adaptation.negative_advantage_scale,
                "reward_metadata": reward_metadata,
                "seconds": float(time.perf_counter() - epoch_start),
                "loss": float(np.mean([m["loss"] for m in step_metrics])),
                "weighted_nll": float(
                    np.mean([m["weighted_nll"] for m in step_metrics])
                ),
                "kl": float(np.mean([m["kl"] for m in step_metrics])),
                "grad_norm": float(np.mean([m["grad_norm"] for m in step_metrics])),
                "max_grad_norm": float(np.max([m["grad_norm"] for m in step_metrics])),
                "mean_weight": float(np.mean([m["mean_weight"] for m in step_metrics])),
                "effective_positive_weight": float(
                    np.mean([m["effective_positive_weight"] for m in step_metrics])
                ),
                "effective_negative_weight": float(
                    np.mean([m["effective_negative_weight"] for m in step_metrics])
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
                        "event": "prosst_adaptation_round_complete",
                        "round": int(round_index),
                        "epochs": adaptation.epochs,
                        "n_sequences": int(len(sequences)),
                        "total_seconds": float(time.perf_counter() - train_start),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        self.model.eval()
        self._mms_cache = None
        return metrics
