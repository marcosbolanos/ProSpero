#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from prospero.dataset import RegressionDataset
from prospero.experiments_config import WT_SEQUENCES
from prospero.optimization.core import AMINO_ACIDS
from prospero.optimization.prosst import ProSSTModel
from prospero.optimization.types import (
    DecodingVocabulary,
    MaskingConfig,
    ProSSTConfig,
)


DEFAULT_TASKS = ("AAV", "AMIE", "E4B", "GFP", "LGK", "Pab1", "TEM", "UBE2I")


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare marginal mutation scores with whole-sequence PLL."
    )
    p.add_argument("--output-directory", required=True)
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    p.add_argument(
        "--models",
        nargs="+",
        choices=["evodiff", "esm", "prosst"],
        default=["evodiff", "esm", "prosst"],
    )
    p.add_argument("--max-sequences", dest="max_sequences", type=int, default=64)
    p.add_argument("--chunk-size", dest="chunk_size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=142857)
    p.add_argument(
        "--esm-model",
        dest="esm_model",
        default="facebook/esm2_t33_650M_UR50D",
    )
    p.add_argument(
        "--prosst-model",
        dest="prosst_model",
        default="AI4Protein/ProSST-2048",
    )
    p.add_argument(
        "--structure-tokens-directory",
        dest="structure_tokens_directory",
        default="assets/prosst_structure_tokens",
    )
    return p


def normalize_sequence(seq):
    return str(seq)


def load_task_frame(task, max_sequences, seed):
    dataset = RegressionDataset(task)
    sequences = [normalize_sequence(sequence) for sequence in dataset.valid]
    fitness = np.asarray(dataset.valid_scores, dtype=float)
    wild_type = WT_SEQUENCES[task]
    df = pd.DataFrame(
        {
            "task": task,
            "split": "valid",
            "sequence": sequences,
            "fitness": fitness,
            "n_mutations": [
                sum(a != b for a, b in zip(sequence, wild_type))
                for sequence in sequences
            ],
        }
    )
    df = df.drop_duplicates("sequence", keep="first")
    if max_sequences and len(df) > max_sequences:
        df = df.sample(n=max_sequences, random_state=seed).sort_values("sequence")
    return df[["task", "split", "sequence", "fitness", "n_mutations"]].reset_index(
        drop=True
    )


def marginal_mutation_scores(
    reference, candidates, reference_log_probs, aa_to_col, covered_length=None
):
    """Score mutations from one fixed, unmasked reference distribution."""
    length = len(reference) if covered_length is None else covered_length
    reference = reference[:length]
    reference_ids = np.asarray([aa_to_col[aa] for aa in reference], dtype=np.int64)
    positions = np.arange(length)
    scores = np.zeros(len(candidates), dtype=np.float64)
    for owner, candidate in enumerate(candidates):
        candidate = candidate[:length]
        if len(candidate) != length:
            raise ValueError("MMS requires equal reference and candidate lengths.")
        candidate_ids = np.asarray([aa_to_col[aa] for aa in candidate], dtype=np.int64)
        mutated = candidate_ids != reference_ids
        scores[owner] = np.sum(
            reference_log_probs[positions[mutated], candidate_ids[mutated]]
            - reference_log_probs[positions[mutated], reference_ids[mutated]],
            dtype=np.float64,
        )
    return scores


def atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def metric_row(task, plm, score_type, scores, fitness, seconds, n_scored, note=""):
    valid = np.isfinite(scores) & np.isfinite(fitness)
    rho = (
        float(cast(Any, spearmanr(scores[valid], fitness[valid])).statistic)
        if valid.sum() >= 3
        else np.nan
    )
    return {
        "task": task,
        "plm": plm,
        "score_type": score_type,
        "spearman": float(rho) if np.isfinite(rho) else np.nan,
        "n_sequences": int(len(scores)),
        "n_scored": int(n_scored),
        "seconds": float(seconds),
        "note": note,
    }


class EvoDiffScorer:
    def __init__(self, device):
        from evodiff.pretrained import OA_DM_38M

        model, _, tokenizer, _ = OA_DM_38M()
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.mask_id = tokenizer.mask_id
        self.aa_to_col = {aa: int(tokenizer.tokenize([aa])[0]) for aa in AMINO_ACIDS}

    @torch.inference_mode()
    def marginal_mutation_scores(self, reference, candidates):
        tokens = torch.tensor(
            np.asarray(self.tokenizer.tokenize([reference]))[None, :],
            device=self.device,
        )
        timestep = torch.zeros(1, dtype=torch.long, device=self.device)
        logits = self.model(tokens, timestep)[0, :, :20]
        log_probs = F.log_softmax(logits, dim=-1).cpu().numpy()
        return marginal_mutation_scores(
            reference, candidates, log_probs, self.aa_to_col
        )

    @torch.inference_mode()
    def pseudo_log_likelihood(self, sequences, chunk_size):
        scores = np.zeros(len(sequences), dtype=np.float64)
        rows = []
        positions = []
        owners = []
        tokens_by_owner = []
        for idx, sequence in enumerate(sequences):
            tokens = np.asarray(self.tokenizer.tokenize([sequence]), dtype=np.int64)
            tokens_by_owner.append(tokens)
            for pos, token in enumerate(tokens):
                if 0 <= int(token) < 20:
                    row = tokens.copy()
                    row[pos] = self.mask_id
                    rows.append(row)
                    positions.append(pos)
                    owners.append(idx)
        for start in range(0, len(rows), chunk_size):
            chunk = torch.tensor(
                np.stack(rows[start : start + chunk_size]), device=self.device
            )
            timestep = torch.zeros(chunk.shape[0], dtype=torch.long, device=self.device)
            logits = self.model(chunk, timestep)
            chunk_positions = torch.tensor(
                positions[start : start + chunk_size], device=self.device
            )
            aa_logits = logits[
                torch.arange(chunk.shape[0], device=self.device), chunk_positions, :20
            ]
            log_probs = F.log_softmax(aa_logits, dim=1)
            for local, owner in enumerate(owners[start : start + chunk_size]):
                token = int(tokens_by_owner[owner][positions[start + local]])
                scores[owner] += float(log_probs[local, token].detach().cpu())
        return scores


class ESMScorer:
    def __init__(self, model_name, device):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
        self.device = torch.device(device)
        self.aa_to_id = {
            aa: self.tokenizer.convert_tokens_to_ids(aa) for aa in AMINO_ACIDS
        }
        self.mask_id = self.tokenizer.mask_token_id
        self.aa_ids = torch.tensor(
            [self.aa_to_id[aa] for aa in AMINO_ACIDS], device=self.device
        )
        self.aa_to_col = {aa: index for index, aa in enumerate(AMINO_ACIDS)}

    @torch.inference_mode()
    def marginal_mutation_scores(self, reference, candidates):
        encoded = self.tokenizer(
            reference, return_tensors="pt", add_special_tokens=True
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits[
            0, 1 : len(reference) + 1
        ]
        log_probs = F.log_softmax(logits[:, self.aa_ids], dim=-1).cpu().numpy()
        return marginal_mutation_scores(
            reference, candidates, log_probs, self.aa_to_col
        )

    @torch.inference_mode()
    def pseudo_log_likelihood(self, sequences, chunk_size):
        scores = np.zeros(len(sequences), dtype=np.float64)
        rows = []
        positions = []
        owners = []
        for idx, sequence in enumerate(sequences):
            encoded = self.tokenizer(
                sequence, return_tensors="pt", add_special_tokens=True
            )
            ids = encoded["input_ids"][0].numpy()
            for pos, aa in enumerate(sequence):
                if aa not in self.aa_to_id:
                    continue
                row = ids.copy()
                row[pos + 1] = self.mask_id
                rows.append(row)
                positions.append(pos + 1)
                owners.append((idx, aa))
        for start in range(0, len(rows), chunk_size):
            max_len = max(len(row) for row in rows[start : start + chunk_size])
            input_ids = torch.full(
                (min(chunk_size, len(rows) - start), max_len),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for local, row in enumerate(rows[start : start + chunk_size]):
                input_ids[local, : len(row)] = torch.tensor(
                    row, dtype=torch.long, device=self.device
                )
                attention_mask[local, : len(row)] = 1
            logits = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            for local, pos in enumerate(positions[start : start + chunk_size]):
                owner, aa = owners[start + local]
                log_probs = F.log_softmax(logits[local, pos], dim=0)
                scores[owner] += float(log_probs[self.aa_to_id[aa]].detach().cpu())
        return scores


class ProSSTScorer:
    def __init__(self, task, device, model_path, structure_tokens_dir):
        config = ProSSTConfig(
            task=task,
            device=device,
            masking=MaskingConfig(),
            decoding_vocabulary=DecodingVocabulary.RESTRICTED,
            substitution_alphabet="CHARGE",
            adaptation=None,
            model_name_or_path=model_path,
            structure_tokens_directory=Path(structure_tokens_dir),
        )
        self.generator = ProSSTModel(config)

    @torch.inference_mode()
    def marginal_mutation_scores(self, reference, candidates):
        return self.generator.marginal_mutation_scores(reference, candidates)

    @torch.inference_mode()
    def pseudo_log_likelihood(self, sequences, chunk_size):
        gen = self.generator
        scores = np.zeros(len(sequences), dtype=np.float64)
        rows = []
        positions = []
        owners = []
        for idx, sequence in enumerate(sequences):
            covered = sequence[: gen.covered_length]
            for pos, aa in enumerate(covered):
                if aa not in gen.aa_to_id:
                    continue
                rows.append(covered)
                positions.append(pos)
                owners.append((idx, aa))
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start : start + chunk_size]
            chunk_positions = positions[start : start + chunk_size]
            input_ids, attention_mask = gen._tokenize_batch(chunk)
            for local, pos in enumerate(chunk_positions):
                input_ids[local, pos + 1] = gen.tokenizer.mask_token_id
            logits = gen.logits_for_input_ids(input_ids, attention_mask)
            for local, pos in enumerate(chunk_positions):
                owner, aa = owners[start + local]
                log_probs = F.log_softmax(logits[local, pos, gen.full_ids], dim=0)
                scores[owner] += float(log_probs[AMINO_ACIDS.index(aa)].detach().cpu())
        return scores


def main():
    args = get_parser().parse_args()
    out_dir = Path(args.output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = out_dir / "samples"
    checkpoint_dir = out_dir / "checkpoints"
    sample_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    config_path = out_dir / "run_config.json"
    config = {
        **vars(args),
        "status": "running",
        "mms_definition": "mutation log-odds from one fixed unmasked WT forward pass",
        "pll_definition": "sum of leave-one-position-out conditional log likelihoods",
    }
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )

    samples = {}
    for task in args.tasks:
        sample_path = sample_dir / f"{task}.csv"
        if sample_path.exists():
            frame = pd.read_csv(sample_path)
        else:
            frame = load_task_frame(task, args.max_sequences, args.seed)
            atomic_csv(frame, sample_path)
        persisted_sequences = frame["sequence"].astype(str).tolist()
        if len(frame) != args.max_sequences or len(set(persisted_sequences)) != len(
            persisted_sequences
        ):
            raise ValueError(f"Invalid persisted sample at {sample_path}")
        samples[task] = frame

    def rebuild_summary():
        rows = []
        for path in sorted(checkpoint_dir.glob("*.json")):
            payload = json.loads(path.read_text())
            if payload.get("status") == "complete":
                rows.extend(payload["metrics"])
        summary = pd.DataFrame(rows)
        if not summary.empty:
            summary = summary.sort_values(["task", "plm", "score_type"])
            atomic_csv(summary, out_dir / "summary_long.csv")
        return summary

    for plm in args.models:
        shared_scorer = None
        note = "whole covered-region PLL under ProSST; uncovered LGK tail excluded"
        if plm == "evodiff":
            shared_scorer = EvoDiffScorer(args.device)
            note = "EvoDiff OA_DM_38M"
        elif plm == "esm":
            shared_scorer = ESMScorer(args.esm_model, args.device)
            note = args.esm_model
        for task in args.tasks:
            checkpoint = checkpoint_dir / f"{task}__{plm}.json"
            result_path = checkpoint_dir / f"{task}__{plm}.csv"
            if checkpoint.exists() and result_path.exists():
                payload = json.loads(checkpoint.read_text())
                if (
                    payload.get("status") == "complete"
                    and len(pd.read_csv(result_path)) == args.max_sequences
                ):
                    print(
                        json.dumps({"event": "resume_skip", "task": task, "plm": plm}),
                        flush=True,
                    )
                    continue
            scorer = (
                shared_scorer
                if shared_scorer is not None
                else ProSSTScorer(
                    task,
                    args.device,
                    args.prosst_model,
                    args.structure_tokens_directory,
                )
            )
            frame = samples[task].copy()
            sequences = frame["sequence"].astype(str).tolist()
            fitness = frame["fitness"].to_numpy(dtype=float)
            print(
                json.dumps(
                    {"event": "start", "task": task, "plm": plm, "n": len(frame)}
                ),
                flush=True,
            )
            mms_start = time.perf_counter()
            mms = scorer.marginal_mutation_scores(WT_SEQUENCES[task], sequences)
            torch.cuda.synchronize()
            mms_seconds = time.perf_counter() - mms_start
            pll_start = time.perf_counter()
            pll = scorer.pseudo_log_likelihood(sequences, args.chunk_size)
            torch.cuda.synchronize()
            pll_seconds = time.perf_counter() - pll_start
            frame["mms"] = mms
            frame["pll"] = pll
            metrics = [
                metric_row(
                    task,
                    plm,
                    "mms",
                    mms,
                    fitness,
                    mms_seconds,
                    np.isfinite(mms).sum(),
                    "fixed unmasked WT marginals",
                ),
                metric_row(
                    task,
                    plm,
                    "pll",
                    pll,
                    fitness,
                    pll_seconds,
                    np.isfinite(pll).sum(),
                    note,
                ),
            ]
            atomic_csv(frame, result_path)
            checkpoint.write_text(
                json.dumps({"status": "complete", "metrics": metrics}, indent=2),
                encoding="utf-8",
            )
            rebuild_summary()
            print(
                json.dumps(
                    {
                        "event": "complete",
                        "task": task,
                        "plm": plm,
                        "mms": metrics[0]["spearman"],
                        "pll": metrics[1]["spearman"],
                    }
                ),
                flush=True,
            )
            if shared_scorer is None:
                del scorer
                gc.collect()
                torch.cuda.empty_cache()
        if shared_scorer is not None:
            del shared_scorer
            gc.collect()
            torch.cuda.empty_cache()

    summary = rebuild_summary()
    expected = len(args.tasks) * len(args.models) * 2
    if len(summary) != expected:
        raise RuntimeError(f"Expected {expected} metric rows, found {len(summary)}")
    pivot = summary.pivot_table(
        index=["plm", "score_type"], columns="task", values="spearman"
    )
    atomic_csv(pivot.reset_index(), out_dir / "spearman_pivot.csv")
    config["status"] = "complete"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "complete.json").write_text(
        json.dumps({"status": "complete", "metric_rows": len(summary)}, indent=2)
    )
    print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
