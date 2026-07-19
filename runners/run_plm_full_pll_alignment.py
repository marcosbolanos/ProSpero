#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from prospero.experiments_config import ALPHABETS
from prospero.runners.run_zero_shot_prosst import AA20, ProSSTGenerator, WT_SEQUENCES


DEFAULT_TASKS = ("AAV", "AMIE", "E4B", "GFP", "LGK", "Pab1", "TEM", "UBE2I")
OLD_SCORE_FILES = {
    "evodiff": Path("outputs/evodiff_zero_shot_alignment_20260602/predictive_scores"),
    "esm": Path("outputs/esm2_650m_zero_shot_alignment_20260603/predictive_scores"),
    "prosst": Path("outputs/prosst_zero_shot_alignment_20260603/predictive_scores"),
}
OLD_SCORE_COLUMNS = {
    "evodiff": "masked_marginals",
    "esm": "masked_marginals",
    "prosst": "prosst_log_odds",
}


def get_parser():
    p = argparse.ArgumentParser(description="Compare mutated-position PLM scores with proper whole-sequence PLL.")
    p.add_argument("--out_dir", default="outputs/plm_full_pll_alignment_20260609")
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    p.add_argument("--plms", nargs="+", choices=["evodiff", "esm", "prosst"], default=["evodiff", "esm", "prosst"])
    p.add_argument("--max_sequences", type=int, default=64)
    p.add_argument("--chunk_size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=142857)
    p.add_argument("--esm_model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--prosst_model", default="AI4Protein/ProSST-2048")
    return p


def normalize_sequence(seq):
    return str(seq)


def load_task_frame(task, max_sequences, seed):
    base_path = OLD_SCORE_FILES["esm"] / f"{task}.csv"
    if not base_path.exists():
        base_path = OLD_SCORE_FILES["prosst"] / f"{task}.csv"
    if not base_path.exists():
        base_path = OLD_SCORE_FILES["evodiff"] / f"{task}.csv"
    if not base_path.exists():
        raise FileNotFoundError(f"No old score CSV found for {task}.")

    df = pd.read_csv(base_path)
    df["sequence"] = df["sequence"].map(normalize_sequence)
    df = df.drop_duplicates("sequence", keep="first")
    if max_sequences and len(df) > max_sequences:
        df = df.sample(n=max_sequences, random_state=seed).sort_values("sequence")
    return df[["task", "split", "sequence", "fitness", "n_mutations"]].reset_index(drop=True)


def load_old_scores(task, sequences):
    out = {}
    wanted = pd.DataFrame({"sequence": list(sequences)})
    for plm, folder in OLD_SCORE_FILES.items():
        path = folder / f"{task}.csv"
        column = OLD_SCORE_COLUMNS[plm]
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["sequence"] = df["sequence"].map(normalize_sequence)
        merged = wanted.merge(df[["sequence", column]], on="sequence", how="left")
        out[plm] = merged[column].to_numpy(dtype=float)
    return out


def mutation_log_odds(reference, candidates, reference_log_probs, aa_to_col, covered_length=None):
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
    rho = float(cast(Any, spearmanr(scores[valid], fitness[valid])).statistic) if valid.sum() >= 3 else np.nan
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


class EvoDiffPLL:
    def __init__(self, device):
        from evodiff.pretrained import OA_DM_38M

        model, _, tokenizer, _ = OA_DM_38M()
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.mask_id = tokenizer.mask_id
        self.aa_to_col = {aa: int(tokenizer.tokenize([aa])[0]) for aa in AA20}

    @torch.inference_mode()
    def mms(self, reference, candidates):
        tokens = torch.tensor(np.asarray(self.tokenizer.tokenize([reference]))[None, :], device=self.device)
        timestep = torch.zeros(1, dtype=torch.long, device=self.device)
        logits = self.model(tokens, timestep)[0, :, :20]
        log_probs = F.log_softmax(logits, dim=-1).cpu().numpy()
        return mutation_log_odds(reference, candidates, log_probs, self.aa_to_col)

    @torch.inference_mode()
    def score(self, sequences, chunk_size):
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
            chunk = torch.tensor(np.stack(rows[start : start + chunk_size]), device=self.device)
            timestep = torch.zeros(chunk.shape[0], dtype=torch.long, device=self.device)
            logits = self.model(chunk, timestep)
            chunk_positions = torch.tensor(positions[start : start + chunk_size], device=self.device)
            aa_logits = logits[torch.arange(chunk.shape[0], device=self.device), chunk_positions, :20]
            log_probs = F.log_softmax(aa_logits, dim=1)
            for local, owner in enumerate(owners[start : start + chunk_size]):
                token = int(tokens_by_owner[owner][positions[start + local]])
                scores[owner] += float(log_probs[local, token].detach().cpu())
        return scores


class ESMPLL:
    def __init__(self, model_name, device):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
        self.device = torch.device(device)
        self.aa_to_id = {aa: self.tokenizer.convert_tokens_to_ids(aa) for aa in AA20}
        self.mask_id = self.tokenizer.mask_token_id
        self.aa_ids = torch.tensor([self.aa_to_id[aa] for aa in AA20], device=self.device)
        self.aa_to_col = {aa: idx for idx, aa in enumerate(AA20)}

    @torch.inference_mode()
    def mms(self, reference, candidates):
        encoded = self.tokenizer(reference, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits[0, 1 : len(reference) + 1]
        log_probs = F.log_softmax(logits[:, self.aa_ids], dim=-1).cpu().numpy()
        return mutation_log_odds(reference, candidates, log_probs, self.aa_to_col)

    @torch.inference_mode()
    def score(self, sequences, chunk_size):
        scores = np.zeros(len(sequences), dtype=np.float64)
        rows = []
        positions = []
        owners = []
        for idx, sequence in enumerate(sequences):
            encoded = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
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
                input_ids[local, : len(row)] = torch.tensor(row, dtype=torch.long, device=self.device)
                attention_mask[local, : len(row)] = 1
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            for local, pos in enumerate(positions[start : start + chunk_size]):
                owner, aa = owners[start + local]
                log_probs = F.log_softmax(logits[local, pos], dim=0)
                scores[owner] += float(log_probs[self.aa_to_id[aa]].detach().cpu())
        return scores


class ProSSTPLL:
    def __init__(self, task, device, model_path):
        args = SimpleNamespace(
            task=task,
            device=device,
            model_path=model_path,
            finetune_prosst=False,
            alphabet="CHARGE",
            decoding_vocab="restricted",
            structure_tokens_dir="outputs/prosst_structure_tokens",
            structure_vocab_size="2048",
            entropy_chunk_size=16,
            entropy_quantile=0.5,
            entropy_sigma=None,
            seed_grow_coupling_tau=4.0,
            seed_grow_alpha=1.0,
            seed_grow_beta=1.0,
            mask_budget=4,
        )
        self.generator = ProSSTGenerator(args)

    @torch.inference_mode()
    def mms(self, reference, candidates):
        return self.generator.marginal_mutation_scores(reference, candidates)

    @torch.inference_mode()
    def score(self, sequences, chunk_size):
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
                scores[owner] += float(log_probs[AA20.index(aa)].detach().cpu())
        return scores


def main():
    args = get_parser().parse_args()
    out_dir = Path(args.out_dir)
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
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    samples = {}
    for task in args.tasks:
        sample_path = sample_dir / f"{task}.csv"
        if sample_path.exists():
            frame = pd.read_csv(sample_path)
        else:
            frame = load_task_frame(task, args.max_sequences, args.seed)
            atomic_csv(frame, sample_path)
        persisted_sequences = frame["sequence"].astype(str).tolist()
        if len(frame) != args.max_sequences or len(set(persisted_sequences)) != len(persisted_sequences):
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

    for plm in args.plms:
        shared_scorer = None
        note = "whole covered-region PLL under ProSST; uncovered LGK tail excluded"
        if plm == "evodiff":
            shared_scorer = EvoDiffPLL(args.device)
            note = "EvoDiff OA_DM_38M"
        elif plm == "esm":
            shared_scorer = ESMPLL(args.esm_model, args.device)
            note = args.esm_model
        for task in args.tasks:
            checkpoint = checkpoint_dir / f"{task}__{plm}.json"
            result_path = checkpoint_dir / f"{task}__{plm}.csv"
            if checkpoint.exists() and result_path.exists():
                payload = json.loads(checkpoint.read_text())
                if payload.get("status") == "complete" and len(pd.read_csv(result_path)) == args.max_sequences:
                    print(json.dumps({"event": "resume_skip", "task": task, "plm": plm}), flush=True)
                    continue
            scorer = shared_scorer if shared_scorer is not None else ProSSTPLL(task, args.device, args.prosst_model)
            frame = samples[task].copy()
            sequences = frame["sequence"].astype(str).tolist()
            fitness = frame["fitness"].to_numpy(dtype=float)
            print(json.dumps({"event": "start", "task": task, "plm": plm, "n": len(frame)}), flush=True)
            mms_start = time.perf_counter()
            mms = scorer.mms(WT_SEQUENCES[task], sequences)
            torch.cuda.synchronize()
            mms_seconds = time.perf_counter() - mms_start
            pll_start = time.perf_counter()
            pll = scorer.score(sequences, args.chunk_size)
            torch.cuda.synchronize()
            pll_seconds = time.perf_counter() - pll_start
            frame["mms"] = mms
            frame["pll"] = pll
            metrics = [
                metric_row(task, plm, "mms", mms, fitness, mms_seconds, np.isfinite(mms).sum(), "fixed unmasked WT marginals"),
                metric_row(task, plm, "pll", pll, fitness, pll_seconds, np.isfinite(pll).sum(), note),
            ]
            atomic_csv(frame, result_path)
            checkpoint.write_text(json.dumps({"status": "complete", "metrics": metrics}, indent=2), encoding="utf-8")
            rebuild_summary()
            print(json.dumps({"event": "complete", "task": task, "plm": plm, "mms": metrics[0]["spearman"], "pll": metrics[1]["spearman"]}), flush=True)
            if shared_scorer is None:
                del scorer
                gc.collect()
                torch.cuda.empty_cache()
        if shared_scorer is not None:
            del shared_scorer
            gc.collect()
            torch.cuda.empty_cache()

    summary = rebuild_summary()
    expected = len(args.tasks) * len(args.plms) * 2
    if len(summary) != expected:
        raise RuntimeError(f"Expected {expected} metric rows, found {len(summary)}")
    pivot = summary.pivot_table(index=["plm", "score_type"], columns="task", values="spearman")
    atomic_csv(pivot.reset_index(), out_dir / "spearman_pivot.csv")
    config["status"] = "complete"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "complete.json").write_text(json.dumps({"status": "complete", "metric_rows": len(summary)}, indent=2))
    print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
