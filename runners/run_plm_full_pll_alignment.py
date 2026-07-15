#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from prospero.experiments_config import ALPHABETS
from prospero.runners.run_zero_shot_prosst import AA20, ProSSTGenerator


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
    p.add_argument("--chunk_size", type=int, default=64)
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


def metric_row(task, plm, score_type, scores, fitness, seconds, n_scored, note=""):
    valid = np.isfinite(scores) & np.isfinite(fitness)
    rho = spearmanr(scores[valid], fitness[valid]).statistic if valid.sum() >= 3 else np.nan
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
    summaries = []
    old_rows = []

    shared_models = {}
    for task in args.tasks:
        frame = load_task_frame(task, args.max_sequences, args.seed)
        sequences = frame["sequence"].tolist()
        fitness = frame["fitness"].to_numpy(dtype=float)
        old_scores = load_old_scores(task, sequences)
        task_scores = frame.copy()

        for plm, scores in old_scores.items():
            task_scores[f"{plm}_old_estimate"] = scores
            summaries.append(metric_row(task, plm, "old_estimate", scores, fitness, 0.0, np.isfinite(scores).sum()))

        for plm in args.plms:
            if plm == "evodiff":
                scorer = shared_models.get("evodiff")
                if scorer is None:
                    scorer = EvoDiffPLL(args.device)
                    shared_models["evodiff"] = scorer
                note = "whole-sequence PLL under EvoDiff OA_DM_38M"
            elif plm == "esm":
                scorer = shared_models.get("esm")
                if scorer is None:
                    scorer = ESMPLL(args.esm_model, args.device)
                    shared_models["esm"] = scorer
                note = f"whole-sequence PLL under {args.esm_model}"
            elif plm == "prosst":
                scorer = ProSSTPLL(task, args.device, args.prosst_model)
                note = "whole covered-region PLL under ProSST; LGK unstructured tail excluded"
            else:
                raise ValueError(plm)

            torch.cuda.empty_cache()
            start_time = time.perf_counter()
            scores = scorer.score(sequences, args.chunk_size)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start_time
            task_scores[f"{plm}_proper_full_pll"] = scores
            summaries.append(metric_row(task, plm, "proper_full_pll", scores, fitness, seconds, np.isfinite(scores).sum(), note))
            task_scores.to_csv(out_dir / f"{task}.csv", index=False)
            pd.DataFrame(summaries).to_csv(out_dir / "summary_long.csv", index=False)

    summary = pd.DataFrame(summaries)
    summary.to_csv(out_dir / "summary_long.csv", index=False)
    pivot = summary.pivot_table(index=["plm", "score_type"], columns="task", values="spearman")
    pivot.to_csv(out_dir / "spearman_pivot.csv")
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
