from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from prospero.dataset import RegressionDataset
from prospero.experiments_config import WT_SEQUENCES
from prospero.landscapes import get_landscape
from prospero.plm.evolutionaryscale import EvolutionaryScaleBackend
from prospero.surrogate import normalize_sequences
from prospero.utils import set_seed

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}


@dataclass
class RidgeBundle:
    scaler: StandardScaler
    ridge: Ridge

    def predict(self, features: np.ndarray | sparse.csr_matrix) -> np.ndarray:
        scaled = self.scaler.transform(features)
        return self.ridge.predict(scaled).astype(np.float32)


def _enumerate_single_mutants(sequence: str) -> list[str]:
    mutants: list[str] = []
    for pos, wt_aa in enumerate(sequence):
        for aa in AA_ALPHABET:
            if aa == wt_aa:
                continue
            chars = list(sequence)
            chars[pos] = aa
            mutants.append("".join(chars))
    return mutants


def _neighbors_single_mutations(sequence: str) -> list[str]:
    return _enumerate_single_mutants(sequence)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _fit_ridge(
    train_features: np.ndarray | sparse.csr_matrix,
    train_labels: np.ndarray,
    alpha: float,
    fit_intercept: bool,
) -> RidgeBundle:
    if sparse.issparse(train_features):
        scaler = StandardScaler(with_mean=False)
    else:
        scaler = StandardScaler()
    x_scaled = scaler.fit_transform(train_features)
    ridge = Ridge(alpha=alpha, fit_intercept=fit_intercept, solver="lsqr")
    ridge.fit(x_scaled, train_labels)
    return RidgeBundle(scaler=scaler, ridge=ridge)


def _sequence_onehot_dense(sequences: list[str], seq_len: int) -> np.ndarray:
    n = len(sequences)
    x = np.zeros((n, seq_len * len(AA_ALPHABET)), dtype=np.float32)
    for i, sequence in enumerate(sequences):
        for pos, aa in enumerate(sequence):
            col = pos * len(AA_ALPHABET) + AA_TO_IDX[aa]
            x[i, col] = 1.0
    return x


def _extract_esm3_flat_embeddings(
    backend: EvolutionaryScaleBackend,
    sequences: list[str],
    seq_len: int,
    progress_prefix: str,
) -> np.ndarray:
    reps = backend.compute_representations(
        sequences,
        representation_name="per_residue_embeddings_v1",
        expected_sequence_length=seq_len,
    )
    arr = reps.numpy().astype(np.float32, copy=False).reshape(len(sequences), -1)
    print(f"[{progress_prefix}] built ESM3 flat embeddings: {arr.shape}", flush=True)
    return arr


def _extract_structure_token_ids(
    backend: EvolutionaryScaleBackend,
    sequences: list[str],
    progress_prefix: str,
) -> tuple[np.ndarray, int]:
    from esm.sdk.api import ESMProtein
    from esm.sdk.api import LogitsConfig

    all_ids: list[np.ndarray] = []
    vocab_size: int | None = None
    token_len: int | None = None
    for i, sequence in enumerate(sequences, start=1):
        protein = ESMProtein(sequence=sequence)
        protein_tensor = backend.model.encode(protein)
        out = backend.model.logits(
            protein_tensor,
            LogitsConfig(sequence=False, structure=True, return_embeddings=False),
        )
        if out.logits is None or out.logits.structure is None:
            raise RuntimeError("ESM3 structure logits missing.")
        logits = out.logits.structure
        token_ids = logits.argmax(dim=-1)[0].detach().cpu().numpy().astype(np.int32)
        all_ids.append(token_ids)
        if vocab_size is None:
            vocab_size = int(logits.shape[-1])
            token_len = int(token_ids.shape[0])
        if i % 250 == 0 or i == len(sequences):
            print(f"[{progress_prefix}] structure ids {i}/{len(sequences)}", flush=True)
    return np.stack(all_ids, axis=0), int(vocab_size or 0)


def _structure_onehot_sparse(token_ids: np.ndarray, vocab_size: int) -> sparse.csr_matrix:
    n, seq_len = token_ids.shape
    row_idx = np.repeat(np.arange(n, dtype=np.int64), seq_len)
    col_idx = np.tile(np.arange(seq_len, dtype=np.int64), n) * vocab_size + token_ids.reshape(-1)
    data = np.ones(row_idx.shape[0], dtype=np.float32)
    return sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n, seq_len * vocab_size))


def _score_with_models(
    sequences: list[str],
    *,
    onehot_model: RidgeBundle,
    esm3_model: RidgeBundle,
    structure_model: RidgeBundle,
    seq_len: int,
    backend: EvolutionaryScaleBackend,
) -> dict[str, np.ndarray]:
    onehot_x = _sequence_onehot_dense(sequences, seq_len)
    esm3_x = _extract_esm3_flat_embeddings(backend, sequences, seq_len, "score")
    struct_ids, struct_vocab = _extract_structure_token_ids(backend, sequences, "score")
    struct_x = _structure_onehot_sparse(struct_ids, struct_vocab)
    return {
        "onehot": onehot_model.predict(onehot_x),
        "esm3": esm3_model.predict(esm3_x),
        "structure": structure_model.predict(struct_x),
    }


def _rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values)
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.int64)
    return ranks


def _aggregate_scores(scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    onehot = scores["onehot"]
    esm3 = scores["esm3"]
    structure = scores["structure"]

    stack = np.stack([onehot, esm3, structure], axis=1)
    z = (stack - stack.mean(axis=0, keepdims=True)) / (stack.std(axis=0, keepdims=True) + 1e-8)
    median_z = np.median(z, axis=1)

    r1 = _rank_desc(onehot)
    r2 = _rank_desc(esm3)
    r3 = _rank_desc(structure)
    rank_agg = -(r1 + r2 + r3).astype(np.float32)

    min_z = np.min(z, axis=1)
    conservative = min_z
    return {
        "median_z": median_z.astype(np.float32),
        "rank_agg": rank_agg.astype(np.float32),
        "conservative": conservative.astype(np.float32),
    }


def _top_k_indices(values: np.ndarray, k: int) -> np.ndarray:
    k = min(max(int(k), 1), len(values))
    idx = np.argpartition(-values, kth=k - 1)[:k]
    idx = idx[np.argsort(-values[idx])]
    return idx


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "LGK 4-stage multi-surrogate search: top-k unions from one-hot/ESM3/"
            "structure, aggregate re-score, local expansion, conservative final selection."
        )
    )
    parser.add_argument("--task", type=str, default="LGK")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--esm-model-name", type=str, default="EvolutionaryScale/esm3-sm-open-v1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--ridge-fit-intercept", action="store_true", default=False)
    parser.add_argument("--top-k-initial", type=int, default=200)
    parser.add_argument("--top-k-seeds", type=int, default=50)
    parser.add_argument("--final-top-k", type=int, default=256)
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["median_z", "rank_agg"],
        default="median_z",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/0426_experiments/lgk_union_multisurrogate_search.json",
    )
    parser.add_argument(
        "--checkpoint-json",
        type=str,
        default="outputs/0426_experiments/lgk_union_multisurrogate_checkpoint.json",
    )
    args = parser.parse_args()

    if args.task != "LGK":
        raise ValueError("This runner is currently scoped to LGK.")

    set_seed(args.seed, False)
    run_start = _now()
    t_global = time.perf_counter()

    checkpoint_path = Path(args.checkpoint_json)
    output_path = Path(args.output_json)
    timings: dict[str, float] = {}

    wt = WT_SEQUENCES[args.task]
    seq_len = len(wt)
    dataset = RegressionDataset(args.task)
    train_sequences = normalize_sequences(dataset.train.tolist())
    train_labels = np.asarray(dataset.train_scores, dtype=np.float32)

    # Stage 0: train all three surrogate models.
    t0 = time.perf_counter()
    backend = EvolutionaryScaleBackend.load(args.esm_model_name, device=args.device)
    train_onehot = _sequence_onehot_dense(train_sequences, seq_len)
    train_esm3 = _extract_esm3_flat_embeddings(backend, train_sequences, seq_len, "train")
    train_struct_ids, struct_vocab = _extract_structure_token_ids(backend, train_sequences, "train")
    train_struct = _structure_onehot_sparse(train_struct_ids, struct_vocab)

    onehot_model = _fit_ridge(
        train_features=train_onehot,
        train_labels=train_labels,
        alpha=args.ridge_alpha,
        fit_intercept=args.ridge_fit_intercept,
    )
    esm3_model = _fit_ridge(
        train_features=train_esm3,
        train_labels=train_labels,
        alpha=args.ridge_alpha,
        fit_intercept=args.ridge_fit_intercept,
    )
    structure_model = _fit_ridge(
        train_features=train_struct,
        train_labels=train_labels,
        alpha=args.ridge_alpha,
        fit_intercept=args.ridge_fit_intercept,
    )
    timings["stage0_train_surrogates_seconds"] = time.perf_counter() - t0

    _save_json(
        checkpoint_path,
        {
            "saved_at_utc": _now(),
            "stage": "stage0_trained",
            "task": args.task,
            "esm_model_name": args.esm_model_name,
            "structure_vocab_size": int(struct_vocab),
            "sequence_length": int(seq_len),
            "timings_seconds": timings,
        },
    )

    # Stage 1: top-200 from each model over all single mutants from WT.
    t0 = time.perf_counter()
    single_mutants = _enumerate_single_mutants(wt)
    single_scores = _score_with_models(
        single_mutants,
        onehot_model=onehot_model,
        esm3_model=esm3_model,
        structure_model=structure_model,
        seq_len=seq_len,
        backend=backend,
    )
    top_onehot = _top_k_indices(single_scores["onehot"], args.top_k_initial)
    top_esm3 = _top_k_indices(single_scores["esm3"], args.top_k_initial)
    top_structure = _top_k_indices(single_scores["structure"], args.top_k_initial)
    union_stage1 = _dedupe(
        [single_mutants[i] for i in top_onehot]
        + [single_mutants[i] for i in top_esm3]
        + [single_mutants[i] for i in top_structure]
    )
    timings["stage1_initial_union_seconds"] = time.perf_counter() - t0

    _save_json(
        checkpoint_path,
        {
            "saved_at_utc": _now(),
            "stage": "stage1_done",
            "task": args.task,
            "top_k_initial": int(args.top_k_initial),
            "union_size": int(len(union_stage1)),
            "timings_seconds": timings,
        },
    )

    # Stage 2: re-score union and aggregate.
    t0 = time.perf_counter()
    stage2_scores = _score_with_models(
        union_stage1,
        onehot_model=onehot_model,
        esm3_model=esm3_model,
        structure_model=structure_model,
        seq_len=seq_len,
        backend=backend,
    )
    stage2_agg = _aggregate_scores(stage2_scores)
    stage2_rank = _top_k_indices(stage2_agg[args.aggregation], len(union_stage1))
    stage2_top50 = [union_stage1[i] for i in stage2_rank[: min(args.top_k_seeds, len(stage2_rank))]]
    timings["stage2_aggregate_rescore_seconds"] = time.perf_counter() - t0

    _save_json(
        checkpoint_path,
        {
            "saved_at_utc": _now(),
            "stage": "stage2_done",
            "task": args.task,
            "top_k_seeds": int(args.top_k_seeds),
            "selected_seed_count": int(len(stage2_top50)),
            "timings_seconds": timings,
        },
    )

    # Stage 3: local expansion around top-50 (all single-mut neighbors around each seed).
    t0 = time.perf_counter()
    expanded = []
    for seed_sequence in stage2_top50:
        expanded.extend(_neighbors_single_mutations(seed_sequence))
    expanded = _dedupe(expanded)
    expanded = [seq for seq in expanded if seq not in set(union_stage1)]
    timings["stage3_local_expansion_seconds"] = time.perf_counter() - t0

    # Stage 4: final conservative selection over union+expanded.
    t0 = time.perf_counter()
    all_candidates = _dedupe(union_stage1 + expanded)
    all_scores = _score_with_models(
        all_candidates,
        onehot_model=onehot_model,
        esm3_model=esm3_model,
        structure_model=structure_model,
        seq_len=seq_len,
        backend=backend,
    )
    all_agg = _aggregate_scores(all_scores)
    final_idx = _top_k_indices(all_agg["conservative"], args.final_top_k)
    final_sequences = [all_candidates[i] for i in final_idx]

    # Oracle only for final shortlist.
    oracle = get_landscape(args.task)
    oracle_vals = np.asarray(oracle.get_fitness(np.asarray(final_sequences)), dtype=np.float32)
    oracle_rank = np.argsort(-oracle_vals)
    timings["stage4_final_selection_seconds"] = time.perf_counter() - t0
    timings["total_seconds"] = time.perf_counter() - t_global

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    rows = []
    for rank_pos, idx in enumerate(final_idx, start=1):
        rows.append(
            {
                "sequence": all_candidates[int(idx)],
                "final_rank_by_conservative": int(rank_pos),
                "score_onehot": float(all_scores["onehot"][int(idx)]),
                "score_esm3": float(all_scores["esm3"][int(idx)]),
                "score_structure": float(all_scores["structure"][int(idx)]),
                "score_median_z": float(all_agg["median_z"][int(idx)]),
                "score_rank_agg": float(all_agg["rank_agg"][int(idx)]),
                "score_conservative_min_z": float(all_agg["conservative"][int(idx)]),
            }
        )

    # Attach oracle ranks for final shortlist.
    oracle_order = np.argsort(-oracle_vals)
    oracle_rank_map = {int(i): r + 1 for r, i in enumerate(oracle_order.tolist())}
    for i, row in enumerate(rows):
        row["oracle_fitness"] = float(oracle_vals[i])
        row["oracle_rank_within_final_shortlist"] = int(oracle_rank_map[i])

    payload = {
        "run_started_utc": run_start,
        "run_completed_utc": _now(),
        "task": args.task,
        "esm_model_name": args.esm_model_name,
        "settings": {
            "top_k_initial": int(args.top_k_initial),
            "top_k_seeds": int(args.top_k_seeds),
            "final_top_k": int(args.final_top_k),
            "aggregation": args.aggregation,
            "ridge_alpha": float(args.ridge_alpha),
            "ridge_fit_intercept": bool(args.ridge_fit_intercept),
        },
        "stage_counts": {
            "num_single_mutants": int(len(single_mutants)),
            "num_stage1_union": int(len(union_stage1)),
            "num_stage2_seeds": int(len(stage2_top50)),
            "num_stage3_expanded_unique": int(len(expanded)),
            "num_stage4_all_candidates": int(len(all_candidates)),
            "num_final_selected": int(len(final_sequences)),
        },
        "timings_seconds": timings,
        "final_rows": rows,
    }
    _save_json(output_path, payload)
    _save_json(
        checkpoint_path,
        {
            "saved_at_utc": _now(),
            "stage": "completed",
            "task": args.task,
            "timings_seconds": timings,
            "output_json": str(output_path),
        },
    )
    print(f"[lgk-union-search] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
