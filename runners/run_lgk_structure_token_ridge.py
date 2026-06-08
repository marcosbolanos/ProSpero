from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from prospero.dataset import RegressionDataset
from prospero.experiments_config import WT_SEQUENCES
from prospero.plm.evolutionaryscale import EvolutionaryScaleBackend
from prospero.surrogate import normalize_sequences
from prospero.utils import set_seed

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class PredictedSingleMutantRecord:
    mutant: str
    position: int
    wt_residue: str
    mutant_residue: str
    predicted_fitness: float
    predicted_delta_fitness: float
    predicted_delta_energy: float


def _enumerate_single_mutants(wt_sequence: str) -> tuple[list[str], list[int], list[str]]:
    mutants: list[str] = []
    positions: list[int] = []
    mutant_residues: list[str] = []
    for idx, wt_residue in enumerate(wt_sequence):
        for aa in AA_ALPHABET:
            if aa == wt_residue:
                continue
            seq_chars = list(wt_sequence)
            seq_chars[idx] = aa
            mutants.append("".join(seq_chars))
            positions.append(idx)
            mutant_residues.append(aa)
    return mutants, positions, mutant_residues


def _extract_structure_token_ids_and_logits_embeddings(
    backend: EvolutionaryScaleBackend,
    sequences: list[str],
    proxy_batch_size: int,
) -> tuple[np.ndarray, int]:
    from esm.sdk.api import ESMProtein
    from esm.sdk.api import LogitsConfig

    all_token_ids: list[np.ndarray] = []
    token_length: int | None = None
    structure_vocab_size: int | None = None

    for i, sequence in enumerate(sequences, start=1):
        protein = ESMProtein(sequence=sequence)
        protein_tensor = backend.model.encode(protein)
        output = backend.model.logits(
            protein_tensor,
            LogitsConfig(sequence=True, structure=True, return_embeddings=True),
        )

        if output.logits is None or output.logits.structure is None:
            raise RuntimeError("ESM3 did not return structure logits.")
        if output.embeddings is None:
            raise RuntimeError("ESM3 did not return embeddings.")

        structure_logits = output.logits.structure
        if structure_logits.ndim != 3 or structure_logits.shape[0] != 1:
            raise ValueError(
                f"Unexpected structure logits shape: {tuple(structure_logits.shape)}"
            )
        token_ids = structure_logits.argmax(dim=-1)[0].detach().cpu().numpy().astype(np.int32)

        if token_length is None:
            token_length = int(token_ids.shape[0])
        elif int(token_ids.shape[0]) != token_length:
            raise ValueError(
                "Inconsistent structure token lengths across sequences. "
                f"Expected {token_length}, got {token_ids.shape[0]}."
            )

        if structure_vocab_size is None:
            structure_vocab_size = int(structure_logits.shape[-1])
        elif int(structure_logits.shape[-1]) != structure_vocab_size:
            raise ValueError(
                "Inconsistent structure vocab size across sequences. "
                f"Expected {structure_vocab_size}, got {structure_logits.shape[-1]}."
            )

        all_token_ids.append(token_ids)

        if i % max(proxy_batch_size, 1) == 0 or i == len(sequences):
            print(
                f"[structure-ablation-ridge] extraction progress: {i}/{len(sequences)}",
                flush=True,
            )

    return (
        np.stack(all_token_ids, axis=0),
        int(structure_vocab_size or 0),
    )


def _save_extraction_checkpoint(
    *,
    checkpoint_dir: Path,
    sequence_token_ids: np.ndarray,
    structure_embed_table: np.ndarray,
    structure_vocab_size: int,
    extraction_seconds: float,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    np.save(checkpoint_dir / "sequence_token_ids.npy", sequence_token_ids)
    np.save(checkpoint_dir / "structure_embed_table.npy", structure_embed_table.astype(np.float32))
    metadata = {
        "format_version": "lgk_structure_feature_ablation_v1",
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_sequences": int(sequence_token_ids.shape[0]),
        "structure_token_length": int(sequence_token_ids.shape[1]),
        "structure_vocab_size": int(structure_vocab_size),
        "structure_embed_table_shape": [
            int(structure_embed_table.shape[0]),
            int(structure_embed_table.shape[1]),
        ],
        "extraction_seconds": float(extraction_seconds),
    }
    (checkpoint_dir / "extraction_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    (checkpoint_dir / "extraction_complete.json").write_text(
        json.dumps(
            {
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "format_version": "lgk_structure_feature_ablation_v1",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_extraction_checkpoint(
    checkpoint_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    token_ids_path = checkpoint_dir / "sequence_token_ids.npy"
    embed_table_path = checkpoint_dir / "structure_embed_table.npy"
    metadata_path = checkpoint_dir / "extraction_metadata.json"
    complete_path = checkpoint_dir / "extraction_complete.json"
    if not (
        token_ids_path.exists()
        and embed_table_path.exists()
        and metadata_path.exists()
        and complete_path.exists()
    ):
        raise FileNotFoundError(
            "Missing extraction checkpoint artifacts. Expected: "
            "sequence_token_ids.npy, structure_embed_table.npy, "
            "extraction_metadata.json, extraction_complete.json."
        )
    sequence_token_ids = np.load(token_ids_path)
    structure_embed_table = np.load(embed_table_path).astype(np.float32, copy=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return sequence_token_ids, structure_embed_table, metadata


def _build_full_sequence_onehot_features(token_ids: np.ndarray, vocab_size: int) -> sparse.csr_matrix:
    n_samples, seq_len = token_ids.shape
    if token_ids.min() < 0 or token_ids.max() >= vocab_size:
        raise ValueError(
            f"Token ids out of range for vocab_size={vocab_size}. "
            f"min={int(token_ids.min())}, max={int(token_ids.max())}."
        )

    row_indices = np.repeat(np.arange(n_samples, dtype=np.int64), seq_len)
    col_indices = (
        np.tile(np.arange(seq_len, dtype=np.int64), n_samples) * vocab_size
        + token_ids.reshape(-1).astype(np.int64)
    )
    values = np.ones(row_indices.shape[0], dtype=np.float32)
    shape = (n_samples, seq_len * vocab_size)
    return sparse.csr_matrix((values, (row_indices, col_indices)), shape=shape)


def _fit_and_score_ridge(
    *,
    train_features: np.ndarray | sparse.csr_matrix,
    train_labels: np.ndarray,
    mutant_features: np.ndarray | sparse.csr_matrix,
    wt_feature: np.ndarray | sparse.csr_matrix,
    alpha: float,
    fit_intercept: bool,
) -> tuple[np.ndarray, float, dict[str, float]]:
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    if sparse.issparse(train_features):
        scaler = StandardScaler(with_mean=False)
        scaled_train = scaler.fit_transform(train_features)
        scaled_mutant = scaler.transform(mutant_features)
        scaled_wt = scaler.transform(wt_feature)
    else:
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(train_features)
        scaled_mutant = scaler.transform(mutant_features)
        scaled_wt = scaler.transform(wt_feature.reshape(1, -1))
    timings["fit_scaler_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ridge = Ridge(alpha=alpha, fit_intercept=fit_intercept, solver="lsqr")
    ridge.fit(scaled_train, train_labels)
    timings["fit_ridge_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mutant_pred = ridge.predict(scaled_mutant).astype(np.float32)
    wt_pred = float(ridge.predict(scaled_wt)[0])
    timings["predict_seconds"] = time.perf_counter() - t0
    return mutant_pred, wt_pred, timings


def _build_model_result(
    *,
    model_name: str,
    wt_sequence: str,
    mutants: list[str],
    positions: list[int],
    mutant_residues: list[str],
    predicted_fitness: np.ndarray,
    predicted_fitness_wt: float,
    timings_seconds: dict[str, float],
    feature_shape: tuple[int, int],
) -> dict[str, Any]:
    predicted_delta_fitness = predicted_fitness - predicted_fitness_wt
    predicted_delta_energy = -predicted_delta_fitness
    best_idx = int(np.argmin(predicted_delta_energy))

    rows: list[PredictedSingleMutantRecord] = []
    for i, mutant in enumerate(mutants):
        pos = int(positions[i])
        rows.append(
            PredictedSingleMutantRecord(
                mutant=mutant,
                position=pos,
                wt_residue=wt_sequence[pos],
                mutant_residue=mutant_residues[i],
                predicted_fitness=float(predicted_fitness[i]),
                predicted_delta_fitness=float(predicted_delta_fitness[i]),
                predicted_delta_energy=float(predicted_delta_energy[i]),
            )
        )

    return {
        "model_name": model_name,
        "feature_shape": {
            "num_sequences": int(feature_shape[0]),
            "num_features": int(feature_shape[1]),
        },
        "wt_predicted_fitness": float(predicted_fitness_wt),
        "best_predicted_delta_energy": asdict(rows[best_idx]),
        "timings_seconds": timings_seconds,
        "per_mutant_rows": [asdict(row) for row in rows],
    }


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "LGK surrogate-only structure ablation using ESM3 structure embeddings "
            "(BOS/EOS/BOS+EOS) and full concatenated one-hot structure tokens."
        )
    )
    parser.add_argument("--task", type=str, default="LGK")
    parser.add_argument(
        "--esm-model-name",
        type=str,
        default="EvolutionaryScale/esm3-sm-open-v1",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--proxy-batch-size", type=int, default=16)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--ridge-fit-intercept", action="store_true", default=False)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "extract", "fit"],
        default="all",
        help="all: extract then fit, extract: only checkpoint embeddings/tokens, fit: use existing checkpoints only",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="outputs/0424_experiments/lgk_structure_feature_ablation_checkpoints",
    )
    parser.add_argument(
        "--force-reextract",
        action="store_true",
        default=False,
        help="Force extraction even if extraction_complete.json exists.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/0424_experiments/lgk_structure_feature_ablation_compact.json",
    )
    parser.add_argument(
        "--output-json-full",
        type=str,
        default="outputs/0424_experiments/lgk_structure_feature_ablation_full.json",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def main() -> None:
    args = get_parser().parse_args()
    run_start = datetime.now(timezone.utc).isoformat()
    set_seed(args.seed, False)

    if args.task != "LGK":
        raise ValueError("This runner is currently scoped to LGK only.")

    wt_sequence = WT_SEQUENCES[args.task]
    dataset = RegressionDataset(args.task)
    train_sequences = normalize_sequences(dataset.train.tolist())
    train_labels = np.asarray(dataset.train_scores, dtype=np.float32)

    mutants, positions, mutant_residues = _enumerate_single_mutants(wt_sequence)

    unique_sequences = list(dict.fromkeys(train_sequences + mutants + [wt_sequence]))
    checkpoint_dir = Path(args.checkpoint_dir)
    extraction_complete_path = checkpoint_dir / "extraction_complete.json"

    if args.mode in {"all", "extract"} and (
        args.force_reextract or not extraction_complete_path.exists()
    ):
        print(
            "[structure-ablation-ridge] loading ESM backend "
            f"{args.esm_model_name} on {args.device}",
            flush=True,
        )
        backend = EvolutionaryScaleBackend.load(args.esm_model_name, device=args.device)
        if not hasattr(backend.model, "encoder") or not hasattr(
            backend.model.encoder, "structure_tokens_embed"
        ):
            raise RuntimeError("ESM3 model does not expose encoder.structure_tokens_embed.")

        structure_embed_table = (
            backend.model.encoder.structure_tokens_embed.weight.detach().float().cpu().numpy()
        )
        t0 = time.perf_counter()
        sequence_token_ids, structure_vocab_size = _extract_structure_token_ids_and_logits_embeddings(
            backend,
            unique_sequences,
            proxy_batch_size=args.proxy_batch_size,
        )
        extraction_seconds = time.perf_counter() - t0
        _save_extraction_checkpoint(
            checkpoint_dir=checkpoint_dir,
            sequence_token_ids=sequence_token_ids,
            structure_embed_table=structure_embed_table,
            structure_vocab_size=structure_vocab_size,
            extraction_seconds=extraction_seconds,
        )
        print(
            f"[structure-ablation-ridge] wrote extraction checkpoints to {checkpoint_dir}",
            flush=True,
        )
        del backend
        gc.collect()
    elif args.mode in {"all", "extract"}:
        print(
            f"[structure-ablation-ridge] extraction checkpoint already exists at {checkpoint_dir}; reuse",
            flush=True,
        )

    if args.mode == "extract":
        print("[structure-ablation-ridge] extract mode complete", flush=True)
        return

    sequence_token_ids, structure_embed_table, extraction_metadata = _load_extraction_checkpoint(
        checkpoint_dir
    )
    structure_vocab_size = int(extraction_metadata["structure_vocab_size"])
    extraction_seconds = float(extraction_metadata.get("extraction_seconds", 0.0))
    structure_embed_vocab_size = int(structure_embed_table.shape[0])
    structure_embed_dim = int(structure_embed_table.shape[1])

    if sequence_token_ids.shape[1] < 2:
        raise ValueError("Structure token sequence is too short to contain BOS/EOS tokens.")

    if sequence_token_ids.max() >= structure_embed_vocab_size:
        raise ValueError(
            "Predicted token id exceeds structure embedding table size: "
            f"max_id={int(sequence_token_ids.max())}, table_rows={structure_embed_vocab_size}."
        )

    index = {sequence: i for i, sequence in enumerate(unique_sequences)}
    train_ids = sequence_token_ids[[index[s] for s in train_sequences]]
    mutant_ids = sequence_token_ids[[index[s] for s in mutants]]
    wt_ids = sequence_token_ids[index[wt_sequence]]

    train_bos_emb = structure_embed_table[train_ids[:, 0]]
    mutant_bos_emb = structure_embed_table[mutant_ids[:, 0]]
    wt_bos_emb = structure_embed_table[wt_ids[0]]

    train_eos_emb = structure_embed_table[train_ids[:, -1]]
    mutant_eos_emb = structure_embed_table[mutant_ids[:, -1]]
    wt_eos_emb = structure_embed_table[wt_ids[-1]]

    feature_sets: dict[str, tuple[np.ndarray | sparse.csr_matrix, np.ndarray | sparse.csr_matrix, np.ndarray | sparse.csr_matrix]] = {
        "structure_bos_embedding_ridge": (
            train_bos_emb,
            mutant_bos_emb,
            wt_bos_emb,
        ),
        "structure_eos_embedding_ridge": (
            train_eos_emb,
            mutant_eos_emb,
            wt_eos_emb,
        ),
        "structure_bos_eos_embedding_concat_ridge": (
            np.concatenate([train_bos_emb, train_eos_emb], axis=1),
            np.concatenate([mutant_bos_emb, mutant_eos_emb], axis=1),
            np.concatenate([wt_bos_emb, wt_eos_emb], axis=0),
        ),
    }

    t0 = time.perf_counter()
    train_onehot = _build_full_sequence_onehot_features(train_ids, vocab_size=structure_vocab_size)
    mutant_onehot = _build_full_sequence_onehot_features(mutant_ids, vocab_size=structure_vocab_size)
    wt_onehot = _build_full_sequence_onehot_features(wt_ids.reshape(1, -1), vocab_size=structure_vocab_size)
    build_onehot_seconds = time.perf_counter() - t0
    feature_sets["structure_full_onehot_concat_ridge"] = (train_onehot, mutant_onehot, wt_onehot)

    results: list[dict[str, Any]] = []
    for model_name, (train_features, mutant_features, wt_feature) in feature_sets.items():
        print(f"[structure-ablation-ridge] fitting and scoring {model_name}", flush=True)
        predicted_fitness, predicted_wt, fit_timings = _fit_and_score_ridge(
            train_features=train_features,
            train_labels=train_labels,
            mutant_features=mutant_features,
            wt_feature=wt_feature,
            alpha=args.ridge_alpha,
            fit_intercept=args.ridge_fit_intercept,
        )
        timings = {
            "structure_extraction_seconds": extraction_seconds,
            "build_onehot_seconds": build_onehot_seconds,
            **fit_timings,
        }
        results.append(
            _build_model_result(
                model_name=model_name,
                wt_sequence=wt_sequence,
                mutants=mutants,
                positions=positions,
                mutant_residues=mutant_residues,
                predicted_fitness=predicted_fitness,
                predicted_fitness_wt=predicted_wt,
                timings_seconds=timings,
                feature_shape=train_features.shape,
            )
        )

    compact_payload = {
        "run_started_utc": run_start,
        "run_completed_utc": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "esm_model_name": args.esm_model_name,
        "num_train_sequences": int(len(train_sequences)),
        "num_single_mutants": int(len(mutants)),
        "structure_token_length": int(sequence_token_ids.shape[1]),
        "structure_vocab_size": int(structure_vocab_size),
        "structure_embedding_table_shape": [
            int(structure_embed_vocab_size),
            int(structure_embed_dim),
        ],
        "structure_special_token_usage": {
            "bos_position": 0,
            "eos_position": int(sequence_token_ids.shape[1] - 1),
            "bos_token_ids_observed": sorted(
                list(dict.fromkeys(sequence_token_ids[:, 0].tolist()))
            ),
            "eos_token_ids_observed": sorted(
                list(dict.fromkeys(sequence_token_ids[:, -1].tolist()))
            ),
        },
        "models": [
            {
                "model_name": result["model_name"],
                "feature_shape": result["feature_shape"],
                "wt_predicted_fitness": result["wt_predicted_fitness"],
                "best_predicted_delta_energy": result["best_predicted_delta_energy"],
                "timings_seconds": result["timings_seconds"],
            }
            for result in results
        ],
    }
    full_payload = {
        "context": {
            "purpose": (
                "Compare four LGK surrogate-only regressors from one ESM3 pass per "
                "sequence: BOS structure embedding, EOS structure embedding, BOS+EOS "
                "concatenated structure embeddings, and full sequence concatenated "
                "one-hot structure token encoding."
            ),
            "energy_definition": "energy = -predicted_fitness",
            "delta_energy_definition": "predicted_delta_energy = -predicted_delta_fitness",
        },
        **compact_payload,
        "models": results,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(compact_payload, indent=2), encoding="utf-8")

    output_json_full = Path(args.output_json_full)
    output_json_full.parent.mkdir(parents=True, exist_ok=True)
    output_json_full.write_text(json.dumps(full_payload, indent=2), encoding="utf-8")
    print(f"[structure-ablation-ridge] wrote {output_json}", flush=True)
    print(f"[structure-ablation-ridge] wrote {output_json_full}", flush=True)


if __name__ == "__main__":
    main()
