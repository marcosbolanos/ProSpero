from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer  # type: ignore[reportMissingImports]

from prospero.dataset import RegressionDataset
from prospero.esm.cache import ESMEmbeddingFileCache
from prospero.surrogate import extract_residue_embeddings, normalize_sequences


DEFAULT_REPRESENTATION_NAME = "per_residue_embeddings_v1"
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AA_ALPHABET)}


@dataclass(frozen=True)
class TaskEmbeddings:
    task: str
    sequence_length: int
    train_sequences: list[str]
    valid_sequences: list[str]
    train_scores: np.ndarray
    valid_scores: np.ndarray
    train_residue_embeddings: torch.Tensor
    valid_residue_embeddings: torch.Tensor

    @property
    def n_train(self) -> int:
        return len(self.train_sequences)

    @property
    def n_valid(self) -> int:
        return len(self.valid_sequences)


class CachedTaskEmbeddingsLoader:
    def __init__(
        self,
        model_name: str,
        max_length: int | None,
        cache_root: str | Path | None = None,
        representation_name: str = DEFAULT_REPRESENTATION_NAME,
        require_cache: bool = True,
        compute_missing_embeddings: bool = False,
        embedding_batch_size: int = 64,
    ) -> None:
        self.cache = ESMEmbeddingFileCache(
            model_name=model_name,
            max_length=max_length,
            representation_name=representation_name,
            cache_root=cache_root,
        )
        self.require_cache = require_cache
        self.compute_missing_embeddings = compute_missing_embeddings
        self.embedding_batch_size = embedding_batch_size
        self.model_name = model_name
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = None
        self._esm = None

    def load_task(self, task: str) -> TaskEmbeddings:
        dataset = RegressionDataset(task)
        train_sequences = normalize_sequences(dataset.train.tolist())
        valid_sequences = normalize_sequences(dataset.valid.tolist())
        sequence_length = len(train_sequences[0])

        train_embeddings = self._load_sequences(train_sequences, task, "train")
        valid_embeddings = self._load_sequences(valid_sequences, task, "valid")
        self._validate_shapes(task, train_embeddings, valid_embeddings, sequence_length)

        return TaskEmbeddings(
            task=task,
            sequence_length=sequence_length,
            train_sequences=train_sequences,
            valid_sequences=valid_sequences,
            train_scores=np.asarray(dataset.train_scores, dtype=np.float64),
            valid_scores=np.asarray(dataset.valid_scores, dtype=np.float64),
            train_residue_embeddings=train_embeddings,
            valid_residue_embeddings=valid_embeddings,
        )

    def _load_sequences(
        self,
        sequences: list[str],
        task: str,
        split_name: str,
    ) -> torch.Tensor:
        embeddings_by_sequence, missing_sequences = self.cache.get_many(sequences)
        if missing_sequences and self.require_cache and not self.compute_missing_embeddings:
            raise FileNotFoundError(
                f"Missing {len(missing_sequences)} cached embeddings for task={task} "
                f"split={split_name}. Example sequence: {missing_sequences[0]!r}"
            )

        if missing_sequences:
            if not self.compute_missing_embeddings:
                raise FileNotFoundError(
                    f"Cache-only benchmark cannot proceed with missing embeddings for "
                    f"task={task} split={split_name}."
                )
            computed = self._compute_residue_embeddings(missing_sequences)
            for sequence, embedding in zip(missing_sequences, computed):
                embeddings_by_sequence[sequence] = embedding

        ordered = [embeddings_by_sequence[sequence].float() for sequence in sequences]
        return torch.stack(ordered, dim=0)

    def _validate_shapes(
        self,
        task: str,
        train_embeddings: torch.Tensor,
        valid_embeddings: torch.Tensor,
        sequence_length: int,
    ) -> None:
        if train_embeddings.ndim != 3 or valid_embeddings.ndim != 3:
            raise ValueError(
                f"Expected per-residue cached embeddings for task={task}, got "
                f"train shape {tuple(train_embeddings.shape)} and valid shape "
                f"{tuple(valid_embeddings.shape)}."
            )
        if train_embeddings.shape[1] != sequence_length:
            raise ValueError(
                f"Train embedding length mismatch for task={task}: expected "
                f"{sequence_length}, got {train_embeddings.shape[1]}."
            )
        if valid_embeddings.shape[1] != sequence_length:
            raise ValueError(
                f"Valid embedding length mismatch for task={task}: expected "
                f"{sequence_length}, got {valid_embeddings.shape[1]}."
            )

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def _get_esm(self):
        if self._esm is None:
            self._esm = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._esm.eval()
            for param in self._esm.parameters():
                param.requires_grad = False
        return self._esm

    def _tokenize_batch(self, sequences: list[str]):
        tokenizer = self._get_tokenizer()
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": self.max_length is not None,
        }
        if self.max_length is not None:
            tokenizer_kwargs["max_length"] = self.max_length
        encoded = tokenizer(sequences, **tokenizer_kwargs)
        return encoded["input_ids"], encoded["attention_mask"]

    def _compute_residue_embeddings(self, sequences: list[str]) -> torch.Tensor:
        esm = self._get_esm()
        batches = []
        for start in range(0, len(sequences), self.embedding_batch_size):
            batch_sequences = sequences[start : start + self.embedding_batch_size]
            input_ids, attention_mask = self._tokenize_batch(batch_sequences)
            attention_mask = attention_mask.to(self.device)
            with torch.no_grad():
                hidden = esm(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask,
                ).last_hidden_state
                residue_embeddings = extract_residue_embeddings(hidden, attention_mask)
            batches.append(residue_embeddings.cpu())
        return torch.cat(batches, dim=0)


def sequences_to_one_hot_flat(sequences: list[str]) -> np.ndarray:
    if not sequences:
        return np.zeros((0, 0), dtype=np.float32)

    sequence_length = len(sequences[0])
    features = np.zeros((len(sequences), sequence_length * len(AA_ALPHABET)), dtype=np.float32)
    for row_idx, sequence in enumerate(sequences):
        if len(sequence) != sequence_length:
            raise ValueError("All sequences must have the same length for one-hot features.")
        for position, residue in enumerate(sequence):
            aa_idx = AA_TO_INDEX.get(residue)
            if aa_idx is None:
                raise ValueError(f"Unknown residue {residue!r} encountered in sequence features.")
            features[row_idx, position * len(AA_ALPHABET) + aa_idx] = 1.0
    return features


def build_feature_matrix(
    feature_name: str,
    residue_embeddings: torch.Tensor,
    sequences: list[str] | None = None,
) -> np.ndarray:
    embeddings = residue_embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
    if feature_name in {
        "one_hot_flat",
        "flatten_plus_one_hot",
        "mean_pool_plus_one_hot",
        "mean_pool_l2_plus_one_hot",
        "mean_pool_zscore_plus_one_hot",
    } and sequences is None:
        raise ValueError(f"Feature {feature_name!r} requires sequences.")

    if feature_name == "one_hot_flat":
        return sequences_to_one_hot_flat(sequences or [])

    if feature_name == "flatten_plus_one_hot":
        flatten = embeddings.reshape(embeddings.shape[0], -1)
        one_hot = sequences_to_one_hot_flat(sequences or [])
        return np.concatenate([flatten, one_hot], axis=1)

    if feature_name == "mean_pool":
        return embeddings.mean(axis=1)
    if feature_name == "max_pool":
        return embeddings.max(axis=1)
    if feature_name == "mean_max":
        return np.concatenate([embeddings.mean(axis=1), embeddings.max(axis=1)], axis=1)
    if feature_name == "mean_pool_plus_one_hot":
        mean_pool = embeddings.mean(axis=1)
        one_hot = sequences_to_one_hot_flat(sequences or [])
        return np.concatenate([mean_pool, one_hot], axis=1)
    if feature_name == "mean_pool_l2_plus_one_hot":
        mean_pool = embeddings.mean(axis=1)
        norms = np.linalg.norm(mean_pool, axis=1, keepdims=True)
        l2_mean_pool = mean_pool / np.clip(norms, a_min=1e-8, a_max=None)
        one_hot = sequences_to_one_hot_flat(sequences or [])
        return np.concatenate([l2_mean_pool, one_hot], axis=1)
    if feature_name == "mean_pool_zscore_plus_one_hot":
        mean_pool = embeddings.mean(axis=1)
        row_means = mean_pool.mean(axis=1, keepdims=True)
        row_stds = mean_pool.std(axis=1, keepdims=True)
        z_mean_pool = (mean_pool - row_means) / np.clip(row_stds, a_min=1e-8, a_max=None)
        one_hot = sequences_to_one_hot_flat(sequences or [])
        return np.concatenate([z_mean_pool, one_hot], axis=1)
    if feature_name == "flatten":
        return embeddings.reshape(embeddings.shape[0], -1)
    raise ValueError(f"Unsupported feature_name={feature_name!r}")
