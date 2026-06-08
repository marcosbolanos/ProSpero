from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch


class PersistentDatasetRepresentationStore:
    """Disk-backed store for fixed task dataset representations.

    Representations are saved as one contiguous `.npy` matrix plus JSON metadata,
    and are intended to be bulk-loaded into the in-memory persistent cache at the
    start of a run.
    """

    def __init__(self, cache_root: str | Path) -> None:
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def dataset_fingerprint(sequences: list[str]) -> str:
        digest = hashlib.sha256()
        for sequence in sequences:
            digest.update(sequence.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()[:16]

    def _safe(self, value: str) -> str:
        return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(value))

    def _namespace(
        self,
        *,
        task: str,
        model_name: str,
        max_length: int | None,
        representation_name: str,
        dataset_fingerprint: str,
    ) -> Path:
        namespace = (
            self.cache_root
            / self._safe(task)
            / self._safe(model_name)
            / self._safe(f"{representation_name}__maxlen_{max_length}")
            / self._safe(dataset_fingerprint)
        )
        namespace.mkdir(parents=True, exist_ok=True)
        return namespace

    def load_many(
        self,
        *,
        task: str,
        model_name: str,
        max_length: int | None,
        representation_name: str,
        dataset_fingerprint: str,
        sequences: list[str],
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        namespace = self._namespace(
            task=task,
            model_name=model_name,
            max_length=max_length,
            representation_name=representation_name,
            dataset_fingerprint=dataset_fingerprint,
        )
        metadata_path = namespace / "metadata.json"
        array_path = namespace / "representations.npy"
        if not metadata_path.exists() or not array_path.exists():
            return {}, list(sequences)

        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        sequence_to_row = {
            sequence: int(row_index)
            for row_index, sequence in enumerate(metadata["sequences"])
        }
        matrix = np.load(array_path, mmap_mode="r")

        loaded: dict[str, torch.Tensor] = {}
        missing: list[str] = []
        for sequence in sequences:
            row_index = sequence_to_row.get(sequence)
            if row_index is None:
                missing.append(sequence)
                continue
            loaded[sequence] = torch.from_numpy(
                np.asarray(matrix[row_index], dtype=np.float32)
            )
        return loaded, missing

    def exists(
        self,
        *,
        task: str,
        model_name: str,
        max_length: int | None,
        representation_name: str,
        dataset_fingerprint: str,
    ) -> bool:
        namespace = self._namespace(
            task=task,
            model_name=model_name,
            max_length=max_length,
            representation_name=representation_name,
            dataset_fingerprint=dataset_fingerprint,
        )
        return (namespace / "metadata.json").exists() and (
            namespace / "representations.npy"
        ).exists()

    def write_complete(
        self,
        *,
        task: str,
        model_name: str,
        max_length: int | None,
        representation_name: str,
        dataset_fingerprint: str,
        ordered_sequences: list[str],
        embeddings: list[torch.Tensor],
        storage_dtype: str = "float16",
    ) -> None:
        namespace = self._namespace(
            task=task,
            model_name=model_name,
            max_length=max_length,
            representation_name=representation_name,
            dataset_fingerprint=dataset_fingerprint,
        )
        metadata_path = namespace / "metadata.json"
        array_path = namespace / "representations.npy"
        if metadata_path.exists() and array_path.exists():
            return

        np_dtype = np.float16 if storage_dtype == "float16" else np.float32
        array = np.stack(
            [
                embedding.detach()
                .cpu()
                .numpy()
                .astype(np_dtype, copy=False)
                for embedding in embeddings
            ],
            axis=0,
        )
        metadata = {
            "task": task,
            "model_name": model_name,
            "max_length": max_length,
            "representation_name": representation_name,
            "dataset_fingerprint": dataset_fingerprint,
            "storage_dtype": str(np_dtype),
            "sequences": list(ordered_sequences),
            "shape": list(array.shape),
        }

        fd, tmp_array_str = tempfile.mkstemp(
            dir=namespace,
            prefix="representations.",
            suffix=".npy.tmp",
        )
        os.close(fd)
        tmp_array_path = Path(tmp_array_str)
        tmp_metadata_path = namespace / "metadata.json.tmp"
        try:
            with open(tmp_metadata_path, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle)
            with open(tmp_array_path, "wb") as handle:
                np.save(handle, array, allow_pickle=False)
            os.replace(tmp_metadata_path, metadata_path)
            os.replace(tmp_array_path, array_path)
        finally:
            tmp_metadata_path.unlink(missing_ok=True)
            tmp_array_path.unlink(missing_ok=True)
