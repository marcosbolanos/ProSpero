import hashlib
import os
import tempfile
import time
import threading
from collections import OrderedDict
from pathlib import Path

import torch


class ESMEmbeddingFileCache:
    def __init__(
        self,
        model_name,
        max_length,
        representation_name,
        cache_root=None,
        max_disk_bytes=None,
        max_memory_bytes=None,
        storage_dtype="float16",
        eviction_check_interval=128,
    ):
        if cache_root is None:
            cache_root = Path(__file__).resolve().parents[2] / ".cache" / "esm_embeddings"

        self.cache_root = Path(cache_root)
        self.model_name = model_name
        self.max_length = max_length
        self.representation_name = representation_name
        self.max_disk_bytes = int(
            max_disk_bytes
            if max_disk_bytes is not None
            else float(os.environ.get("PROSPERO_ESM_CACHE_MAX_GB", "500")) * (1024**3)
        )
        self.max_memory_bytes = int(
            max_memory_bytes
            if max_memory_bytes is not None
            else float(os.environ.get("PROSPERO_ESM_CACHE_MEMORY_MB", "1024")) * (1024**2)
        )
        self.storage_dtype = storage_dtype
        self.eviction_check_interval = max(1, int(eviction_check_interval))
        self._writes_since_eviction_check = 0
        self._memory_lru = OrderedDict()
        self._memory_bytes = 0
        self._memory_lock = threading.Lock()

        self.namespace = self.cache_root / self._safe(model_name) / self._safe(
            f"{representation_name}__maxlen_{max_length}"
        )
        self.namespace.mkdir(parents=True, exist_ok=True)

    def _safe(self, s: str) -> str:
        return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(s))

    def _cache_key(self, sequence: str) -> str:
        cache_input = (
            f"{self.model_name}|{self.max_length}|{self.representation_name}|{sequence}"
        )
        return hashlib.sha256(cache_input.encode("utf-8")).hexdigest()

    def _path_for_sequence(self, sequence: str) -> Path:
        key = self._cache_key(sequence)
        subdir = self.namespace / key[:2]
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{key}.pt"

    def _estimate_tensor_bytes(self, tensor: torch.Tensor) -> int:
        return int(tensor.numel() * tensor.element_size())

    def _add_to_memory_cache(self, sequence: str, embedding: torch.Tensor) -> None:
        if self.max_memory_bytes <= 0:
            return

        tensor = embedding.detach().cpu()
        tensor_bytes = self._estimate_tensor_bytes(tensor)
        if tensor_bytes > self.max_memory_bytes:
            return

        with self._memory_lock:
            existing = self._memory_lru.pop(sequence, None)
            if existing is not None:
                self._memory_bytes -= self._estimate_tensor_bytes(existing)

            self._memory_lru[sequence] = tensor
            self._memory_bytes += tensor_bytes

            while self._memory_bytes > self.max_memory_bytes and self._memory_lru:
                _, evicted = self._memory_lru.popitem(last=False)
                self._memory_bytes -= self._estimate_tensor_bytes(evicted)

    def _get_from_memory_cache(self, sequence: str):
        with self._memory_lock:
            cached = self._memory_lru.pop(sequence, None)
            if cached is None:
                return None
            self._memory_lru[sequence] = cached
            return cached

    def _serialize_embedding(self, embedding: torch.Tensor):
        tensor = embedding.detach().cpu().contiguous()
        if not torch.is_floating_point(tensor):
            return {
                "format": "tensor_v2",
                "encoding": "raw",
                "tensor": tensor,
            }

        if self.storage_dtype == "float32":
            return {
                "format": "tensor_v2",
                "encoding": "raw",
                "tensor": tensor.to(torch.float32),
            }

        if self.storage_dtype == "int8":
            max_abs = float(tensor.abs().max().item())
            scale = max_abs / 127.0 if max_abs > 0 else 1.0
            quantized = torch.clamp((tensor / scale).round(), -127, 127).to(torch.int8)
            return {
                "format": "tensor_v2",
                "encoding": "int8_sym",
                "scale": scale,
                "shape": tuple(tensor.shape),
                "tensor": quantized,
            }

        return {
            "format": "tensor_v2",
            "encoding": "float16",
            "tensor": tensor.to(torch.float16),
        }

    def _deserialize_embedding(self, payload):
        if not isinstance(payload, dict) or payload.get("format") != "tensor_v2":
            if isinstance(payload, torch.Tensor):
                return payload
            raise ValueError("Unsupported cache payload format.")

        encoding = payload["encoding"]
        if encoding == "raw":
            return payload["tensor"]
        if encoding == "float16":
            return payload["tensor"].to(torch.float32)
        if encoding == "int8_sym":
            quantized = payload["tensor"].to(torch.float32)
            return quantized.mul(float(payload["scale"])).view(tuple(payload["shape"]))
        raise ValueError(f"Unknown cache payload encoding: {encoding}")

    def _disk_usage_bytes(self) -> int:
        total = 0
        for path in self.namespace.rglob("*.pt"):
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                continue
        return total

    def _evict_disk_if_needed(self) -> None:
        if self.max_disk_bytes <= 0:
            return
        if self._disk_usage_bytes() <= self.max_disk_bytes:
            return

        lock_path = self.namespace / ".evict.lock"
        try:
            lock_path.mkdir()
        except FileExistsError:
            return

        try:
            entries = []
            total = 0
            for path in self.namespace.rglob("*.pt"):
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                total += stat.st_size
                entries.append((stat.st_mtime, stat.st_size, path))
            if total <= self.max_disk_bytes:
                return

            entries.sort(key=lambda item: item[0])
            for _, size, path in entries:
                if total <= self.max_disk_bytes:
                    break
                try:
                    path.unlink()
                    total -= size
                except FileNotFoundError:
                    continue
                except PermissionError:
                    continue
        finally:
            try:
                lock_path.rmdir()
            except OSError:
                pass

    def get_many(self, sequences):
        embeddings_by_sequence = {}
        missing_sequences = []

        for sequence in sequences:
            memory_hit = self._get_from_memory_cache(sequence)
            if memory_hit is not None:
                embeddings_by_sequence[sequence] = memory_hit
                continue

            path = self._path_for_sequence(sequence)
            if path.exists():
                try:
                    loaded = torch.load(path, map_location="cpu")
                    embedding = self._deserialize_embedding(loaded)
                    embeddings_by_sequence[sequence] = embedding
                    self._add_to_memory_cache(sequence, embedding)
                    try:
                        now = time.time()
                        os.utime(path, (now, now))
                    except OSError:
                        pass
                except Exception:
                    # Corrupt cache entry: delete and recompute
                    path.unlink(missing_ok=True)
                    missing_sequences.append(sequence)
            else:
                missing_sequences.append(sequence)

        return embeddings_by_sequence, missing_sequences

    def set_many(self, sequences, embeddings):
        for sequence, embedding in zip(sequences, embeddings):
            path = self._path_for_sequence(sequence)

            self._add_to_memory_cache(sequence, embedding)
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)

            fd, tmp_path_str = tempfile.mkstemp(
                dir=path.parent,
                prefix=path.name + ".",
                suffix=".tmp",
            )
            os.close(fd)
            tmp_path = Path(tmp_path_str)

            try:
                payload = self._serialize_embedding(embedding)
                torch.save(payload, tmp_path)
                os.replace(tmp_path, path)  # atomic on same filesystem
            finally:
                tmp_path.unlink(missing_ok=True)

            self._writes_since_eviction_check += 1
            if self._writes_since_eviction_check >= self.eviction_check_interval:
                self._writes_since_eviction_check = 0
                self._evict_disk_if_needed()
