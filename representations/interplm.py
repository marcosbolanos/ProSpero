from __future__ import annotations

import json
from pathlib import Path
import threading

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download  # type: ignore[reportMissingImports]


DEFAULT_INTERPLM_REPO_ID = "Elana/InterPLM-esm2-8m"


class InterPLMSparseAutoencoder(nn.Module):
    """Minimal InterPLM sparse autoencoder used for feature extraction."""

    def __init__(self, input_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.feature_dim = int(feature_dim)
        self.register_parameter("bias", nn.Parameter(torch.zeros(self.input_dim)))
        self.encoder = nn.Linear(self.input_dim, self.feature_dim)
        self.decoder = nn.Linear(self.feature_dim, self.input_dim, bias=False)

    def encode(self, residue_embeddings: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(residue_embeddings - self.bias))

    def forward(self, residue_embeddings: torch.Tensor) -> torch.Tensor:
        activations = self.encode(residue_embeddings)
        return self.decoder(activations) + self.bias


def load_interplm_sae(
    plm_layer: int,
    repo_id: str = DEFAULT_INTERPLM_REPO_ID,
    normalized: bool = True,
    cache_dir: str | Path | None = None,
    map_location: str | torch.device = "cpu",
) -> InterPLMSparseAutoencoder:
    layer_dir = f"layer_{int(plm_layer)}"
    weights_filename = "ae_normalized.pt" if normalized else "ae_unnormalized.pt"
    download_kwargs = {
        "repo_id": repo_id,
        "cache_dir": None if cache_dir is None else str(cache_dir),
    }
    config_filename = f"{layer_dir}/config.json"
    weights_filename_path = f"{layer_dir}/{weights_filename}"
    # Prefer cached local files to avoid repeated remote metadata checks.
    try:
        config_path = hf_hub_download(
            filename=config_filename,
            local_files_only=True,
            **download_kwargs,
        )
    except Exception:
        config_path = hf_hub_download(
            filename=config_filename,
            local_files_only=False,
            **download_kwargs,
        )
    try:
        weights_path = hf_hub_download(
            filename=weights_filename_path,
            local_files_only=True,
            **download_kwargs,
        )
    except Exception:
        weights_path = hf_hub_download(
            filename=weights_filename_path,
            local_files_only=False,
            **download_kwargs,
        )

    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    architecture = config["architecture"]
    model = InterPLMSparseAutoencoder(
        input_dim=int(architecture["esm_dim"]),
        feature_dim=int(architecture["feature_dim"]),
    )
    state_dict = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(state_dict)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


class SharedInterPLMSAEPool:
    """Thread-safe singleton pool for InterPLM SAE models."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: dict[tuple[int, str, bool, str], InterPLMSparseAutoencoder] = {}

    def get(
        self,
        *,
        plm_layer: int,
        repo_id: str,
        normalized: bool,
        device: str | torch.device,
    ) -> InterPLMSparseAutoencoder:
        device_key = str(device)
        key = (int(plm_layer), str(repo_id), bool(normalized), device_key)
        with self._lock:
            cached = self._models.get(key)
            if cached is not None:
                return cached
            model = load_interplm_sae(
                plm_layer=plm_layer,
                repo_id=repo_id,
                normalized=normalized,
                map_location=device,
            ).to(device)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad = False
            self._models[key] = model
            return model


def build_interplm_representation_name(
    *,
    layer: int,
    repo_id: str,
    normalized: bool,
) -> str:
    safe_repo_id = repo_id.replace("/", "__")
    normalization_tag = "normalized" if normalized else "unnormalized"
    return (
        f"interplm_mean_pool_v1__layer_{int(layer)}"
        f"__repo_{safe_repo_id}__{normalization_tag}"
    )


def mean_pool_sae_activations(
    sae: InterPLMSparseAutoencoder,
    residue_embeddings: torch.Tensor,
    *,
    token_chunk_size: int,
) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = residue_embeddings.shape
    flattened = residue_embeddings.reshape(batch_size * sequence_length, hidden_dim)
    pooled = torch.zeros(
        (batch_size, sae.feature_dim),
        dtype=torch.float32,
        device=flattened.device,
    )

    chunk_size = max(1, int(token_chunk_size))
    for start in range(0, flattened.shape[0], chunk_size):
        stop = min(flattened.shape[0], start + chunk_size)
        activations = sae.encode(flattened[start:stop]).to(torch.float32)
        token_indices = torch.arange(start, stop, device=flattened.device)
        sequence_indices = token_indices // sequence_length
        pooled.index_add_(0, sequence_indices, activations)

    return pooled / float(sequence_length)
