from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


def _ensure_esm_pretrained_compat() -> None:
    # Some environments ship an `esm` build where `esm.pretrained` exposes the
    # local registry but not the helper used by `ESMC.from_pretrained()` and
    # `ESM3.from_pretrained()`. Patch it in before model construction.
    import esm.pretrained as pretrained

    if hasattr(pretrained, "load_local_model"):
        return

    registry = None
    for candidate in ("LOCAL_MODEL_REGISTRY", "_LOCAL_MODEL_REGISTRY", "MODEL_REGISTRY"):
        maybe = getattr(pretrained, candidate, None)
        if isinstance(maybe, dict):
            registry = maybe
            break

    # Some wheel variants don't expose the registry publicly. Build a small
    # fallback from known local constructors if available.
    if registry is None:
        registry = {}
        known_builders = {
            "esm3_sm_open_v1": "ESM3_sm_open_v0",
            "esmc_300m": "ESMC_300M_202412",
            "esmc_600m": "ESMC_600M_202412",
        }
        for model_name, attr in known_builders.items():
            builder = getattr(pretrained, attr, None)
            if callable(builder):
                registry[model_name] = builder

        if not registry:
            # Let upstream loading fail with its own error if we cannot
            # synthesize a compatible shim.
            return

    def load_local_model(
        model_name: str,
        device: torch.device = torch.device("cpu"),
    ):
        if model_name not in registry:
            raise ValueError(
                f"Model {model_name} not found in local model registry."
            )
        return registry[model_name](device)

    pretrained.load_local_model = load_local_model


def is_evolutionaryscale_model(model_id: str) -> bool:
    model_id = str(model_id)
    return model_id.startswith("EvolutionaryScale/esm3") or model_id.startswith(
        "EvolutionaryScale/esmc"
    )


def hf_to_esm_sdk_name(model_id: str) -> str:
    if model_id == "EvolutionaryScale/esm3-sm-open-v1":
        return "esm3_sm_open_v1"
    if model_id == "EvolutionaryScale/esmc-300m-2024-12":
        return "esmc_300m"
    if model_id == "EvolutionaryScale/esmc-600m-2024-12":
        return "esmc_600m"
    raise ValueError(
        f"Unsupported EvolutionaryScale model mapping for '{model_id}'. "
        "Supported currently: esm3-sm-open-v1, esmc-300m-2024-12, esmc-600m-2024-12."
    )


@dataclass(frozen=True)
class EvolutionaryScaleBackend:
    model_id: str
    sdk_name: str
    device: str
    model: object
    hidden_size: int

    @staticmethod
    def load(model_id: str, *, device: str) -> "EvolutionaryScaleBackend":
        try:
            _ensure_esm_pretrained_compat()
            from esm.models.esm3 import ESM3
            from esm.models.esmc import ESMC
            import esm.pretrained as esm_pretrained
            from esm.sdk.api import ESMProtein
            from esm.sdk.api import LogitsConfig
        except Exception as exc:
            raise ImportError(
                "EvolutionaryScale backend requires esm>=3.x. "
                "Run with `uv run --with \"esm==3.2.3\" ...` or add it to the environment."
            ) from exc

        sdk_name = hf_to_esm_sdk_name(model_id)
        torch_device = torch.device(device)
        try:
            if sdk_name.startswith("esmc_"):
                model = ESMC.from_pretrained(sdk_name, device=torch_device)
            else:
                model = ESM3.from_pretrained(sdk_name, device=torch_device)
        except ImportError as exc:
            # Some `esm` builds miss `esm.pretrained.load_local_model`, which
            # breaks `from_pretrained`. Fallback to direct builder functions.
            if "load_local_model" not in str(exc):
                raise
            direct_builders = {
                "esm3_sm_open_v1": "ESM3_sm_open_v0",
                "esmc_300m": "ESMC_300M_202412",
                "esmc_600m": "ESMC_600M_202412",
            }
            builder_name = direct_builders.get(sdk_name)
            builder = getattr(esm_pretrained, builder_name, None) if builder_name else None
            if not callable(builder):
                raise
            model = builder(device=torch_device)

        # Infer hidden size once.
        probe = ESMProtein(sequence="ACDEFGHIKLMNPQRSTVWY")
        probe_tensor = model.encode(probe)
        probe_out = model.logits(
            probe_tensor,
            LogitsConfig(sequence=True, return_embeddings=True),
        )
        probe_embeddings = probe_out.embeddings
        hidden_size = int(probe_embeddings.shape[-1])

        return EvolutionaryScaleBackend(
            model_id=model_id,
            sdk_name=sdk_name,
            device=device,
            model=model,
            hidden_size=hidden_size,
        )

    def _residue_embeddings_for_sequence(self, sequence: str) -> torch.Tensor:
        from esm.sdk.api import ESMProtein
        from esm.sdk.api import LogitsConfig

        protein = ESMProtein(sequence=sequence)
        protein_tensor = self.model.encode(protein)
        output = self.model.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True),
        )
        embeddings = output.embeddings
        if embeddings.dim() != 3 or embeddings.shape[0] != 1:
            raise ValueError(
                f"Unexpected embedding tensor shape from EvolutionaryScale backend: {tuple(embeddings.shape)}"
            )
        residue_embeddings = embeddings[0]

        seq_len = len(sequence)
        if residue_embeddings.shape[0] == seq_len + 2:
            residue_embeddings = residue_embeddings[1:-1]
        elif residue_embeddings.shape[0] == seq_len:
            pass
        else:
            raise ValueError(
                f"Unexpected token length {residue_embeddings.shape[0]} for sequence length {seq_len}."
            )
        # Keep backend outputs on CPU so they are compatible with the shared
        # representation cache/store pipeline.
        return residue_embeddings.to(torch.float32).cpu()

    @torch.no_grad()
    def compute_representations(
        self,
        sequences: Sequence[str],
        *,
        representation_name: str,
        expected_sequence_length: int | None = None,
    ) -> torch.Tensor:
        if not sequences:
            if representation_name == "mean_pool_residue_embeddings_v1":
                return torch.empty((0, self.hidden_size), dtype=torch.float32)
            seq_len = int(expected_sequence_length or 0)
            return torch.empty((0, seq_len, self.hidden_size), dtype=torch.float32)

        residue_batch = [self._residue_embeddings_for_sequence(sequence) for sequence in sequences]

        if expected_sequence_length is not None:
            for emb in residue_batch:
                if emb.shape[0] != expected_sequence_length:
                    raise ValueError(
                        "Per-residue embeddings length mismatch. "
                        f"Expected {expected_sequence_length}, got {emb.shape[0]}."
                    )

        if representation_name == "mean_pool_residue_embeddings_v1":
            return torch.stack([emb.mean(dim=0) for emb in residue_batch], dim=0)
        if representation_name == "per_residue_embeddings_v1":
            reference = residue_batch[0].shape[0]
            if any(emb.shape[0] != reference for emb in residue_batch):
                raise ValueError(
                    "Per-residue backend expects fixed sequence length within each batch."
                )
            return torch.stack(residue_batch, dim=0)
        raise ValueError(f"Unsupported representation_name={representation_name}")
