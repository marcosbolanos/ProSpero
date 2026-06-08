from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Small local LoRA wrapper to avoid adding a PEFT dependency."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        self.lora_a.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_b.to(device=base.weight.device, dtype=base.weight.dtype)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.lora_b(self.lora_a(self.dropout(inputs))) * self.scaling


def truncate_esm_encoder(model, *, hidden_state_layer: int):
    """Keep only enough encoder blocks to produce hidden_states[hidden_state_layer]."""

    if hidden_state_layer < 1:
        raise ValueError("hidden_state_layer must be >= 1 for encoder-layer CLS")
    n_blocks = int(hidden_state_layer)
    current_layers = model.encoder.layer
    if n_blocks > len(current_layers):
        raise ValueError(
            f"Requested hidden_state_layer={hidden_state_layer}, but model has "
            f"{len(current_layers)} encoder blocks"
        )
    model.encoder.layer = nn.ModuleList(list(current_layers[:n_blocks]))
    model.config.num_hidden_layers = n_blocks
    return model


def _resolve_parent_module(root: nn.Module, dotted_parent: str) -> nn.Module:
    module = root
    if not dotted_parent:
        return module
    for part in dotted_parent.split("."):
        module = getattr(module, part)
    return module


def apply_lora_to_esm_layer(
    model,
    *,
    encoder_layer_index: int,
    target_modules: tuple[str, ...] = ("attention.self.query", "attention.self.value"),
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
) -> list[str]:
    layer = model.encoder.layer[int(encoder_layer_index)]
    replaced: list[str] = []
    for target in target_modules:
        parent_name, child_name = target.rsplit(".", 1)
        parent = _resolve_parent_module(layer, parent_name)
        child = getattr(parent, child_name)
        if isinstance(child, LoRALinear):
            replaced.append(f"encoder.layer.{encoder_layer_index}.{target}")
            continue
        if not isinstance(child, nn.Linear):
            raise TypeError(f"LoRA target {target!r} is {type(child).__name__}, not Linear")
        setattr(
            parent,
            child_name,
            LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout),
        )
        replaced.append(f"encoder.layer.{encoder_layer_index}.{target}")
    return replaced


def lora_state_dict(model) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if ".lora_a." in key or ".lora_b." in key
    }


def save_lora_adapter(
    *,
    output_dir: Path,
    model,
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), output_dir / "adapter.pt")
    (output_dir / "adapter_config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_lora_adapter(
    *,
    adapter_dir: Path,
    model,
    device: str,
) -> dict[str, object]:
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    target_modules = tuple(config.get("target_modules", ("attention.self.query", "attention.self.value")))
    apply_lora_to_esm_layer(
        model,
        encoder_layer_index=int(config["lora_encoder_layer_index"]),
        target_modules=target_modules,
        rank=int(config["lora_rank"]),
        alpha=float(config["lora_alpha"]),
        dropout=float(config.get("lora_dropout", 0.0)),
    )
    state = torch.load(adapter_dir / "adapter.pt", map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected_lora = [key for key in unexpected if "lora_" in key]
    missing_lora = [key for key in missing if "lora_" in key]
    if unexpected_lora or missing_lora:
        raise RuntimeError(
            f"LoRA adapter mismatch: missing_lora={missing_lora}, "
            f"unexpected_lora={unexpected_lora}"
        )
    return config
