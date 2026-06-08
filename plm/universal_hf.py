from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class UniversalHFComponents:
    tokenizer: Any
    model: Any
    model_id: str
    model_type: str
    hidden_size: int
    special_token_ids: tuple[int, ...]
    local_files_only_used: bool
    model_loader_class: str


def infer_hidden_size(config: Any) -> int:
    for attr in ("hidden_size", "d_model", "n_embd", "embed_dim"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError(
        "Could not infer hidden size from config; expected one of "
        "hidden_size/d_model/n_embd/embed_dim."
    )


def _safe_tokenize(tokenizer: Any, inputs: Any, **kwargs: Any) -> dict[str, Any] | None:
    try:
        return tokenizer(inputs, **kwargs)
    except Exception:
        return None


def _residue_count(input_ids: torch.Tensor, attention_mask: torch.Tensor, special_ids: set[int]) -> list[int]:
    counts: list[int] = []
    for token_row, mask_row in zip(input_ids, attention_mask):
        valid_positions = torch.nonzero(mask_row.bool(), as_tuple=False).squeeze(-1)
        valid_token_ids = token_row[valid_positions]
        counts.append(sum(int(tok.item()) not in special_ids for tok in valid_token_ids))
    return counts


def _to_spaced_sequence(sequence: str) -> str:
    return " ".join(list(sequence))


def _detect_tokenization_mode(tokenizer: Any, sequences: list[str]) -> str:
    sample = sequences[: min(8, len(sequences))]
    if not sample:
        return "raw"

    special_ids = set(int(x) for x in (tokenizer.all_special_ids or []))
    modes = {
        "raw": sample,
        "spaced": [_to_spaced_sequence(seq) for seq in sample],
        "split": [list(seq) for seq in sample],
    }

    best_mode = "raw"
    best_score = -1
    for mode, mode_inputs in modes.items():
        kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": False,
            "add_special_tokens": True,
        }
        if mode == "split":
            kwargs["is_split_into_words"] = True
        encoded = _safe_tokenize(tokenizer, mode_inputs, **kwargs)
        if encoded is None or "input_ids" not in encoded or "attention_mask" not in encoded:
            continue
        counts = _residue_count(encoded["input_ids"], encoded["attention_mask"], special_ids)
        score = sum(int(count == len(seq)) for count, seq in zip(counts, sample))
        if score > best_score:
            best_score = score
            best_mode = mode

    return best_mode


def tokenize_protein_sequences(
    tokenizer: Any,
    sequences: list[str],
    *,
    max_length: int | None = None,
    mode: str = "auto",
) -> dict[str, torch.Tensor]:
    if mode == "auto":
        mode = _detect_tokenization_mode(tokenizer, sequences)

    tokenizer_kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": max_length is not None,
    }
    if max_length is not None:
        tokenizer_kwargs["max_length"] = max_length

    if mode == "spaced":
        encoded = tokenizer([_to_spaced_sequence(seq) for seq in sequences], **tokenizer_kwargs)
    elif mode == "split":
        tokenizer_kwargs["is_split_into_words"] = True
        encoded = tokenizer([list(seq) for seq in sequences], **tokenizer_kwargs)
    elif mode == "raw":
        encoded = tokenizer(sequences, **tokenizer_kwargs)
    else:
        raise ValueError(f"Unsupported tokenization mode: {mode}")

    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "tokenization_mode": mode,
    }


def load_universal_hf_components(
    model_id: str,
    *,
    device: str | None = None,
    trust_remote_code: bool = True,
) -> UniversalHFComponents:
    from transformers import AutoModel  # type: ignore[reportMissingImports]
    from transformers import AutoModelForCausalLM  # type: ignore[reportMissingImports]
    from transformers import AutoModelForMaskedLM  # type: ignore[reportMissingImports]
    from transformers import AutoTokenizer  # type: ignore[reportMissingImports]
    from transformers import BertTokenizer  # type: ignore[reportMissingImports]
    from transformers import EsmTokenizer  # type: ignore[reportMissingImports]
    from transformers import T5Tokenizer  # type: ignore[reportMissingImports]

    def _load_tokenizer(local_files_only: bool):
        for use_fast in (True, False):
            try:
                return AutoTokenizer.from_pretrained(
                    model_id,
                    local_files_only=local_files_only,
                    trust_remote_code=trust_remote_code,
                    use_fast=use_fast,
                )
            except Exception:
                continue
        for tokenizer_class in (EsmTokenizer, BertTokenizer, T5Tokenizer):
            try:
                return tokenizer_class.from_pretrained(
                    model_id,
                    local_files_only=local_files_only,
                    trust_remote_code=trust_remote_code,
                )
            except Exception:
                continue
        raise ValueError(f"Failed to load tokenizer for {model_id}")

    def _load_model(local_files_only: bool):
        loader_classes = [AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM]
        for loader_class in loader_classes:
            for use_safetensors in (True, None):
                try:
                    kwargs = {
                        "local_files_only": local_files_only,
                        "trust_remote_code": trust_remote_code,
                    }
                    if use_safetensors is not None:
                        kwargs["use_safetensors"] = use_safetensors
                    model = loader_class.from_pretrained(model_id, **kwargs).to(runtime_device)
                    return model, loader_class.__name__
                except Exception:
                    continue
        raise ValueError(f"Failed to load model for {model_id}")

    runtime_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    local_files_only_used = True
    try:
        tokenizer = _load_tokenizer(local_files_only=True)
        model, model_loader_class = _load_model(local_files_only=True)
    except Exception:
        tokenizer = _load_tokenizer(local_files_only=False)
        model, model_loader_class = _load_model(local_files_only=False)
        local_files_only_used = False

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    if getattr(tokenizer, "pad_token", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        unk_token = getattr(tokenizer, "unk_token", None)
        if eos_token is not None:
            tokenizer.pad_token = eos_token
        elif unk_token is not None:
            tokenizer.pad_token = unk_token

    special_ids = tuple(int(x) for x in (tokenizer.all_special_ids or []))
    hidden_size = infer_hidden_size(model.config)
    model_type = str(getattr(model.config, "model_type", "unknown"))

    return UniversalHFComponents(
        tokenizer=tokenizer,
        model=model,
        model_id=model_id,
        model_type=model_type,
        hidden_size=hidden_size,
        special_token_ids=special_ids,
        local_files_only_used=local_files_only_used,
        model_loader_class=model_loader_class,
    )
