from __future__ import annotations

import argparse
import inspect
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)
from transformers import BertTokenizer, EsmTokenizer, T5Tokenizer

from prospero.plm.universal_hf import infer_hidden_size
from prospero.plm.universal_hf import tokenize_protein_sequences


DEFAULT_SEQUENCE = "MKTIIALSYIFCLVFADYKDDDDA"
_LARGE_MODEL_PATTERN = re.compile(r"(?:^|[_-])(3B|7B|8B|15B|34B|70B)(?:$|[_-])", re.IGNORECASE)


def _device_for_probe(force_cpu: bool) -> str:
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _num_params(model: Any) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def _format_gb(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024.0**3)


def _forward(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    is_encoder_decoder = bool(getattr(getattr(model, "config", None), "is_encoder_decoder", False))
    target = model
    if is_encoder_decoder:
        encoder = getattr(model, "encoder", None)
        if encoder is None and hasattr(model, "get_encoder"):
            encoder = model.get_encoder()
        if encoder is not None:
            target = encoder

    params = set(inspect.signature(target.forward).parameters)
    kwargs: dict[str, Any] = {"input_ids": input_ids}
    if "attention_mask" in params:
        kwargs["attention_mask"] = attention_mask
    if "output_hidden_states" in params:
        kwargs["output_hidden_states"] = True
    if "return_dict" in params:
        kwargs["return_dict"] = True
    return target(**kwargs)


def _extract_main_output_shape(outputs: Any) -> list[int] | None:
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        tensor = outputs.last_hidden_state
        return [int(x) for x in tensor.shape]

    if isinstance(outputs, tuple) and outputs:
        first = outputs[0]
        if torch.is_tensor(first):
            return [int(x) for x in first.shape]

    if hasattr(outputs, "keys"):
        for key in outputs.keys():
            value = outputs[key]
            if torch.is_tensor(value) and value.dim() >= 2:
                return [int(x) for x in value.shape]
    return None


def _load_tokenizer(model_id: str, trust_remote_code: bool):
    errors: list[str] = []
    for use_fast in (True, False):
        try:
            tok = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                use_fast=use_fast,
            )
            return tok
        except Exception as exc:
            errors.append(f"AutoTokenizer(use_fast={use_fast}): {exc}")
            continue
    for tokenizer_class in (EsmTokenizer, BertTokenizer, T5Tokenizer):
        try:
            return tokenizer_class.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
            )
        except Exception as exc:
            errors.append(f"{tokenizer_class.__name__}: {exc}")
            continue
    raise ValueError("Tokenizer load failed. " + " | ".join(errors[:4]))


def _token_alignment_stats(tokenizer: Any, sequence: str) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for mode in ("raw", "spaced", "split"):
        try:
            encoded = tokenize_protein_sequences(tokenizer, [sequence], mode=mode)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            special_ids = set(int(x) for x in (tokenizer.all_special_ids or []))
            valid_positions = torch.nonzero(attention_mask[0].bool(), as_tuple=False).squeeze(-1)
            tokens = input_ids[0, valid_positions]
            residue_count = sum(int(tok.item()) not in special_ids for tok in tokens)
            stats[mode] = {
                "ok": True,
                "residue_token_count": int(residue_count),
                "matches_sequence_length": bool(residue_count == len(sequence)),
                "tokenized_length": int(attention_mask[0].sum().item()),
            }
        except Exception as exc:
            stats[mode] = {"ok": False, "error": str(exc)}
    return stats


def _resolve_load_mode(requested_mode: str, model_id: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    return "meta" if _LARGE_MODEL_PATTERN.search(model_id) else "full"


def probe_model(
    model_id: str,
    device: str,
    trust_remote_code: bool,
    sequence: str,
    load_mode: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    effective_load_mode = _resolve_load_mode(load_mode, model_id)
    report: dict[str, Any] = {
        "model_id": model_id,
        "device": device,
        "trust_remote_code": bool(trust_remote_code),
        "load_mode_requested": load_mode,
        "load_mode_effective": effective_load_mode,
        "status": "ok",
        "error": None,
    }

    try:
        t0 = time.perf_counter()
        tokenizer = _load_tokenizer(model_id=model_id, trust_remote_code=trust_remote_code)
        if getattr(tokenizer, "pad_token", None) is None:
            if getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif getattr(tokenizer, "unk_token", None) is not None:
                tokenizer.pad_token = tokenizer.unk_token
        t1 = time.perf_counter()

        model = None
        model_class = None
        config = None
        model_errors: list[str] = []
        for loader_class in (AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM):
            if effective_load_mode == "meta":
                try:
                    if config is None:
                        config = AutoConfig.from_pretrained(
                            model_id,
                            trust_remote_code=trust_remote_code,
                        )
                    model = loader_class.from_config(
                        config,
                        trust_remote_code=trust_remote_code,
                    ).to(device)
                    model_class = loader_class.__name__
                    break
                except Exception as exc:
                    model_errors.append(f"{loader_class.__name__}(from_config): {exc}")
                    continue
            else:
                for use_safetensors in (True, None):
                    try:
                        kwargs = {"trust_remote_code": trust_remote_code}
                        if use_safetensors is not None:
                            kwargs["use_safetensors"] = use_safetensors
                        model = loader_class.from_pretrained(model_id, **kwargs).to(device)
                        model_class = loader_class.__name__
                        break
                    except Exception as exc:
                        model_errors.append(
                            f"{loader_class.__name__}(use_safetensors={use_safetensors}): {exc}"
                        )
                        continue
            if model is not None:
                break
        if model is None:
            raise ValueError("Model load failed. " + " | ".join(model_errors[:4]))
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        t2 = time.perf_counter()

        first_param = next(iter(model.parameters()), None)
        dtype = str(first_param.dtype) if first_param is not None else "unknown"
        hidden_size = infer_hidden_size(model.config)

        token_alignment = _token_alignment_stats(tokenizer, sequence)
        encoded = tokenize_protein_sequences(tokenizer, [sequence], mode="auto")
        chosen_mode = encoded["tokenization_mode"]
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device=device)

        t3 = time.perf_counter()
        with torch.no_grad():
            outputs = _forward(model, input_ids=input_ids, attention_mask=attention_mask)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device=device)
        t4 = time.perf_counter()

        main_shape = _extract_main_output_shape(outputs)
        report.update(
            {
                "tokenizer_class": tokenizer.__class__.__name__,
                "model_class": model.__class__.__name__,
                "model_loader_class": model_class,
                "model_type": str(getattr(model.config, "model_type", "unknown")),
                "library_name": "transformers",
                "hidden_size": int(hidden_size),
                "num_parameters": _num_params(model),
                "parameter_dtype": dtype,
                "special_token_ids": [int(x) for x in (tokenizer.all_special_ids or [])],
                "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
                "token_alignment": token_alignment,
                "selected_tokenization_mode": chosen_mode,
                "input_ids_shape": [int(x) for x in input_ids.shape],
                "attention_mask_shape": [int(x) for x in attention_mask.shape],
                "main_output_shape": main_shape,
                "output_keys": list(outputs.keys()) if hasattr(outputs, "keys") else None,
                "timings_sec": {
                    "tokenizer_load": t1 - t0,
                    "model_load": t2 - t1,
                    "forward": t4 - t3,
                    "total": t4 - started,
                },
                "gpu_peak_memory_gb": (
                    _format_gb(torch.cuda.max_memory_allocated(device=device))
                    if device.startswith("cuda")
                    else None
                ),
            }
        )
    except Exception as exc:
        report["status"] = "error"
        report["error"] = str(exc)
        report["timings_sec"] = {"total": time.perf_counter() - started}

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Hugging Face PLMs for tokenizer/shape/runtime compatibility.")
    parser.add_argument(
        "--inventory-json",
        type=Path,
        default=Path(__file__).resolve().parent / "hf_protein_models_inventory_curated.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).resolve().parent / "hf_plm_probe_results.json",
    )
    parser.add_argument("--sequence", type=str, default=DEFAULT_SEQUENCE)
    parser.add_argument("--max-models", type=int, default=30)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-cpu", action="store_true", default=False)
    parser.add_argument(
        "--families",
        type=str,
        nargs="*",
        default=["esm1", "esm2", "esm3", "esmc", "protbert", "prott5", "ankh", "progen", "evo"],
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        choices=["downloads", "last_modified"],
        default="downloads",
        help="How to prioritize candidate models when --model-ids is not provided.",
    )
    parser.add_argument(
        "--model-ids",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit model IDs to probe (overrides family/download selection).",
    )
    parser.add_argument(
        "--load-mode",
        type=str,
        choices=["auto", "full", "meta"],
        default="auto",
        help="Model loading strategy: full weights, config-only/meta, or automatic by model size.",
    )
    args = parser.parse_args()

    payload = json.loads(args.inventory_json.read_text(encoding="utf-8"))
    device = _device_for_probe(force_cpu=args.force_cpu)

    if args.model_ids:
        curated_by_id = {row["model_id"]: row for row in payload["models"]}
        selected = []
        for model_id in args.model_ids:
            row = curated_by_id.get(model_id, {"model_id": model_id, "family": "manual", "downloads": 0})
            selected.append(row)
    else:
        selected = [
            row
            for row in payload["models"]
            if row.get("family") in set(args.families)
        ]
        if args.sort_by == "last_modified":
            selected.sort(
                key=lambda row: (
                    str(row.get("last_modified") or ""),
                    int(row.get("downloads", 0)),
                    row["model_id"],
                ),
                reverse=True,
            )
        else:
            selected.sort(key=lambda row: (-int(row.get("downloads", 0)), row["model_id"]))
        selected = selected[: max(1, args.max_models)]

    reports = []
    for index, row in enumerate(selected, start=1):
        model_id = row["model_id"]
        print(f"[{index}/{len(selected)}] probing {model_id}")
        report = probe_model(
            model_id=model_id,
            device=device,
            trust_remote_code=args.trust_remote_code,
            sequence=args.sequence,
            load_mode=args.load_mode,
        )
        report["family"] = row.get("family")
        report["downloads"] = row.get("downloads")
        reports.append(report)

    output_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "sequence": args.sequence,
        "inventory_json": str(args.inventory_json),
        "max_models": args.max_models,
        "reports": reports,
    }
    args.output_json.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    n_ok = sum(1 for rep in reports if rep.get("status") == "ok")
    print(f"Wrote probe results to {args.output_json} (ok={n_ok}/{len(reports)})")


if __name__ == "__main__":
    main()
