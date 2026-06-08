from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from evodiff.pretrained import OA_DM_38M  # type: ignore[reportMissingImports]
from transformers import AutoModelForMaskedLM, AutoTokenizer

from latent_search.latent_direction_search import masked_latent_direction_search
from prospero.dataset import RegressionDataset
from prospero.experiments_config import ALPHABETS, WT_SEQUENCES
from prospero.inference import ProteinSampler
from prospero.landscapes import get_landscape
from prospero.runners.run_protein import FROZEN_ESM_SURROGATE_ARCHS
from prospero.surrogate import (
    Ensemble,
    build_surrogate_model,
    normalize_sequences,
    prepare_shared_esm_components,
)
from prospero.utils import set_seed


@dataclass(frozen=True)
class FeatureDirection:
    feature_index: int
    coefficient: float
    decoder_weight_norm: float


@dataclass(frozen=True)
class AveragedDirection:
    top_feature_count: int
    features: list[FeatureDirection]
    vector: torch.Tensor


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _decode_to_sequence(decoded: str) -> str:
    return decoded.replace(" ", "")


def _prepare_proxy(args, sequence_length: int, dataset: RegressionDataset) -> Ensemble:
    if getattr(args, "disable_esm_cache", False):
        args.cache_allowed_sequences = set()
        args.cache_allowed_sequences_ordered = []
        args.dataset_cache_task = None
    else:
        ordered = _dedupe_preserving_order(
            normalize_sequences(list(dataset.train) + list(dataset.valid))
        )
        args.cache_allowed_sequences = set(ordered)
        args.cache_allowed_sequences_ordered = ordered
        args.dataset_cache_task = args.task

    shared_esm_components = None
    if args.surrogate_arch in FROZEN_ESM_SURROGATE_ARCHS:
        shared_esm_components = prepare_shared_esm_components(args)

    proxy = Ensemble(
        [
            build_surrogate_model(
                sequence_length,
                args,
                shared_esm_components=shared_esm_components,
            )
            for _ in range(args.ensemble_size)
        ]
    )
    proxy.train(dataset)
    return proxy


def _numeric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
    }


def _extract_directions(
    proxy: Ensemble,
    top_feature_counts: list[int],
) -> tuple[torch.Tensor, list[AveragedDirection], np.ndarray]:
    models = proxy.models
    coef_stack = []
    for model in models:
        coef = np.asarray(model.regressor.coef_, dtype=np.float32).reshape(-1)
        scale = np.asarray(model.scaler.scale_, dtype=np.float32).reshape(-1)
        effective_coef = coef / np.where(scale == 0.0, 1.0, scale)
        coef_stack.append(effective_coef)

    mean_coef = np.mean(np.stack(coef_stack, axis=0), axis=0)
    ranked_indices = np.argsort(mean_coef)[::-1]

    sae = models[0]._get_sae()
    decoder = sae.decoder.weight.detach().to(torch.float32)

    averaged_directions: list[AveragedDirection] = []
    direction_vectors: list[torch.Tensor] = []
    for top_k in top_feature_counts:
        capped_top_k = max(1, min(int(top_k), int(mean_coef.size)))
        feature_indices = ranked_indices[:capped_top_k].tolist()
        feature_meta: list[FeatureDirection] = []
        unit_vectors: list[torch.Tensor] = []

        for feature_idx in feature_indices:
            raw_vec = decoder[:, feature_idx]
            raw_norm = float(raw_vec.norm(p=2).item())
            if raw_norm <= 0.0:
                unit_vec = raw_vec
            else:
                unit_vec = raw_vec / raw_vec.norm(p=2)
            unit_vectors.append(unit_vec)
            feature_meta.append(
                FeatureDirection(
                    feature_index=int(feature_idx),
                    coefficient=float(mean_coef[feature_idx]),
                    decoder_weight_norm=raw_norm,
                )
            )

        average_vec = torch.stack(unit_vectors, dim=0).mean(dim=0)
        average_norm = average_vec.norm(p=2)
        if float(average_norm.item()) > 0.0:
            average_vec = average_vec / average_norm

        direction_vectors.append(average_vec)
        averaged_directions.append(
            AveragedDirection(
                top_feature_count=capped_top_k,
                features=feature_meta,
                vector=average_vec,
            )
        )

    return torch.stack(direction_vectors, dim=0), averaged_directions, mean_coef


def _build_masked_esm_inputs(
    *,
    starting_sequence: str,
    masked_positions: list[list[int]],
    tokenizer,
    device: torch.device,
    max_length: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences = [starting_sequence] * len(masked_positions)
    tokenizer_kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": max_length is not None,
    }
    if max_length is not None:
        tokenizer_kwargs["max_length"] = max_length
    encoded = tokenizer(sequences, **tokenizer_kwargs)

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    token_steering_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    seq_width = int(input_ids.shape[1])
    for row_idx, positions in enumerate(masked_positions):
        for residue_idx in positions:
            token_idx = int(residue_idx) + 1
            if 0 <= token_idx < seq_width:
                token_steering_mask[row_idx, token_idx] = True
    return (
        input_ids.to(device),
        attention_mask.to(device),
        token_steering_mask.to(device),
    )


def run_latent_masked_single_iter(args) -> Path:
    run_started_at = datetime.now(timezone.utc).isoformat()
    run_start = time.perf_counter()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    set_seed(args.seed, args.full_deterministic)
    timings["seed_setup_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    wt_sequence = WT_SEQUENCES[args.task]
    dataset = RegressionDataset(args.task)
    timings["dataset_load_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    proxy = _prepare_proxy(args, len(wt_sequence), dataset)
    timings["proxy_prepare_and_train_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    top_feature_counts = [int(x) for x in getattr(args, "top_feature_counts", [args.top_features])]
    top_feature_counts = [x for x in top_feature_counts if x > 0]
    if not top_feature_counts:
        top_feature_counts = [1]
    steering_directions, averaged_directions, mean_coef = _extract_directions(
        proxy, top_feature_counts
    )
    timings["direction_extraction_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    alphabet = ALPHABETS[args.alphabet]
    oadm_model, _, tokenizer_oadm, _ = OA_DM_38M()
    oadm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oadm_model = oadm_model.to(oadm_device)
    sampler = ProteinSampler(oadm_model, tokenizer_oadm, alphabet)

    sample, locs, _ = sampler.shotgun_alanine_scan(
        wt_sequence,
        proxy,
        args.min_corruptions,
        args.max_corruptions,
        args.batch_size,
        args.n_checks_multiplier,
        args.kappa_scan,
    )
    del sample
    timings["targeted_masking_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    masked_positions = [
        [int(idx) for idx in row.tolist() if int(idx) != sampler.PAD]
        for row in locs
    ]
    timings["masked_positions_extract_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    model_name = args.esm_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    timings["masked_lm_load_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    input_ids, attention_mask, token_steering_mask = _build_masked_esm_inputs(
        starting_sequence=wt_sequence,
        masked_positions=masked_positions,
        tokenizer=tokenizer,
        device=device,
        max_length=args.esm_max_length,
    )
    inputs_embeds = model.esm.embeddings(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    timings["masked_input_build_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    s_values = torch.tensor(args.steering_scalars, dtype=torch.float32, device=device)
    result = masked_latent_direction_search(
        model=model,
        n=args.steering_layer,
        steering_directions=steering_directions.to(device),
        s_values=s_values,
        inputs_embeds=inputs_embeds,
        token_steering_mask=token_steering_mask,
        attention_mask=attention_mask,
        combo_chunk_size=args.combo_chunk_size,
    )
    timings["latent_direction_search_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    num_directions, num_scales, batch_size, seq_len = result.token_ids.shape
    flat_records: list[dict[str, Any]] = []
    combo_summaries: dict[str, dict[str, Any]] = {}
    sequence_mutation_counts: list[float] = []
    for d_idx in range(num_directions):
        direction_meta = averaged_directions[d_idx]
        for s_idx in range(num_scales):
            token_batch = result.token_ids[d_idx, s_idx]
            logp_batch = result.pred_token_log_probs[d_idx, s_idx]
            decoded = tokenizer.batch_decode(token_batch, skip_special_tokens=True)
            combo_key = f"top{direction_meta.top_feature_count}_scalar{s_values[s_idx].item():.6g}"
            combo_summary = combo_summaries.setdefault(
                combo_key,
                {
                    "top_feature_count": int(direction_meta.top_feature_count),
                    "steering_scalar": float(s_values[s_idx].item()),
                    "num_candidates": 0,
                    "mean_num_mutations_vs_wt": 0.0,
                    "masked_token_perplexities": [],
                },
            )
            for b_idx in range(batch_size):
                sequence = _decode_to_sequence(decoded[b_idx])
                positions = masked_positions[b_idx]
                token_positions = [pos + 1 for pos in positions]
                masked_logp_values = [
                    float(logp_batch[b_idx, token_pos].item())
                    for token_pos in token_positions
                    if token_pos < seq_len
                ]
                ppl_values = [float(np.exp(-x)) for x in masked_logp_values]
                residue_length = len(sequence)
                residue_token_positions = list(range(1, min(seq_len - 1, residue_length + 1)))
                residue_logp_values = [
                    float(logp_batch[b_idx, token_pos].item()) for token_pos in residue_token_positions
                ]
                residue_ppl_values = [float(np.exp(-x)) for x in residue_logp_values]

                mutation_count = float(
                    sum(1 for aa_new, aa_wt in zip(sequence, wt_sequence) if aa_new != aa_wt)
                )
                sequence_mutation_counts.append(mutation_count)
                combo_summary["num_candidates"] = int(combo_summary["num_candidates"]) + 1
                combo_summary["mean_num_mutations_vs_wt"] += mutation_count
                combo_summary["masked_token_perplexities"].extend(ppl_values)

                flat_records.append(
                    {
                        "sequence": sequence,
                        "top_feature_count": int(direction_meta.top_feature_count),
                        "features_used": [
                            {
                                "feature_index": int(meta.feature_index),
                                "coefficient": float(meta.coefficient),
                                "decoder_weight_norm": float(meta.decoder_weight_norm),
                            }
                            for meta in direction_meta.features
                        ],
                        "steering_scalar": float(s_values[s_idx].item()),
                        "masked_token_positions": positions,
                        "num_masked_positions": int(len(positions)),
                        "predicted_token_log_probs_residue": residue_logp_values,
                        "predicted_token_perplexities_residue": residue_ppl_values,
                        "masked_token_log_probs": masked_logp_values,
                        "masked_token_perplexities": ppl_values,
                        "num_mutations_vs_wt": mutation_count,
                    }
                )
    for combo_summary in combo_summaries.values():
        count = int(combo_summary["num_candidates"])
        if count > 0:
            combo_summary["mean_num_mutations_vs_wt"] = float(
                combo_summary["mean_num_mutations_vs_wt"] / count
            )
        combo_summary["masked_token_perplexity_summary"] = _numeric_summary(
            [float(x) for x in combo_summary["masked_token_perplexities"]]
        )
        del combo_summary["masked_token_perplexities"]
    timings["candidate_decode_and_metrics_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    unique_records_by_sequence: dict[str, dict[str, object]] = {}
    for record in flat_records:
        sequence = str(record["sequence"])
        if sequence in unique_records_by_sequence:
            continue
        unique_records_by_sequence[sequence] = record
    timings["dedupe_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    unique_sequences = list(unique_records_by_sequence.keys())
    surrogate_scores = proxy.get_scores(unique_sequences).detach().cpu().numpy()
    order = np.argsort(surrogate_scores)[::-1]
    top_indices = order[: args.top_k]
    top_sequences = [unique_sequences[idx] for idx in top_indices]
    top_surrogate_scores = [float(surrogate_scores[idx]) for idx in top_indices]
    timings["surrogate_rerank_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    oracle = get_landscape(args.task)
    if args.task.startswith("D_SHIFT"):
        oracle_scores_np = oracle.get_fitness(top_sequences)
    else:
        oracle_scores_np = oracle.get_fitness(np.array(top_sequences))
    top_oracle_scores = [float(score) for score in np.asarray(oracle_scores_np).reshape(-1)]
    timings["oracle_scoring_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    output_records = []
    for sequence, surrogate_score, oracle_score in zip(
        top_sequences, top_surrogate_scores, top_oracle_scores
    ):
        record = dict(unique_records_by_sequence[sequence])
        record["surrogate_score"] = surrogate_score
        record["oracle_score"] = oracle_score
        output_records.append(record)
    timings["output_record_assembly_seconds"] = time.perf_counter() - t0

    run_finished_at = datetime.now(timezone.utc).isoformat()
    timings["total_runtime_seconds"] = time.perf_counter() - run_start
    masked_counts = [len(p) for p in masked_positions]
    top_surrogate_summary = _numeric_summary(top_surrogate_scores)
    top_oracle_summary = _numeric_summary(top_oracle_scores)
    all_surrogate_summary = _numeric_summary([float(x) for x in surrogate_scores.tolist()])
    oracle_minus_surrogate = [
        float(oracle_score - surrogate_score)
        for surrogate_score, oracle_score in zip(top_surrogate_scores, top_oracle_scores)
    ]

    output = {
        "task": args.task,
        "seed": int(args.seed),
        "run_started_at_utc": run_started_at,
        "run_finished_at_utc": run_finished_at,
        "wt_sequence": wt_sequence,
        "steering_layer": int(args.steering_layer),
        "top_features": int(args.top_features),
        "top_feature_counts": [int(x) for x in top_feature_counts],
        "steering_scalars": [float(x) for x in args.steering_scalars],
        "batch_size": int(args.batch_size),
        "top_k": int(args.top_k),
        "surrogate_arch": str(args.surrogate_arch),
        "esm_model_name": str(args.esm_model_name),
        "num_generated": int(len(flat_records)),
        "num_unique": int(len(unique_records_by_sequence)),
        "unique_rate": float(len(unique_records_by_sequence) / max(1, len(flat_records))),
        "num_masked_sequences": int(len(masked_positions)),
        "masked_positions_per_sequence_summary": _numeric_summary(
            [float(x) for x in masked_counts]
        ),
        "mutation_count_summary_vs_wt": _numeric_summary(sequence_mutation_counts),
        "feature_coefficient_summary": {
            "all_features": _numeric_summary([float(x) for x in mean_coef.tolist()]),
            "top_100_coefficients": _numeric_summary(
                [float(x) for x in np.sort(mean_coef)[::-1][:100].tolist()]
            ),
        },
        "direction_sets": [
            {
                "top_feature_count": int(direction.top_feature_count),
                "features_used": [
                    {
                        "feature_index": int(f.feature_index),
                        "coefficient": float(f.coefficient),
                        "decoder_weight_norm": float(f.decoder_weight_norm),
                    }
                    for f in direction.features
                ],
            }
            for direction in averaged_directions
        ],
        "combo_summaries": list(combo_summaries.values()),
        "score_summaries": {
            "unique_candidates_surrogate": all_surrogate_summary,
            "top_k_surrogate": top_surrogate_summary,
            "top_k_oracle": top_oracle_summary,
            "top_k_oracle_minus_surrogate": _numeric_summary(oracle_minus_surrogate),
        },
        "timings_seconds": timings,
        "top_records": output_records,
    }

    timings["output_write_seconds"] = 0.0
    t0 = time.perf_counter()
    output_dir = Path(args.results_dirpath) / args.task
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"seed_{args.seed}_latent_masked_single_iter.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    timings["output_write_seconds"] = time.perf_counter() - t0
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    return output_path
