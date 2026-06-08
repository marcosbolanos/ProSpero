from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from prospero.dataset import RegressionDataset
from prospero.experiments_config import WT_SEQUENCES
from prospero.landscapes import get_landscape
from prospero.runners.run_protein import get_parser as get_base_parser
from prospero.surrogate import (
    build_surrogate_model,
    normalize_sequences,
    prepare_shared_esm_components,
)
from prospero.utils import set_seed

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _fitness(oracle, task: str, sequences: list[str]) -> np.ndarray:
    if task.startswith("D_SHIFT"):
        values = oracle.get_fitness(sequences)
    else:
        values = oracle.get_fitness(np.asarray(sequences))
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _batched_oracle_scores(
    oracle,
    task: str,
    sequences: list[str],
    batch_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    total_batches = int(math.ceil(len(sequences) / float(batch_size)))
    for batch_idx, start in enumerate(range(0, len(sequences), batch_size), start=1):
        batch = sequences[start : start + batch_size]
        outputs.append(_fitness(oracle, task, batch))
        print(
            f"[lambda-ranking] oracle scoring progress: {batch_idx}/{total_batches} batches",
            flush=True,
        )
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def _enumerate_single_mutants(wt_sequence: str) -> tuple[list[str], list[int], list[str], list[str]]:
    mutants: list[str] = []
    positions: list[int] = []
    wt_residues: list[str] = []
    mut_residues: list[str] = []
    for idx, wt_residue in enumerate(wt_sequence):
        for aa in AA_ALPHABET:
            if aa == wt_residue:
                continue
            seq_chars = list(wt_sequence)
            seq_chars[idx] = aa
            mutants.append("".join(seq_chars))
            positions.append(idx)
            wt_residues.append(wt_residue)
            mut_residues.append(aa)
    return mutants, positions, wt_residues, mut_residues


def _prepare_model(args, task: str, dataset: RegressionDataset, shared_esm_components):
    if args.max_train_samples is not None and len(dataset.train) > args.max_train_samples:
        rng = np.random.default_rng(args.seed + abs(hash(task)) % 10_000)
        keep = rng.choice(len(dataset.train), size=int(args.max_train_samples), replace=False)
        keep.sort()
        dataset.train = dataset.train[keep]
        dataset.train_scores = dataset.train_scores[keep]
    if args.max_valid_samples is not None and len(dataset.valid) > args.max_valid_samples:
        rng = np.random.default_rng(args.seed + 17 + abs(hash(task)) % 10_000)
        keep = rng.choice(len(dataset.valid), size=int(args.max_valid_samples), replace=False)
        keep.sort()
        dataset.valid = dataset.valid[keep]
        dataset.valid_scores = dataset.valid_scores[keep]

    ordered = _dedupe_preserving_order(normalize_sequences(list(dataset.train) + list(dataset.valid)))
    args.task = task
    args.cache_allowed_sequences = set(ordered)
    args.cache_allowed_sequences_ordered = ordered
    args.dataset_cache_task = task
    model = build_surrogate_model(
        seq_length=len(WT_SEQUENCES[task]),
        args=args,
        shared_esm_components=shared_esm_components,
    )
    model.train(dataset)
    return model


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr  # type: ignore

        return float(spearmanr(x, y).statistic)
    except Exception:
        # Tie-aware average rank via stable argsort fallback is overkill here; ties are rare.
        x_rank = np.argsort(np.argsort(x))
        y_rank = np.argsort(np.argsort(y))
        return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _compute_metrics_for_lambdas(
    lambdas: list[float],
    oracle_delta: np.ndarray,
    surrogate_delta: np.ndarray,
    prior_values: np.ndarray,
    prior_mode: str,
    recall_ks: list[int],
) -> list[dict[str, object]]:
    max_k = int(min(len(oracle_delta), max(recall_ks)))
    oracle_sorted = np.argsort(oracle_delta)[::-1]
    oracle_top_by_k = {k: set(map(int, oracle_sorted[: min(k, max_k)])) for k in recall_ks}
    out: list[dict[str, object]] = []

    for lam in lambdas:
        if prior_mode == "l2":
            score = surrogate_delta - float(lam) * prior_values
        elif prior_mode == "esm_logprob":
            score = surrogate_delta + float(lam) * prior_values
        else:
            raise ValueError(f"Unsupported prior_mode={prior_mode}")
        rank = np.argsort(score)[::-1]
        recalls: dict[str, float] = {}
        for k in recall_ks:
            kk = min(k, len(score))
            pred_top = set(map(int, rank[:kk]))
            oracle_top = oracle_top_by_k[k]
            recalls[f"recall_at_{k}"] = float(len(pred_top.intersection(oracle_top)) / float(kk))
        out.append(
            {
                "lambda": float(lam),
                "pearson_score_vs_oracle_delta": _safe_pearson(score, oracle_delta),
                "spearman_score_vs_oracle_delta": _spearman(score, oracle_delta),
                **recalls,
            }
        )
    return out


def _batched_surrogate_scores(model, sequences: list[str], batch_size: int) -> np.ndarray:
    outputs: list[np.ndarray] = []
    total_batches = int(math.ceil(len(sequences) / float(batch_size)))
    for batch_idx, start in enumerate(range(0, len(sequences), batch_size), start=1):
        batch = sequences[start : start + batch_size]
        preds = model.get_fitness(batch).detach().cpu().numpy().astype(np.float32)
        outputs.append(preds)
        print(
            f"[lambda-ranking] surrogate scoring progress: {batch_idx}/{total_batches} batches",
            flush=True,
        )
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def _compute_raw_token_distances(model, wt_residues: list[str], mut_residues: list[str]) -> np.ndarray:
    tokenizer = model.tokenizer
    embed_table = model.esm.get_input_embeddings().weight.detach()

    aa_ids = {aa: int(tokenizer.convert_tokens_to_ids(aa)) for aa in AA_ALPHABET}
    aa_emb = {aa: embed_table[idx] for aa, idx in aa_ids.items()}
    pair_distance: dict[tuple[str, str], float] = {}
    for a in AA_ALPHABET:
        for b in AA_ALPHABET:
            if a == b:
                pair_distance[(a, b)] = 0.0
            else:
                pair_distance[(a, b)] = float(
                    torch_norm(aa_emb[b].unsqueeze(0) - aa_emb[a].unsqueeze(0))[0]
                )

    distances = np.empty((len(wt_residues),), dtype=np.float32)
    for i, (a, b) in enumerate(zip(wt_residues, mut_residues)):
        distances[i] = np.float32(pair_distance[(a, b)])
    return distances


def _compute_esm_mutant_logprobs(
    tokenizer,
    esm_mlm,
    wt_sequence: str,
    positions: list[int],
    mut_residues: list[str],
    device: str,
) -> np.ndarray:
    encoded = tokenizer(
        [wt_sequence],
        padding=False,
        truncation=False,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        outputs = esm_mlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits[0]  # [T, V]
        log_probs = torch.log_softmax(logits, dim=-1)

    aa_ids = {aa: int(tokenizer.convert_tokens_to_ids(aa)) for aa in AA_ALPHABET}
    out = np.empty((len(positions),), dtype=np.float32)
    for i, (pos, aa) in enumerate(zip(positions, mut_residues)):
        token_pos = int(pos) + 1  # account for BOS token
        out[i] = float(log_probs[token_pos, aa_ids[aa]].item())
    return out


def _save_colored_table(rows: list[dict[str, object]], out_png: Path, recall_ks: list[int]) -> None:
    columns = [
        "task",
        "lambda",
        "pearson_score_vs_oracle_delta",
        "spearman_score_vs_oracle_delta",
        *[f"recall_at_{k}" for k in recall_ks],
    ]

    def _fmt(col: str, val: object) -> str:
        if col == "task":
            return str(val)
        if col == "lambda":
            return f"{float(val):.4g}"
        return f"{float(val):.4f}"

    cell_text = [[_fmt(col, row[col]) for col in columns] for row in rows]
    numeric_cols = [c for c in columns if c not in {"task"}]
    col_values = {
        c: np.asarray([float(r[c]) for r in rows], dtype=np.float64)
        for c in numeric_cols
    }
    col_norms: dict[str, tuple[float, float]] = {}
    for c, values in col_values.items():
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-12
        col_norms[c] = (vmin, vmax)

    cmap = plt.get_cmap("RdYlGn")
    n_rows = len(rows)
    fig_h = max(4.2, 0.35 * (n_rows + 3))
    fig_w = 15
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)

    # Header formatting.
    for col_idx in range(len(columns)):
        cell = table[0, col_idx]
        cell.set_facecolor("#253746")
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")

    # Body coloring by column-wise normalized value.
    for row_idx, row in enumerate(rows, start=1):
        for col_idx, col in enumerate(columns):
            cell = table[row_idx, col_idx]
            if col == "task":
                cell.set_facecolor("#f5f7fb")
                cell.get_text().set_weight("bold")
                continue
            val = float(row[col])
            vmin, vmax = col_norms[col]
            frac = (val - vmin) / (vmax - vmin)
            rgba = cmap(frac)
            cell.set_facecolor(rgba)

    ax.set_title(
        "Lambda-Penalized Single-Mutant Ranking Metrics\n"
        r"$S(j,b)=\Delta \hat f(j,a\to b)-\lambda\|z(x_{j\to b})-z(x)\|_2$",
        fontsize=12,
        pad=18,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _task_run(
    args,
    task: str,
    lambdas: list[float],
    recall_ks: list[int],
    shared_esm_components,
    prior_components,
    oracle_cache: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    start = time.perf_counter()
    wt_sequence = WT_SEQUENCES[task]
    dataset = RegressionDataset(task)
    oracle = get_landscape(task)

    model = _prepare_model(args, task, dataset, shared_esm_components=shared_esm_components)

    mutants, positions, wt_residues, mut_residues = _enumerate_single_mutants(wt_sequence)

    oracle_scores = None
    wt_oracle = None
    if oracle_cache is not None and task in oracle_cache:
        cached = oracle_cache[task]
        wt_oracle = float(cached["wt_oracle_fitness"])
        score_by_mutant: dict[str, float] = cached["score_by_mutant"]  # type: ignore[assignment]
        if all(mutant in score_by_mutant for mutant in mutants):
            oracle_scores = np.asarray(
                [score_by_mutant[mutant] for mutant in mutants], dtype=np.float32
            )
            print(f"[lambda-ranking] using cached oracle scores for task={task}", flush=True)
    if oracle_scores is None:
        wt_oracle = float(_fitness(oracle, task, [wt_sequence])[0])
        oracle_scores = _batched_oracle_scores(
            oracle=oracle,
            task=task,
            sequences=mutants,
            batch_size=args.oracle_batch_size,
        )
    assert wt_oracle is not None
    oracle_delta = oracle_scores - wt_oracle

    wt_surrogate = float(model.get_fitness([wt_sequence]).detach().cpu().item())
    predicted_scores = _batched_surrogate_scores(
        model=model,
        sequences=mutants,
        batch_size=args.proxy_batch_size,
    )
    if args.prior_mode == "l2":
        prior_values = _compute_raw_token_distances(
            model=model,
            wt_residues=wt_residues,
            mut_residues=mut_residues,
        )
    elif args.prior_mode == "esm_logprob":
        prior_tokenizer = prior_components["tokenizer"]
        prior_model = prior_components["model"]
        prior_device = prior_components["device"]
        prior_values = _compute_esm_mutant_logprobs(
            tokenizer=prior_tokenizer,
            esm_mlm=prior_model,
            wt_sequence=wt_sequence,
            positions=positions,
            mut_residues=mut_residues,
            device=prior_device,
        )
    else:
        raise ValueError(f"Unsupported prior_mode={args.prior_mode}")

    surrogate_delta = predicted_scores - wt_surrogate
    lambda_metrics = _compute_metrics_for_lambdas(
        lambdas=lambdas,
        oracle_delta=oracle_delta,
        surrogate_delta=surrogate_delta,
        prior_values=prior_values,
        prior_mode=args.prior_mode,
        recall_ks=recall_ks,
    )

    mutant_rows = []
    for i, seq in enumerate(mutants):
        mutant_rows.append(
            {
                "mutant_sequence": seq,
                "position": int(positions[i]),
                "wt_residue": wt_residues[i],
                "mutant_residue": mut_residues[i],
                "oracle_fitness": float(oracle_scores[i]),
                "oracle_delta_fitness": float(oracle_delta[i]),
                "surrogate_fitness": float(predicted_scores[i]),
                "surrogate_delta_fitness": float(surrogate_delta[i]),
                "prior_mode": args.prior_mode,
                "prior_value": float(prior_values[i]),
            }
        )

    return {
        "task": task,
        "wt_sequence_length": len(wt_sequence),
        "num_single_mutants": len(mutants),
        "wt_oracle_fitness": wt_oracle,
        "wt_surrogate_fitness": wt_surrogate,
        "lambda_metrics": lambda_metrics,
        "mutant_rows": mutant_rows,
        "timings_seconds": {"task_total_seconds": time.perf_counter() - start},
    }


def torch_norm(arr) -> np.ndarray:
    import torch

    if isinstance(arr, torch.Tensor):
        return torch.linalg.vector_norm(arr, ord=2, dim=1).cpu().numpy().astype(np.float32)
    arr_np = np.asarray(arr, dtype=np.float32)
    return np.linalg.norm(arr_np, axis=1).astype(np.float32)


def get_parser() -> argparse.ArgumentParser:
    parser = get_base_parser()
    parser.description = (
        "Rank single mutants by surrogate delta fitness penalized by ESM token-distance, "
        "then evaluate against oracle across lambda sweep."
    )
    parser.add_argument("--tasks", type=str, nargs="+", default=["AAV", "LGK"])
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0],
        help="Lambda sweep.",
    )
    parser.add_argument(
        "--prior-mode",
        type=str,
        choices=["l2", "esm_logprob"],
        default="l2",
        help=(
            "Prior term used in ranking score. "
            "`l2`: S = surrogate_delta - lambda*l2_distance. "
            "`esm_logprob`: S = surrogate_delta + lambda*logprob(mutant_aa)."
        ),
    )
    parser.add_argument(
        "--topk-recall",
        type=int,
        nargs="+",
        default=[10, 25, 50, 100, 250, 500],
        help="K values for top-K recall.",
    )
    parser.add_argument(
        "--oracle-batch-size",
        type=int,
        default=256,
        help="Oracle scoring batch size.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on surrogate training samples per task (for memory control).",
    )
    parser.add_argument(
        "--max-valid-samples",
        type=int,
        default=None,
        help="Optional cap on surrogate validation samples per task (for memory control).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/0423_epistasis",
        help="Output directory for json and png table.",
    )
    parser.add_argument(
        "--output-json-name",
        type=str,
        default=None,
        help="Optional output JSON filename (inside output-dir).",
    )
    parser.add_argument(
        "--output-png-name",
        type=str,
        default=None,
        help="Optional output PNG table filename (inside output-dir).",
    )
    parser.add_argument(
        "--oracle-cache-json",
        type=str,
        default=None,
        help=(
            "Optional JSON path with exhaustive single-mutant oracle scores to reuse. "
            "Expected format from run_single_mutant_energy_test *_full.json output."
        ),
    )
    parser.set_defaults(
        surrogate_arch="frozen_esm_flat_ridge",
        ensemble_size=1,
        proxy_batch_size=128,
        task="AAV",
        results_dirpath="outputs/0423_epistasis",
    )
    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed, args.full_deterministic)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_esm_components = prepare_shared_esm_components(args)
    prior_components = None
    if args.prior_mode == "esm_logprob":
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        prior_device = "cuda" if torch.cuda.is_available() else "cpu"
        prior_tokenizer = AutoTokenizer.from_pretrained(
            args.esm_model_name,
            local_files_only=True,
        )
        prior_model = AutoModelForMaskedLM.from_pretrained(
            args.esm_model_name,
            local_files_only=True,
        ).to(prior_device)
        prior_model.eval()
        for p in prior_model.parameters():
            p.requires_grad = False
        prior_components = {
            "tokenizer": prior_tokenizer,
            "model": prior_model,
            "device": prior_device,
        }
    else:
        prior_components = {}
    oracle_cache = None
    if args.oracle_cache_json:
        cache_path = Path(args.oracle_cache_json)
        if cache_path.exists():
            raw = json.loads(cache_path.read_text())
            parsed: dict[str, dict[str, object]] = {}
            for row in raw.get("results", []):
                task = row.get("task")
                if task is None:
                    continue
                wt_oracle = row.get("wt_scores", {}).get("oracle_fitness")
                per_rows = row.get("per_mutant_rows", [])
                score_by_mutant = {}
                for m in per_rows:
                    seq = m.get("mutant") or m.get("mutant_sequence")
                    score = m.get("oracle_fitness")
                    if seq is not None and score is not None:
                        score_by_mutant[str(seq)] = float(score)
                if wt_oracle is not None and score_by_mutant:
                    parsed[str(task)] = {
                        "wt_oracle_fitness": float(wt_oracle),
                        "score_by_mutant": score_by_mutant,
                    }
            oracle_cache = parsed
            print(
                f"[lambda-ranking] loaded oracle cache for tasks={sorted(oracle_cache.keys())}",
                flush=True,
            )
        else:
            print(
                f"[lambda-ranking] warning: oracle cache file not found: {cache_path}",
                flush=True,
            )
    run_start = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()

    task_results = []
    for task in args.tasks:
        print(f"[lambda-ranking] start task={task}", flush=True)
        task_results.append(
            _task_run(
                args=args,
                task=task,
                lambdas=[float(x) for x in args.lambdas],
                recall_ks=[int(k) for k in args.topk_recall],
                shared_esm_components=shared_esm_components,
                prior_components=prior_components,
                oracle_cache=oracle_cache,
            )
        )
        print(f"[lambda-ranking] done task={task}", flush=True)

    summary_rows: list[dict[str, object]] = []
    for task_result in task_results:
        task = task_result["task"]
        for row in task_result["lambda_metrics"]:
            summary_rows.append({"task": task, **row})

    if args.output_json_name:
        output_json = output_dir / args.output_json_name
    else:
        output_json = output_dir / (
            f"lambda_penalized_single_mutant_ranking_{'_'.join(t.lower() for t in args.tasks)}"
            f"_{args.prior_mode}.json"
        )
    if args.output_png_name:
        table_png = output_dir / args.output_png_name
    else:
        table_png = output_dir / f"lambda_penalized_ranking_metrics_table_{args.prior_mode}.png"
    _save_colored_table(summary_rows, table_png, recall_ks=[int(k) for k in args.topk_recall])

    payload = {
        "run_started_at_utc": started_at,
        "run_finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_runtime_seconds": time.perf_counter() - run_start,
        "config": {
            "surrogate_arch": args.surrogate_arch,
            "prior_mode": args.prior_mode,
            "score_definition": (
                "S = surrogate_delta_fitness - lambda * l2_distance"
                if args.prior_mode == "l2"
                else "S = surrogate_delta_fitness + lambda * esm_logprob(mutant_aa)"
            ),
            "prior_definition": (
                "L2 distance between raw ESM token embeddings for mutant and WT amino acids."
                if args.prior_mode == "l2"
                else "ESM log-probability of the mutant amino acid at the mutated position in WT context."
            ),
            "tasks": args.tasks,
            "lambdas": [float(x) for x in args.lambdas],
            "topk_recall": [int(k) for k in args.topk_recall],
            "proxy_batch_size": args.proxy_batch_size,
            "oracle_batch_size": args.oracle_batch_size,
            "max_train_samples": args.max_train_samples,
            "max_valid_samples": args.max_valid_samples,
        },
        "outputs": {"table_png": str(table_png)},
        "summary_rows": summary_rows,
        "task_results": task_results,
    }

    output_json.write_text(json.dumps(payload, indent=2))
    print(f"saved: {output_json}")
    print(f"saved: {table_png}")


if __name__ == "__main__":
    main()
