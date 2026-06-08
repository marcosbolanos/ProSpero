from __future__ import annotations

import math

import numpy as np


def _safe_float(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]

    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    true_std = np.std(y_true)
    pred_std = np.std(y_pred)
    if true_std == 0.0 or pred_std == 0.0:
        return 0.0
    return _safe_float(np.corrcoef(y_true, y_pred)[0, 1])


def spearmanr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    return pearsonr(_rankdata(y_true), _rankdata(y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    if total == 0.0:
        return 0.0
    residual = np.sum((y_true - y_pred) ** 2)
    return _safe_float(1.0 - residual / total)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return _safe_float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return _safe_float(np.mean(np.abs(y_true - y_pred)))


def top_k_overlap_rate(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    if len(y_true) == 0:
        return 0.0
    k = max(1, min(k, len(y_true)))
    true_top = set(np.argsort(y_true)[::-1][:k].tolist())
    pred_top = set(np.argsort(y_pred)[::-1][:k].tolist())
    return _safe_float(len(true_top & pred_top) / k)


def top_quantile_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float = 0.9,
    k: int | None = None,
) -> float:
    if len(y_true) == 0:
        return 0.0
    threshold = float(np.quantile(y_true, quantile))
    relevant = set(np.flatnonzero(y_true >= threshold).tolist())
    if not relevant:
        return 0.0
    if k is None:
        k = len(relevant)
    k = max(1, min(k, len(y_true)))
    pred_top = set(np.argsort(y_pred)[::-1][:k].tolist())
    return _safe_float(len(relevant & pred_top) / len(relevant))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    top_k = min(10, len(y_true))
    return {
        "pearson": pearsonr(y_true, y_pred),
        "spearman": spearmanr(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "top10_overlap_rate": top_k_overlap_rate(y_true, y_pred, top_k),
        "top_decile_recall": top_quantile_recall(y_true, y_pred, quantile=0.9),
    }
