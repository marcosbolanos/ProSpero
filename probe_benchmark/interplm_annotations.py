from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np


INTERPLM_CONFIG_PATTERN = re.compile(r"^interplm_l(?P<layer>\d+)_(?P<estimator>[A-Za-z0-9_]+)$")


@dataclass(frozen=True)
class AnnotationRecord:
    feature: int
    concept: str
    description: str
    f1_per_domain: float
    precision: float | None
    recall: float | None
    threshold_pct: float | None


def parse_interplm_layer(config_name: str) -> int | None:
    match = INTERPLM_CONFIG_PATTERN.match(config_name.strip())
    if match is None:
        return None
    return int(match.group("layer"))


def parse_csv_list(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items or None


def parse_int_csv_list(value: str | None) -> tuple[int, ...] | None:
    parsed = parse_csv_list(value)
    if parsed is None:
        return None
    return tuple(int(item) for item in parsed)


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _annotation_sort_key(record: AnnotationRecord) -> tuple[float, float]:
    precision = -1.0 if record.precision is None else record.precision
    return (record.f1_per_domain, precision)


def load_interplm_annotations(annotation_csv: Path, min_f1: float = 0.0) -> dict[int, list[AnnotationRecord]]:
    annotations: dict[int, list[AnnotationRecord]] = defaultdict(list)
    if not annotation_csv.exists():
        return {}

    with open(annotation_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            feature_value = row.get("feature")
            concept = (row.get("concept") or "").strip()
            if feature_value is None or not concept:
                continue
            f1_per_domain = _safe_float(row.get("f1_per_domain"))
            if f1_per_domain is None or f1_per_domain < min_f1:
                continue
            feature = int(feature_value)
            annotations[feature].append(
                AnnotationRecord(
                    feature=feature,
                    concept=concept,
                    description=(row.get("description") or concept).strip(),
                    f1_per_domain=f1_per_domain,
                    precision=_safe_float(row.get("precision")),
                    recall=_safe_float(row.get("recall_per_domain") or row.get("recall")),
                    threshold_pct=_safe_float(row.get("threshold_pct")),
                )
            )

    deduped: dict[int, list[AnnotationRecord]] = {}
    for feature, rows in annotations.items():
        best_by_concept: dict[str, AnnotationRecord] = {}
        for row in rows:
            incumbent = best_by_concept.get(row.concept)
            if incumbent is None or _annotation_sort_key(row) > _annotation_sort_key(incumbent):
                best_by_concept[row.concept] = row
        deduped[feature] = sorted(
            best_by_concept.values(),
            key=_annotation_sort_key,
            reverse=True,
        )
    return deduped


def is_probably_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def download_annotation_file(url: str, destination: Path, timeout_seconds: float = 30.0) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=timeout_seconds) as response:
        payload = response.read()
    destination.write_bytes(payload)
    return destination


def resolve_annotation_file(
    *,
    layer: int,
    annotation_template: str | None,
    download_template: str | None,
    download_cache_dir: Path | None,
) -> Path | None:
    if annotation_template is not None:
        rendered = annotation_template.format(layer=layer)
        candidate = Path(rendered)
        if candidate.exists():
            return candidate
        if is_probably_url(rendered):
            if download_cache_dir is None:
                return None
            destination = download_cache_dir / f"layer_{layer}" / Path(urlparse(rendered).path).name
            try:
                return download_annotation_file(rendered, destination)
            except (HTTPError, URLError):
                return None

    if download_template is None:
        return None
    if download_cache_dir is None:
        raise ValueError("download_cache_dir is required when download_template is set.")

    url = download_template.format(layer=layer)
    destination = download_cache_dir / f"layer_{layer}" / Path(urlparse(url).path).name
    if destination.exists():
        return destination
    try:
        return download_annotation_file(url, destination)
    except (HTTPError, URLError):
        return None


def summarize_annotations(
    annotations_by_feature: dict[int, list[AnnotationRecord]],
    feature_id: int,
    max_concepts: int,
) -> tuple[str, str]:
    annotations = annotations_by_feature.get(feature_id, [])
    if not annotations:
        return "", ""
    top_annotations = annotations[:max_concepts]
    concept_names = "|".join(annotation.concept for annotation in top_annotations)
    concept_summary = "|".join(
        f"{annotation.concept}:{annotation.f1_per_domain:.3f}"
        for annotation in top_annotations
    )
    return concept_names, concept_summary


def _rank_desc(values: np.ndarray) -> dict[int, int]:
    if values.size == 0:
        return {}
    order = np.argsort(-values, kind="stable")
    return {int(index): rank + 1 for rank, index in enumerate(order)}


def select_feature_indices(
    coefficients: np.ndarray,
    top_k: int,
    coefficient_threshold: float,
    include_all_nonzero: bool,
) -> set[int]:
    feature_indices: set[int] = set()
    if include_all_nonzero:
        feature_indices.update(int(index) for index in np.flatnonzero(np.abs(coefficients) > coefficient_threshold))
        return feature_indices

    if top_k < 1:
        return feature_indices

    positive_indices = np.flatnonzero(coefficients > coefficient_threshold)
    negative_indices = np.flatnonzero(coefficients < -coefficient_threshold)
    if positive_indices.size > 0:
        positive_values = coefficients[positive_indices]
        order = np.argsort(-positive_values, kind="stable")[:top_k]
        feature_indices.update(int(index) for index in positive_indices[order])
    if negative_indices.size > 0:
        negative_values = -coefficients[negative_indices]
        order = np.argsort(-negative_values, kind="stable")[:top_k]
        feature_indices.update(int(index) for index in negative_indices[order])
    return feature_indices


def build_annotation_rows(
    coefficient_dir: Path,
    coefficient_index_rows: Iterable[dict[str, str]],
    annotation_template: str | None,
    annotation_download_template: str | None,
    annotation_download_cache_dir: Path | None,
    top_k: int,
    coefficient_threshold: float,
    max_concepts_per_feature: int,
    include_all_nonzero: bool,
) -> list[dict[str, object]]:
    annotation_cache: dict[int, dict[int, list[AnnotationRecord]]] = {}
    rows: list[dict[str, object]] = []

    for index_row in coefficient_index_rows:
        config_name = index_row["config"]
        layer = parse_interplm_layer(config_name)
        if layer is None:
            continue

        if layer not in annotation_cache:
            annotations_by_feature: dict[int, list[AnnotationRecord]] = {}
            annotation_path = resolve_annotation_file(
                layer=layer,
                annotation_template=annotation_template,
                download_template=annotation_download_template,
                download_cache_dir=annotation_download_cache_dir,
            )
            if annotation_path is not None:
                annotations_by_feature = load_interplm_annotations(annotation_path)
            annotation_cache[layer] = annotations_by_feature

        coefficient_path = coefficient_dir / index_row["coefficient_path"]
        with np.load(coefficient_path) as artifact:
            coefficients = np.asarray(artifact["coefficients"], dtype=np.float64).reshape(-1)

        selected_indices = select_feature_indices(
            coefficients=coefficients,
            top_k=top_k,
            coefficient_threshold=coefficient_threshold,
            include_all_nonzero=include_all_nonzero,
        )
        if not selected_indices:
            continue

        positive_ranks = _rank_desc(np.maximum(coefficients, 0.0))
        negative_ranks = _rank_desc(np.maximum(-coefficients, 0.0))
        absolute_ranks = _rank_desc(np.abs(coefficients))
        layer_annotations = annotation_cache[layer]

        for feature_id in sorted(selected_indices, key=lambda idx: abs(coefficients[idx]), reverse=True):
            coefficient_value = float(coefficients[feature_id])
            concept_names, concept_summary = summarize_annotations(
                annotations_by_feature=layer_annotations,
                feature_id=int(feature_id),
                max_concepts=max_concepts_per_feature,
            )
            rows.append(
                {
                    "task": index_row["task"],
                    "config": config_name,
                    "layer": layer,
                    "budget": int(index_row["budget"]),
                    "seed": int(index_row["seed"]),
                    "feature_id": int(feature_id),
                    "coefficient": coefficient_value,
                    "abs_coefficient": abs(coefficient_value),
                    "direction": "positive" if coefficient_value > 0 else "negative",
                    "absolute_rank": absolute_ranks[int(feature_id)],
                    "positive_rank": positive_ranks.get(int(feature_id), ""),
                    "negative_rank": negative_ranks.get(int(feature_id), ""),
                    "n_features": int(index_row["n_features"]),
                    "selected_hyperparameter": index_row["selected_hyperparameter"],
                    "selected_search_value": index_row["selected_search_value"],
                    "coefficient_path": index_row["coefficient_path"],
                    "concept_names": concept_names,
                    "concept_summary": concept_summary,
                }
            )
    return rows


def aggregate_annotation_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            row["task"],
            row["config"],
            row["layer"],
            row["budget"],
            row["feature_id"],
        )
        grouped[key].append(row)

    aggregated_rows: list[dict[str, object]] = []
    for key, group_rows in grouped.items():
        coefficients = np.asarray([float(row["coefficient"]) for row in group_rows], dtype=np.float64)
        abs_coefficients = np.abs(coefficients)
        positive_count = int(np.sum(coefficients > 0))
        negative_count = int(np.sum(coefficients < 0))
        top_hits = int(len(group_rows))
        sign_consistency = max(positive_count, negative_count) / top_hits
        concept_counter: dict[str, int] = defaultdict(int)
        for row in group_rows:
            concept_names = str(row["concept_names"]).strip()
            if not concept_names:
                continue
            for concept in concept_names.split("|"):
                if concept:
                    concept_counter[concept] += 1
        concepts_ranked = sorted(
            concept_counter.items(),
            key=lambda item: (-item[1], item[0]),
        )
        aggregated_rows.append(
            {
                "task": key[0],
                "config": key[1],
                "layer": key[2],
                "budget": key[3],
                "feature_id": key[4],
                "n_selected_runs": top_hits,
                "mean_coefficient": float(np.mean(coefficients)),
                "std_coefficient": float(np.std(coefficients)),
                "mean_abs_coefficient": float(np.mean(abs_coefficients)),
                "max_abs_coefficient": float(np.max(abs_coefficients)),
                "positive_runs": positive_count,
                "negative_runs": negative_count,
                "sign_consistency": float(sign_consistency),
                "concept_names": "|".join(concept for concept, _ in concepts_ranked[:5]),
            }
        )

    aggregated_rows.sort(
        key=lambda row: (
            str(row["task"]),
            str(row["config"]),
            int(row["budget"]),
            -float(row["mean_abs_coefficient"]),
        )
    )
    return aggregated_rows


def load_coefficient_index(
    coefficient_dir: Path,
    tasks: tuple[str, ...] | None = None,
    configs: tuple[str, ...] | None = None,
    budgets: tuple[int, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
) -> list[dict[str, str]]:
    coefficient_index_path = coefficient_dir / "coefficient_index.csv"
    if not coefficient_index_path.exists():
        raise FileNotFoundError(f"Missing coefficient index at {coefficient_index_path}")

    task_filter = None if tasks is None else set(tasks)
    config_filter = None if configs is None else set(configs)
    budget_filter = None if budgets is None else {int(value) for value in budgets}
    seed_filter = None if seeds is None else {int(value) for value in seeds}

    with open(coefficient_index_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            if task_filter is not None and row["task"] not in task_filter:
                continue
            if config_filter is not None and row["config"] not in config_filter:
                continue
            if budget_filter is not None and int(row["budget"]) not in budget_filter:
                continue
            if seed_filter is not None and int(row["seed"]) not in seed_filter:
                continue
            rows.append(row)
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_annotation_report(
    coefficient_dir: Path,
    output_dir: Path,
    tasks: tuple[str, ...] | None,
    configs: tuple[str, ...] | None,
    budgets: tuple[int, ...] | None,
    seeds: tuple[int, ...] | None,
    annotation_template: str | None = None,
    annotation_download_template: str | None = None,
    annotation_download_cache_dir: Path | None = None,
    top_k: int = 25,
    coefficient_threshold: float = 0.0,
    max_concepts_per_feature: int = 3,
    include_all_nonzero: bool = False,
) -> dict[str, object]:
    coefficient_rows = load_coefficient_index(
        coefficient_dir=coefficient_dir,
        tasks=tasks,
        configs=configs,
        budgets=budgets,
        seeds=seeds,
    )
    annotated_rows = build_annotation_rows(
        coefficient_dir=coefficient_dir,
        coefficient_index_rows=coefficient_rows,
        annotation_template=annotation_template,
        annotation_download_template=annotation_download_template,
        annotation_download_cache_dir=annotation_download_cache_dir,
        top_k=top_k,
        coefficient_threshold=coefficient_threshold,
        max_concepts_per_feature=max_concepts_per_feature,
        include_all_nonzero=include_all_nonzero,
    )
    aggregated_rows = aggregate_annotation_rows(annotated_rows)

    per_run_path = output_dir / "annotated_interplm_coefficients.csv"
    aggregated_path = output_dir / "annotated_interplm_coefficients_aggregated.csv"
    manifest_path = output_dir / "annotation_manifest.json"

    if annotated_rows:
        write_csv(
            per_run_path,
            annotated_rows,
            fieldnames=list(annotated_rows[0].keys()),
        )
    if aggregated_rows:
        write_csv(
            aggregated_path,
            aggregated_rows,
            fieldnames=list(aggregated_rows[0].keys()),
        )

    manifest = {
        "coefficient_dir": str(coefficient_dir),
        "n_coefficient_runs": len(coefficient_rows),
        "n_annotated_feature_rows": len(annotated_rows),
        "n_aggregated_feature_rows": len(aggregated_rows),
        "annotation_template": annotation_template,
        "annotation_download_template": annotation_download_template,
        "annotation_download_cache_dir": (
            None if annotation_download_cache_dir is None else str(annotation_download_cache_dir)
        ),
        "top_k": top_k,
        "coefficient_threshold": coefficient_threshold,
        "max_concepts_per_feature": max_concepts_per_feature,
        "include_all_nonzero": include_all_nonzero,
        "tasks": None if tasks is None else list(tasks),
        "configs": None if configs is None else list(configs),
        "budgets": None if budgets is None else list(budgets),
        "seeds": None if seeds is None else list(seeds),
        "outputs": {
            "per_run": str(per_run_path) if annotated_rows else None,
            "aggregated": str(aggregated_path) if aggregated_rows else None,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
