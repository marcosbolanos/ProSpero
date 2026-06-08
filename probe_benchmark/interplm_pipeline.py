from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer  # type: ignore[reportMissingImports]

from prospero.dataset import RegressionDataset
from prospero.esm.cache import ESMEmbeddingFileCache
from prospero.experiments_config import WT_SEQUENCES
from prospero.surrogate import extract_residue_embeddings, normalize_sequences

from .features import sequences_to_one_hot_flat
from prospero.representations.interplm import (
    DEFAULT_INTERPLM_REPO_ID,
    load_interplm_sae,
    mean_pool_sae_activations,
)
from .metrics import regression_metrics, spearmanr
from .models import ProbeSpec, extract_linear_probe_parameters, fit_best_probe
from .plotting import (
    save_hyperparameter_diagnostic_plots,
    save_plots,
    save_spearman_summary_plots,
)


logger = logging.getLogger(__name__)

DEFAULT_INTERPLM_LAYERS = (1, 2, 3, 4, 5, 6)
DEFAULT_RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_L1_ALPHA_SCALE_GRID = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003)
DEFAULT_ELASTICNET_L1_RATIO = 0.9


def build_interplm_probe_specs(
    layers: tuple[int, ...] = DEFAULT_INTERPLM_LAYERS,
    estimator_name: str = "ridge",
    search_grid: tuple[float, ...] | None = None,
    elasticnet_l1_ratio: float = DEFAULT_ELASTICNET_L1_RATIO,
) -> tuple[ProbeSpec, ...]:
    normalized_estimator = estimator_name.strip().lower()
    if normalized_estimator == "ridge":
        default_grid = DEFAULT_RIDGE_GRID
        baseline_name = "one_hot_ridge"
        description_suffix = "with ridge regression."
        metadata: dict[str, str] = {}
    elif normalized_estimator == "lasso":
        default_grid = DEFAULT_L1_ALPHA_SCALE_GRID
        baseline_name = "one_hot_lasso"
        description_suffix = "with lasso regression."
        metadata = {"hyperparameter_scale": "alpha_over_alpha_max"}
    elif normalized_estimator == "elasticnet":
        default_grid = DEFAULT_L1_ALPHA_SCALE_GRID
        baseline_name = "one_hot_elasticnet"
        description_suffix = (
            f"with ElasticNet regression (l1_ratio={elasticnet_l1_ratio:.2f})."
        )
        metadata = {
            "hyperparameter_scale": "alpha_over_alpha_max",
            "l1_ratio": str(elasticnet_l1_ratio),
        }
    else:
        raise ValueError(
            f"Unsupported estimator_name={estimator_name!r}. "
            "Expected one of: ridge, lasso, elasticnet."
        )

    resolved_grid = tuple(default_grid if search_grid is None else search_grid)
    specs: list[ProbeSpec] = [
        ProbeSpec(
            name=baseline_name,
            feature_name="one_hot_flat",
            estimator_name=normalized_estimator,
            search_grid=resolved_grid,
            description=f"AA one-hot baseline {description_suffix}",
            metadata=metadata,
        )
    ]
    for layer in layers:
        layer_metadata = {"interplm_layer": str(layer), **metadata}
        specs.append(
            ProbeSpec(
                name=f"interplm_l{layer}_{normalized_estimator}",
                feature_name=f"interplm_l{layer}_mean_pool",
                estimator_name=normalized_estimator,
                search_grid=resolved_grid,
                description=(
                    f"Mean-pooled InterPLM SAE activations from ESM-2 hidden layer {layer} "
                    f"{description_suffix}"
                ),
                metadata=layer_metadata,
            )
        )
    return tuple(specs)


@dataclass(frozen=True)
class TaskFeatureSet:
    task: str
    sequence_length: int
    train_sequences: list[str]
    valid_sequences: list[str]
    train_scores: np.ndarray
    valid_scores: np.ndarray
    train_feature_matrices: dict[str, np.ndarray]
    valid_feature_matrices: dict[str, np.ndarray]

    @property
    def n_train(self) -> int:
        return len(self.train_sequences)

    @property
    def n_valid(self) -> int:
        return len(self.valid_sequences)


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


class InterPLMTaskFeatureLoader:
    def __init__(
        self,
        model_name: str,
        max_length: int | None,
        cache_root: str | Path | None = None,
        interplm_repo_id: str = DEFAULT_INTERPLM_REPO_ID,
        interplm_layers: tuple[int, ...] = DEFAULT_INTERPLM_LAYERS,
        interplm_normalized: bool = True,
        require_cache: bool = True,
        compute_missing_features: bool = False,
        embedding_batch_size: int = 32,
        sae_token_chunk_size: int = 1024,
        device: str = "auto",
        cache_source_embeddings: bool = False,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.interplm_repo_id = interplm_repo_id
        self.interplm_layers = tuple(sorted(dict.fromkeys(int(layer) for layer in interplm_layers)))
        self.interplm_normalized = interplm_normalized
        self.require_cache = require_cache
        self.compute_missing_features = compute_missing_features
        self.embedding_batch_size = embedding_batch_size
        self.sae_token_chunk_size = max(1, int(sae_token_chunk_size))
        self.device = self._resolve_device(device)
        self.cache_source_embeddings = bool(cache_source_embeddings)
        self._tokenizer = None
        self._esm = None
        self._sae_by_layer: dict[int, torch.nn.Module] = {}
        self.feature_caches = {
            layer: ESMEmbeddingFileCache(
                model_name=(
                    f"{self.model_name}__interplm_repo_{self.interplm_repo_id}"
                    f"__normalized_{self.interplm_normalized}"
                ),
                max_length=self.max_length,
                representation_name=f"interplm_mean_pool_v1__layer_{layer}",
                cache_root=cache_root,
            )
            for layer in self.interplm_layers
        }
        self.residue_embedding_caches = {
            layer: ESMEmbeddingFileCache(
                model_name=self.model_name,
                max_length=self.max_length,
                representation_name=f"interplm_source_residue_embeddings_v1__layer_{layer}",
                cache_root=cache_root,
            )
            for layer in self.interplm_layers
        }

    def _resolve_device(self, device: str) -> str:
        requested = str(device).strip().lower()
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Requested device=cuda, but no CUDA device is available.")
            return "cuda"
        if requested == "cpu":
            return "cpu"
        raise ValueError(f"Unsupported device={device!r}. Expected one of: auto, cpu, cuda.")

    def load_task(self, task: str) -> TaskFeatureSet:
        dataset = RegressionDataset(task)
        train_sequences = normalize_sequences(dataset.train.tolist())
        valid_sequences = normalize_sequences(dataset.valid.tolist())
        sequence_length = len(train_sequences[0])

        train_feature_matrices = self._load_feature_matrices(
            sequences=train_sequences,
            task=task,
            split_name="train",
        )
        valid_feature_matrices = self._load_feature_matrices(
            sequences=valid_sequences,
            task=task,
            split_name="valid",
        )

        return TaskFeatureSet(
            task=task,
            sequence_length=sequence_length,
            train_sequences=train_sequences,
            valid_sequences=valid_sequences,
            train_scores=np.asarray(dataset.train_scores, dtype=np.float64),
            valid_scores=np.asarray(dataset.valid_scores, dtype=np.float64),
            train_feature_matrices=train_feature_matrices,
            valid_feature_matrices=valid_feature_matrices,
        )

    def _load_feature_matrices(
        self,
        sequences: list[str],
        task: str,
        split_name: str,
    ) -> dict[str, np.ndarray]:
        features: dict[str, np.ndarray] = {
            "one_hot_flat": sequences_to_one_hot_flat(sequences),
        }
        cached_by_layer: dict[int, dict[str, torch.Tensor]] = {}
        missing_by_layer: dict[int, list[str]] = {}
        missing_all: list[str] = []

        for layer, cache in self.feature_caches.items():
            cached, missing = cache.get_many(sequences)
            cached_by_layer[layer] = cached
            missing_by_layer[layer] = missing
            missing_all.extend(missing)

        missing_unique = _dedupe_preserving_order(missing_all)
        if missing_unique:
            logger.info(
                "Building %d missing InterPLM feature rows for task=%s split=%s",
                len(missing_unique),
                task,
                split_name,
            )
            self._populate_missing_feature_caches(
                sequences=missing_unique,
                missing_by_layer=missing_by_layer,
                task=task,
                split_name=split_name,
            )
            for layer, cache in self.feature_caches.items():
                cached_by_layer[layer], _ = cache.get_many(sequences)

        for layer in self.interplm_layers:
            ordered = [cached_by_layer[layer][sequence].float() for sequence in sequences]
            features[f"interplm_l{layer}_mean_pool"] = (
                torch.stack(ordered, dim=0).cpu().numpy().astype(np.float32, copy=False)
            )

        return features

    def _populate_missing_feature_caches(
        self,
        sequences: list[str],
        missing_by_layer: dict[int, list[str]],
        task: str,
        split_name: str,
    ) -> None:
        residue_embeddings_by_layer = self._load_or_compute_residue_embeddings(
            sequences=sequences,
            task=task,
            split_name=split_name,
        )
        for layer in self.interplm_layers:
            layer_sequences = missing_by_layer[layer]
            if not layer_sequences:
                continue
            layer_embeddings = [residue_embeddings_by_layer[layer][sequence] for sequence in layer_sequences]
            features = self._compute_sae_features_for_layer(layer, layer_embeddings)
            self.feature_caches[layer].set_many(layer_sequences, features)

    def _load_or_compute_residue_embeddings(
        self,
        sequences: list[str],
        task: str,
        split_name: str,
    ) -> dict[int, dict[str, torch.Tensor]]:
        cached_residue_by_layer: dict[int, dict[str, torch.Tensor]] = {}
        missing_residue_by_layer: dict[int, list[str]] = {}
        missing_all: list[str] = []
        for layer, cache in self.residue_embedding_caches.items():
            cached, missing = cache.get_many(sequences)
            cached_residue_by_layer[layer] = cached
            missing_residue_by_layer[layer] = missing
            missing_all.extend(missing)

        missing_unique = _dedupe_preserving_order(missing_all)
        if missing_unique:
            if self.require_cache and not self.compute_missing_features:
                raise FileNotFoundError(
                    f"Missing {len(missing_unique)} cached InterPLM source residue embeddings "
                    f"for task={task} split={split_name}. Example sequence: {missing_unique[0]!r}"
                )
            if not self.compute_missing_features:
                raise FileNotFoundError(
                    f"Cache-only InterPLM benchmark cannot proceed with missing source residue "
                    f"embeddings for task={task} split={split_name}."
                )
            if not self.cache_source_embeddings:
                computed_by_layer = self._compute_residue_embeddings_by_layer(missing_unique)
                for layer in self.interplm_layers:
                    for sequence, embedding in zip(missing_unique, computed_by_layer[layer]):
                        cached_residue_by_layer[layer][sequence] = embedding
                return cached_residue_by_layer
            logger.info(
                "Computing cached ESM hidden-layer residue embeddings for %d sequences on %s",
                len(missing_unique),
                self.device,
            )
            computed_by_layer = self._compute_residue_embeddings_by_layer(missing_unique)
            for layer in self.interplm_layers:
                self.residue_embedding_caches[layer].set_many(
                    missing_unique,
                    computed_by_layer[layer],
                )
                for sequence, embedding in zip(missing_unique, computed_by_layer[layer]):
                    cached_residue_by_layer[layer][sequence] = embedding

        return cached_residue_by_layer

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def _get_esm(self):
        if self._esm is None:
            self._esm = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._esm.eval()
            for parameter in self._esm.parameters():
                parameter.requires_grad = False
        return self._esm

    def _get_sae(self, layer: int):
        if layer not in self._sae_by_layer:
            sae = load_interplm_sae(
                plm_layer=layer,
                repo_id=self.interplm_repo_id,
                normalized=self.interplm_normalized,
            ).to(self.device)
            sae.eval()
            self._sae_by_layer[layer] = sae
        return self._sae_by_layer[layer]

    def _tokenize_batch(self, sequences: list[str]):
        tokenizer = self._get_tokenizer()
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": self.max_length is not None,
        }
        if self.max_length is not None:
            tokenizer_kwargs["max_length"] = self.max_length
        encoded = tokenizer(sequences, **tokenizer_kwargs)
        return encoded["input_ids"], encoded["attention_mask"]

    def _mean_pool_sae_activations(
        self,
        sae,
        residue_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return mean_pool_sae_activations(
            sae,
            residue_embeddings,
            token_chunk_size=self.sae_token_chunk_size,
        )

    def _compute_residue_embeddings_by_layer(
        self,
        sequences: list[str],
    ) -> dict[int, list[torch.Tensor]]:
        esm = self._get_esm()
        embeddings_by_layer = {layer: [] for layer in self.interplm_layers}

        for start in range(0, len(sequences), self.embedding_batch_size):
            batch_sequences = sequences[start : start + self.embedding_batch_size]
            input_ids, attention_mask = self._tokenize_batch(batch_sequences)
            attention_mask = attention_mask.to(self.device)
            with torch.no_grad():
                outputs = esm(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            if outputs.hidden_states is None:
                raise RuntimeError("ESM model did not return hidden states.")

            for layer in self.interplm_layers:
                residue_embeddings = extract_residue_embeddings(
                    outputs.hidden_states[layer],
                    attention_mask,
                )
                for embedding in residue_embeddings.cpu():
                    embeddings_by_layer[layer].append(embedding)

        return embeddings_by_layer

    def _compute_sae_features_for_layer(
        self,
        layer: int,
        residue_embeddings: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if not residue_embeddings:
            return []
        sae = self._get_sae(layer)
        features: list[torch.Tensor] = []
        for start in range(0, len(residue_embeddings), self.embedding_batch_size):
            batch = residue_embeddings[start : start + self.embedding_batch_size]
            stacked = torch.stack(batch, dim=0).to(self.device)
            with torch.no_grad():
                pooled = self._mean_pool_sae_activations(sae, stacked).cpu()
            features.extend(pooled)
        return features


class InterPLMBenchmarkRunner:
    def __init__(
        self,
        tasks: list[str] = list(WT_SEQUENCES),
        budgets: list[int] | tuple[int, ...] = (16, 32, 64, 128, 256, 512),
        seeds: list[int] | tuple[int, ...] = (1, 2, 3, 4, 5),
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        max_length: int | None = None,
        cache_root: str | Path | None = None,
        output_dir: str | Path = "outputs/interplm_proxies",
        interplm_repo_id: str = DEFAULT_INTERPLM_REPO_ID,
        interplm_layers: tuple[int, ...] = DEFAULT_INTERPLM_LAYERS,
        interplm_normalized: bool = True,
        require_cache: bool = True,
        compute_missing_features: bool = False,
        embedding_batch_size: int = 32,
        sae_token_chunk_size: int = 1024,
        skip_missing_tasks: bool = False,
        dpi: int = 200,
        device: str = "auto",
        cache_source_embeddings: bool = False,
        estimator_name: str = "ridge",
        search_grid: tuple[float, ...] | None = None,
        elasticnet_l1_ratio: float = DEFAULT_ELASTICNET_L1_RATIO,
    ) -> None:
        self.tasks = tasks
        self.budgets = sorted(dict.fromkeys(int(budget) for budget in budgets))
        self.seeds = [int(seed) for seed in seeds]
        self.probe_specs = build_interplm_probe_specs(
            interplm_layers,
            estimator_name=estimator_name,
            search_grid=search_grid,
            elasticnet_l1_ratio=elasticnet_l1_ratio,
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.skip_missing_tasks = skip_missing_tasks
        self.estimator_name = estimator_name
        self.search_grid = tuple(search_grid) if search_grid is not None else None
        self.elasticnet_l1_ratio = float(elasticnet_l1_ratio)
        self.loader = InterPLMTaskFeatureLoader(
            model_name=model_name,
            max_length=max_length,
            cache_root=cache_root,
            interplm_repo_id=interplm_repo_id,
            interplm_layers=interplm_layers,
            interplm_normalized=interplm_normalized,
            require_cache=require_cache,
            compute_missing_features=compute_missing_features,
            embedding_batch_size=embedding_batch_size,
            sae_token_chunk_size=sae_token_chunk_size,
            device=device,
            cache_source_embeddings=cache_source_embeddings,
        )
        self.interplm_repo_id = interplm_repo_id
        self.interplm_layers = interplm_layers
        self.interplm_normalized = interplm_normalized
        self.device = self.loader.device
        self.cache_source_embeddings = cache_source_embeddings

    def run(self) -> dict:
        records: list[dict] = []
        diagnostic_records: list[dict] = []
        tasks_summary: dict[str, dict] = {}
        completed_tasks: list[str] = []
        skipped_tasks: dict[str, str] = {}

        for task in self.tasks:
            logger.info("Loading InterPLM features for task=%s", task)
            try:
                task_features = self.loader.load_task(task)
            except FileNotFoundError as error:
                if not self.skip_missing_tasks:
                    raise
                logger.warning("Skipping task=%s because %s", task, error)
                skipped_tasks[task] = str(error)
                continue
            task_result, task_records, task_diagnostics = self._run_task(task_features)
            tasks_summary[task] = {
                "sequence_length": task_result["sequence_length"],
                "n_train": task_result["n_train"],
                "n_valid": task_result["n_valid"],
                "configs": task_result["configs"],
            }
            records.extend(task_records)
            diagnostic_records.extend(task_diagnostics)
            completed_tasks.append(task)

        if not tasks_summary:
            raise RuntimeError("No InterPLM tasks were benchmarked successfully.")

        summary = {
            "metadata": {
                "requested_tasks": self.tasks,
                "completed_tasks": completed_tasks,
                "skipped_tasks": skipped_tasks,
                "budgets": self.budgets,
                "seeds": self.seeds,
                "model_name": self.loader.model_name,
                "max_length": self.loader.max_length,
                "interplm_repo_id": self.interplm_repo_id,
                "interplm_layers": list(self.interplm_layers),
                "interplm_normalized": self.interplm_normalized,
                "device": self.device,
                "cache_source_embeddings": self.cache_source_embeddings,
                "estimator_name": self.estimator_name,
                "search_grid": list(self.search_grid) if self.search_grid is not None else None,
                "elasticnet_l1_ratio": self.elasticnet_l1_ratio,
                "coefficient_export_format": "npz_input_space_v1",
                "probe_configs": {
                    spec.name: {
                        "feature_name": spec.feature_name,
                        "estimator_name": spec.estimator_name,
                        "search_grid": list(spec.search_grid),
                        "pca_components": spec.pca_components,
                        "description": spec.description,
                        "metadata": spec.metadata,
                    }
                    for spec in self.probe_specs
                },
            },
            "tasks": tasks_summary,
        }

        self._write_summary(summary)
        self._write_records(records)
        self._write_diagnostic_records(diagnostic_records)
        self._write_coefficient_index(records)
        save_plots(summary, self.output_dir / "plots", dpi=self.dpi)
        save_spearman_summary_plots(summary, self.output_dir / "summary_plots", dpi=self.dpi)
        save_hyperparameter_diagnostic_plots(
            diagnostic_records,
            self.output_dir / "supplementary_plots",
            dpi=self.dpi,
        )
        return summary

    def _run_task(self, task_features: TaskFeatureSet) -> tuple[dict, list[dict], list[dict]]:
        task_records: list[dict] = []
        task_diagnostics: list[dict] = []
        config_summaries: dict[str, dict] = {}

        for spec in self.probe_specs:
            logger.info("Evaluating task=%s config=%s", task_features.task, spec.name)
            budget_entries = []
            train_features = task_features.train_feature_matrices[spec.feature_name]
            valid_features = task_features.valid_feature_matrices[spec.feature_name]
            for budget in self.budgets:
                if budget > task_features.n_train:
                    continue
                budget_records, budget_diagnostics = self._run_budget(
                    task_features=task_features,
                    spec=spec,
                    budget=budget,
                    train_features=train_features,
                    valid_features=valid_features,
                )
                task_records.extend(budget_records)
                task_diagnostics.extend(budget_diagnostics)
                budget_entries.append(self._summarize_budget(budget, budget_records))

            config_summaries[spec.name] = {
                "description": spec.description,
                "feature_name": spec.feature_name,
                "estimator_name": spec.estimator_name,
                "budgets": budget_entries,
            }

        return (
            {
                "task": task_features.task,
                "sequence_length": task_features.sequence_length,
                "n_train": task_features.n_train,
                "n_valid": task_features.n_valid,
                "configs": config_summaries,
            },
            task_records,
            task_diagnostics,
        )

    def _run_budget(
        self,
        task_features: TaskFeatureSet,
        spec: ProbeSpec,
        budget: int,
        train_features: np.ndarray,
        valid_features: np.ndarray,
    ) -> tuple[list[dict], list[dict]]:
        records = []
        diagnostic_records = []
        y_train_all = task_features.train_scores
        y_valid = task_features.valid_scores

        for seed in self.seeds:
            rng = np.random.RandomState(seed)
            indices = np.sort(rng.choice(task_features.n_train, size=budget, replace=False))
            X_train = train_features[indices]
            y_train = y_train_all[indices]

            fit_result = fit_best_probe(
                spec=spec,
                X_train=X_train,
                y_train=y_train,
                random_seed=seed,
                scorer=spearmanr,
            )
            predictions = fit_result.model.predict(valid_features)
            metrics = regression_metrics(y_valid, predictions)
            record = {
                "task": task_features.task,
                "config": spec.name,
                "budget": budget,
                "seed": seed,
                "selected_hyperparameter": fit_result.selected_hyperparameter,
                "selected_search_value": fit_result.selected_search_value,
                "metrics": metrics,
            }
            linear_parameters = extract_linear_probe_parameters(fit_result.model)
            if linear_parameters is not None:
                record.update(
                    self._write_coefficient_artifact(
                        task=task_features.task,
                        spec=spec,
                        budget=budget,
                        seed=seed,
                        train_indices=indices,
                        selected_hyperparameter=fit_result.selected_hyperparameter,
                        selected_search_value=fit_result.selected_search_value,
                        coefficients=linear_parameters.input_coefficients,
                        intercept=linear_parameters.input_intercept,
                        model_coefficients=linear_parameters.model_coefficients,
                        model_intercept=linear_parameters.model_intercept,
                    )
                )
            records.append(record)
            for diagnostic in fit_result.hyperparameter_diagnostics:
                diagnostic_records.append(
                    {
                        "task": task_features.task,
                        "config": spec.name,
                        "budget": budget,
                        "seed": seed,
                        "search_value": diagnostic["search_value"],
                        "resolved_hyperparameter": diagnostic["resolved_hyperparameter"],
                        "score": diagnostic["score"],
                        "alpha_max": diagnostic["alpha_max"],
                    }
                )
        return records, diagnostic_records

    def _summarize_budget(self, budget: int, budget_records: list[dict]) -> dict:
        metric_names = sorted(budget_records[0]["metrics"])
        metrics_summary = {}
        for metric_name in metric_names:
            values = np.array(
                [record["metrics"][metric_name] for record in budget_records],
                dtype=float,
            )
            metrics_summary[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": [float(value) for value in values],
            }

        return {
            "budget": budget,
            "n_runs": len(budget_records),
            "selected_hyperparameters": [
                record["selected_hyperparameter"] for record in budget_records
            ],
            "selected_search_values": [
                record["selected_search_value"] for record in budget_records
            ],
            "metrics": metrics_summary,
        }

    def _write_summary(self, summary: dict) -> None:
        with open(self.output_dir / "benchmark_results.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    def _write_records(self, records: list[dict]) -> None:
        path = self.output_dir / "benchmark_records.csv"
        fieldnames = [
            "task",
            "config",
            "budget",
            "seed",
            "selected_hyperparameter",
            "selected_search_value",
            "pearson",
            "spearman",
            "r2",
            "rmse",
            "mae",
            "top10_overlap_rate",
            "top_decile_recall",
        ]
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                row = {
                    "task": record["task"],
                    "config": record["config"],
                    "budget": record["budget"],
                    "seed": record["seed"],
                    "selected_hyperparameter": record["selected_hyperparameter"],
                    "selected_search_value": record["selected_search_value"],
                }
                row.update(record["metrics"])
                writer.writerow(row)

    def _write_coefficient_index(self, records: list[dict]) -> None:
        coefficient_records = [record for record in records if "coefficient_path" in record]
        if not coefficient_records:
            return

        path = self.output_dir / "coefficient_index.csv"
        fieldnames = [
            "task",
            "config",
            "feature_name",
            "estimator_name",
            "budget",
            "seed",
            "selected_hyperparameter",
            "selected_search_value",
            "n_train_samples",
            "n_features",
            "intercept",
            "nnz_coefficients",
            "l1_norm",
            "l2_norm",
            "coefficient_path",
        ]
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in coefficient_records:
                writer.writerow(
                    {
                        field: record[field]
                        for field in fieldnames
                    }
                )

    def _write_diagnostic_records(self, records: list[dict]) -> None:
        if not records:
            return
        path = self.output_dir / "alpha_diagnostics.csv"
        fieldnames = [
            "task",
            "config",
            "budget",
            "seed",
            "search_value",
            "resolved_hyperparameter",
            "score",
            "alpha_max",
        ]
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(record)

    def _sanitize_path_component(self, value: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
        return normalized.strip("._") or "item"

    def _write_coefficient_artifact(
        self,
        task: str,
        spec: ProbeSpec,
        budget: int,
        seed: int,
        train_indices: np.ndarray,
        selected_hyperparameter: object,
        selected_search_value: object,
        coefficients: np.ndarray,
        intercept: float,
        model_coefficients: np.ndarray,
        model_intercept: float,
    ) -> dict:
        relative_path = Path("coefficients") / self._sanitize_path_component(task) / self._sanitize_path_component(
            spec.name
        ) / f"budget_{budget}" / f"seed_{seed}.npz"
        artifact_path = self.output_dir / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact_path,
            coefficients=np.asarray(coefficients, dtype=np.float32),
            intercept=np.asarray([intercept], dtype=np.float64),
            model_coefficients=np.asarray(model_coefficients, dtype=np.float32),
            model_intercept=np.asarray([model_intercept], dtype=np.float64),
            train_indices=np.asarray(train_indices, dtype=np.int64),
        )
        return {
            "task": task,
            "config": spec.name,
            "feature_name": spec.feature_name,
            "estimator_name": spec.estimator_name,
            "budget": budget,
            "seed": seed,
            "selected_hyperparameter": selected_hyperparameter,
            "selected_search_value": selected_search_value,
            "n_train_samples": int(len(train_indices)),
            "n_features": int(coefficients.shape[0]),
            "intercept": float(intercept),
            "nnz_coefficients": int(np.count_nonzero(coefficients)),
            "l1_norm": float(np.sum(np.abs(coefficients))),
            "l2_norm": float(np.linalg.norm(coefficients)),
            "coefficient_path": str(relative_path),
        }
