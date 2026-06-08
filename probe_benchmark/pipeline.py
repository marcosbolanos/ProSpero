from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import CachedTaskEmbeddingsLoader, TaskEmbeddings, build_feature_matrix
from .metrics import regression_metrics, spearmanr
from .models import DEFAULT_PROBE_SPECS, ProbeSpec, fit_best_probe
from .plotting import save_plots


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkTaskResult:
    task: str
    sequence_length: int
    n_train: int
    n_valid: int
    configs: dict[str, dict]


class BenchmarkRunner:
    def __init__(
        self,
        tasks: list[str],
        budgets: list[int],
        seeds: list[int],
        model_name: str,
        max_length: int | None,
        cache_root: str | Path | None,
        output_dir: str | Path,
        probe_specs: tuple[ProbeSpec, ...] = DEFAULT_PROBE_SPECS,
        require_cache: bool = True,
        compute_missing_embeddings: bool = False,
        embedding_batch_size: int = 64,
        skip_missing_tasks: bool = False,
        dpi: int = 200,
    ) -> None:
        self.tasks = tasks
        self.budgets = sorted(dict.fromkeys(budgets))
        self.seeds = seeds
        self.probe_specs = probe_specs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.skip_missing_tasks = skip_missing_tasks
        self.loader = CachedTaskEmbeddingsLoader(
            model_name=model_name,
            max_length=max_length,
            cache_root=cache_root,
            require_cache=require_cache,
            compute_missing_embeddings=compute_missing_embeddings,
            embedding_batch_size=embedding_batch_size,
        )

    def run(self) -> dict:
        records: list[dict] = []
        tasks_summary: dict[str, dict] = {}
        completed_tasks: list[str] = []
        skipped_tasks: dict[str, str] = {}

        for task in self.tasks:
            logger.info("Loading cached embeddings for task=%s", task)
            try:
                task_embeddings = self.loader.load_task(task)
            except FileNotFoundError as error:
                if not self.skip_missing_tasks:
                    raise
                logger.warning("Skipping task=%s because %s", task, error)
                skipped_tasks[task] = str(error)
                continue
            task_result, task_records = self._run_task(task_embeddings)
            tasks_summary[task] = {
                "sequence_length": task_result.sequence_length,
                "n_train": task_result.n_train,
                "n_valid": task_result.n_valid,
                "configs": task_result.configs,
            }
            records.extend(task_records)
            completed_tasks.append(task)

        if not tasks_summary:
            raise RuntimeError("No tasks were benchmarked successfully.")

        summary = {
            "metadata": {
                "requested_tasks": self.tasks,
                "completed_tasks": completed_tasks,
                "skipped_tasks": skipped_tasks,
                "budgets": self.budgets,
                "seeds": self.seeds,
                "probe_configs": {
                    spec.name: {
                        "feature_name": spec.feature_name,
                        "estimator_name": spec.estimator_name,
                        "search_grid": list(spec.search_grid),
                        "pca_components": spec.pca_components,
                        "description": spec.description,
                    }
                    for spec in self.probe_specs
                },
            },
            "tasks": tasks_summary,
        }

        self._write_summary(summary)
        self._write_records(records)
        save_plots(summary, self.output_dir / "plots", dpi=self.dpi)
        return summary

    def _run_task(
        self,
        task_embeddings: TaskEmbeddings,
    ) -> tuple[BenchmarkTaskResult, list[dict]]:
        feature_names = sorted({spec.feature_name for spec in self.probe_specs})
        train_features = {
            feature_name: build_feature_matrix(
                feature_name,
                task_embeddings.train_residue_embeddings,
                sequences=task_embeddings.train_sequences,
            )
            for feature_name in feature_names
        }
        valid_features = {
            feature_name: build_feature_matrix(
                feature_name,
                task_embeddings.valid_residue_embeddings,
                sequences=task_embeddings.valid_sequences,
            )
            for feature_name in feature_names
        }

        task_records: list[dict] = []
        config_summaries: dict[str, dict] = {}
        for spec in self.probe_specs:
            logger.info("Evaluating task=%s config=%s", task_embeddings.task, spec.name)
            budget_entries = []
            for budget in self.budgets:
                if budget > task_embeddings.n_train:
                    continue
                budget_records = self._run_budget(
                    task_embeddings=task_embeddings,
                    spec=spec,
                    budget=budget,
                    train_features=train_features[spec.feature_name],
                    valid_features=valid_features[spec.feature_name],
                )
                task_records.extend(budget_records)
                budget_entries.append(self._summarize_budget(budget, budget_records))

            config_summaries[spec.name] = {
                "description": spec.description,
                "feature_name": spec.feature_name,
                "estimator_name": spec.estimator_name,
                "budgets": budget_entries,
            }

        return (
            BenchmarkTaskResult(
                task=task_embeddings.task,
                sequence_length=task_embeddings.sequence_length,
                n_train=task_embeddings.n_train,
                n_valid=task_embeddings.n_valid,
                configs=config_summaries,
            ),
            task_records,
        )

    def _run_budget(
        self,
        task_embeddings: TaskEmbeddings,
        spec: ProbeSpec,
        budget: int,
        train_features: np.ndarray,
        valid_features: np.ndarray,
    ) -> list[dict]:
        records = []
        y_train_all = task_embeddings.train_scores
        y_valid = task_embeddings.valid_scores

        for seed in self.seeds:
            rng = np.random.RandomState(seed)
            indices = np.sort(rng.choice(task_embeddings.n_train, size=budget, replace=False))
            X_train = train_features[indices]
            y_train = y_train_all[indices]

            model, selected_hyperparameter = fit_best_probe(
                spec=spec,
                X_train=X_train,
                y_train=y_train,
                random_seed=seed,
                scorer=spearmanr,
            )
            predictions = model.predict(valid_features)
            metrics = regression_metrics(y_valid, predictions)
            records.append(
                {
                    "task": task_embeddings.task,
                    "config": spec.name,
                    "budget": budget,
                    "seed": seed,
                    "selected_hyperparameter": selected_hyperparameter,
                    "metrics": metrics,
                }
            )

        return records

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
                writer.writerow(
                    {
                        "task": record["task"],
                        "config": record["config"],
                        "budget": record["budget"],
                        "seed": record["seed"],
                        "selected_hyperparameter": record["selected_hyperparameter"],
                        **record["metrics"],
                    }
                )
