from __future__ import annotations

import json
import logging
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from prospero.dataset import RegressionDataset
from prospero.experiment_tracker import ExperimentTracker
from prospero.experiments_config import WT_SEQUENCES
from prospero.landscapes import get_landscape
from prospero.optimization.core import (
    CampaignState,
    TraceWriter,
    load_wild_type_fitness,
)
from prospero.optimization.types import CampaignConfig, ModelConfig
from prospero.utils import set_seed


class OptimizationModel(Protocol):
    adaptation_enabled: bool

    @property
    def config(self) -> ModelConfig: ...

    def set_trace_writer(self, writer: TraceWriter) -> None: ...

    def set_trace_context(self, **context: Any) -> None: ...

    def trace_event(self, event: str, **payload: Any) -> None: ...

    def generate_batch(
        self, incumbent: str, batch_size: int
    ) -> tuple[Sequence[str], Sequence[float] | np.ndarray]: ...

    def update_mask_position_rewards(
        self,
        incumbent: str,
        sequences: Sequence[str],
        fitnesses: Sequence[float],
    ) -> None: ...

    def adapt(
        self,
        sequences: Sequence[str],
        fitnesses: Sequence[float],
        output_directory: Path,
        round_index: int,
    ) -> list[dict[str, Any]]: ...


def _write_trace_summary(writer: TraceWriter) -> None:
    summary_path = writer.path.with_suffix("").with_suffix(".trace_summary.json")
    summary_path.write_text(
        json.dumps({"event_counts": writer.counts}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_optimization_campaign(
    config: CampaignConfig,
    model: OptimizationModel,
    metadata: dict[str, Any],
    *,
    logger: logging.Logger,
    artifact_prefix: str,
    adaptation_result_label: str,
) -> Path:
    set_seed(config.seed, config.deterministic_algorithms)
    output_directory = config.output_directory / config.task
    output_directory.mkdir(parents=True, exist_ok=True)
    result_path = output_directory / f"seed_{config.seed}.pkl"
    completion_path = output_directory / f"seed_{config.seed}.complete.json"
    metadata_path = (
        output_directory / f"seed_{config.seed}.{artifact_prefix}_metadata.json"
    )
    trace_writer = (
        TraceWriter(
            output_directory / "debug_traces" / f"seed_{config.seed}.events.jsonl.gz"
        )
        if config.write_traces
        else None
    )

    oracle = get_landscape(config.task)
    # The benchmark exposes only the wild-type sequence and fitness at startup.
    reference_dataset = RegressionDataset(config.task)
    wild_type = WT_SEQUENCES[config.task]
    campaign = CampaignState(
        wild_type, load_wild_type_fitness(reference_dataset, wild_type)
    )
    # The complete initialization set is retained only for post-hoc reporting.
    tracker = ExperimentTracker(
        logger,
        deepcopy(reference_dataset),
        wild_type,
        best_percentile=0.95,
    )
    if trace_writer is not None:
        model.set_trace_writer(trace_writer)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    replay_sequences: list[str] = []
    replay_fitnesses: list[float] = []
    adaptation_directory = (
        output_directory / f"seed_{config.seed}.{artifact_prefix}_online_adaptation"
    )
    try:
        for round_index in range(1, config.rounds + 1):
            candidate_pool: dict[str, tuple[str, float, str]] = {}
            excluded_sequences = campaign.excluded_sequences
            planned_batches = math.ceil(
                max(config.query_budget, config.candidate_batch_size)
                / config.candidate_batch_size
            )
            generation_batch = 0
            while (
                generation_batch < planned_batches
                or len(candidate_pool) < config.query_budget
            ):
                generation_batch += 1
                incumbent = campaign.incumbent_sequence
                model.set_trace_context(
                    task=config.task,
                    seed=config.seed,
                    query_budget=config.query_budget,
                    optimization_round=round_index,
                    generation_round=generation_batch,
                    method=artifact_prefix,
                    mask_strategy="mixed_position_entropy_anticollapse",
                    mask_budget=model.config.masking.budget,
                )
                generated_sequences, generated_scores = model.generate_batch(
                    incumbent, config.candidate_batch_size
                )
                for sequence, candidate_score in zip(
                    generated_sequences, generated_scores
                ):
                    if sequence in excluded_sequences:
                        continue
                    record = (sequence, float(candidate_score), incumbent)
                    previous = candidate_pool.get(sequence)
                    if previous is None or record[1] > previous[1]:
                        candidate_pool[sequence] = record

            selected = sorted(
                candidate_pool.values(),
                key=lambda record: record[1],
                reverse=True,
            )[: config.query_budget]
            sequences = [record[0] for record in selected]
            marginal_mutation_scores = [record[1] for record in selected]
            model.trace_event(
                "candidate_pool_summary",
                planned_batches=planned_batches,
                actual_batches=generation_batch,
                generated_candidates=(generation_batch * config.candidate_batch_size),
                unique_eligible_candidates=len(candidate_pool),
            )
            for rank, record in enumerate(selected, start=1):
                model.trace_event(
                    "candidate_selected_for_query",
                    selected_rank=rank,
                    sequence=record[0],
                    mms_score=record[1],
                )

            fitnesses = oracle.get_fitness(np.asarray(sequences)).tolist()
            for rank, (sequence, fitness) in enumerate(
                zip(sequences, fitnesses), start=1
            ):
                model.trace_event(
                    "oracle_query",
                    optimization_round=round_index,
                    query_rank=rank,
                    sequence=sequence,
                    oracle_score=float(fitness),
                )
            model.update_mask_position_rewards(
                campaign.incumbent_sequence, sequences, fitnesses
            )
            replay_sequences.extend(sequences)
            replay_fitnesses.extend(float(fitness) for fitness in fitnesses)
            tracker.calculate_top_n_metrics((sequences, fitnesses), round_index, n=100)
            tracker.exp_results[round_index]["Optimization"] = deepcopy(metadata)
            tracker.exp_results[round_index]["Optimization"]["start_sequences"] = [
                campaign.incumbent_sequence
            ]
            tracker.exp_results[round_index]["Optimization"]["selection"] = {
                "generated_candidates": (
                    generation_batch * config.candidate_batch_size
                ),
                "unique_eligible_candidates": len(candidate_pool),
                "selected_mms": marginal_mutation_scores,
            }
            tracker.save_results(result_path)
            campaign.observe(sequences, fitnesses)

            if model.adaptation_enabled and round_index < config.rounds:
                adaptation_start = time.perf_counter()
                metrics = model.adapt(
                    replay_sequences,
                    replay_fitnesses,
                    adaptation_directory,
                    round_index,
                )
                tracker.exp_results[round_index][adaptation_result_label] = {
                    "seconds": float(time.perf_counter() - adaptation_start),
                    "n_sequences": len(replay_sequences),
                    "epochs": (
                        model.config.adaptation.epochs
                        if model.config.adaptation is not None
                        else 0
                    ),
                    "last_epoch": metrics[-1] if metrics else None,
                }
                tracker.save_results(result_path)
    finally:
        if trace_writer is not None:
            trace_writer.close()
            _write_trace_summary(trace_writer)

    completion_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "task": config.task,
                "seed": config.seed,
                "rounds": config.rounds,
                "queries_per_round": config.query_budget,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return result_path
