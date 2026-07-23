from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from prospero.optimization.types import (
    DecodingVocabulary,
    OnlineAdaptationConfig,
)


class ProteinLanguageModel(str, Enum):
    PROSST = "prosst"
    EVODIFF = "evodiff"


@dataclass(frozen=True)
class ProSperoStage:
    name: str = "prospero_cnn"
    tasks: tuple[str, ...] = ()
    query_budgets: tuple[int, ...] = (8, 128)
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    rounds: int = 10
    max_workers: int = 5


@dataclass(frozen=True)
class PlmOptimizationStage:
    name: str
    tasks: tuple[str, ...]
    model: ProteinLanguageModel = ProteinLanguageModel.PROSST
    plot_label: str = "0shotProt (w/ ProSST)"
    query_budgets: tuple[int, ...] = (8, 128)
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    adaptation: OnlineAdaptationConfig | None = field(
        default_factory=OnlineAdaptationConfig
    )
    decoding_vocabulary: DecodingVocabulary = DecodingVocabulary.RESTRICTED
    mask_budget: int = 4
    rounds: int = 10
    candidate_batch_size: int = 64
    structure_tokens_directory: str = "assets/prosst_structure_tokens"


@dataclass(frozen=True)
class ScoringBenchmarkStage:
    name: str = "plm_scoring_benchmark"
    tasks: tuple[str, ...] = ()
    models: tuple[str, ...] = ("evodiff", "esm", "prosst")
    max_sequences: int = 128
    chunk_size: int = 4
    seed: int = 142857
    esm_model: str = "facebook/esm2_t33_650M_UR50D"
    prosst_model: str = "AI4Protein/ProSST-2048"
    structure_tokens_directory: str = "assets/prosst_structure_tokens"


@dataclass(frozen=True)
class EpistasisStage:
    name: str = "epistasis_additivity"
    tasks: tuple[str, ...] = ()
    seed: int = 1
    samples_per_pair_type: int = 250
    oracle_batch_size: int = 128


Stage = ProSperoStage | PlmOptimizationStage | ScoringBenchmarkStage | EpistasisStage


@dataclass(frozen=True)
class ReproductionRecipe:
    stages: tuple[Stage, ...]
    plot_after_runs: bool = True


@dataclass
class RuntimeOptions:
    output_root: Path
    timestamp: str | None = None
    gpu: str | None = None
    dry_run: bool = False
    plots_only: bool = False
    no_plots: bool = False
    skip_existing: bool = True
    selected_stages: set[str] | None = None
    task_filter: tuple[str, ...] | None = None
    seed_filter: tuple[int, ...] | None = None
    budget_filter: tuple[int, ...] | None = None
    extra_manifest: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.plots_only and self.no_plots:
            raise ValueError("plots_only and no_plots cannot both be enabled.")


@dataclass(frozen=True)
class ReproductionContext:
    repo_root: Path
    root: Path
    results: Path
    plots: Path
    logs: Path
    env: dict[str, str]
    dry_run: bool
    skip_existing: bool
