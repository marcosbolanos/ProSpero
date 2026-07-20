from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ProSperoStage:
    name: str = "prospero_cnn_variable_k"
    tasks: tuple[str, ...] = ()
    budgets: tuple[int, ...] = (8, 128)
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    surrogate_arch: str = "cnn"
    n_iters: int = 10
    max_workers: int = 5


@dataclass(frozen=True)
class ZeroShotStage:
    name: str
    tasks: tuple[str, ...]
    plm: str = "prosst"
    plot_label: str = "0shotProt (w/ ProSST)"
    budgets: tuple[int, ...] = (8, 128)
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    finetune: bool = True
    decoding_vocab: str = "restricted"
    mask_budget: int = 4
    n_iters: int = 10
    batch_size: int = 64
    finetune_epochs: int = 5
    finetune_lr: float = 3e-5
    lambda_kl: float = 2.0
    finetune_batch_size: int = 1
    structure_tokens_dir: str = "outputs/prosst_structure_tokens"


@dataclass(frozen=True)
class AlignmentStage:
    name: str = "plm_mms_pll_alignment"
    tasks: tuple[str, ...] = ()
    plms: tuple[str, ...] = ("evodiff", "esm", "prosst")
    max_sequences: int = 128
    chunk_size: int = 4
    seed: int = 142857
    esm_model: str = "facebook/esm2_t33_650M_UR50D"
    prosst_model: str = "AI4Protein/ProSST-2048"
    structure_tokens_dir: str = "outputs/prosst_structure_tokens"


@dataclass(frozen=True)
class EpistasisStage:
    name: str = "epistasis_additivity"
    tasks: tuple[str, ...] = ()
    seed: int = 1
    samples_per_pair_type: int = 250
    oracle_batch_size: int = 128


Stage = ProSperoStage | ZeroShotStage | AlignmentStage | EpistasisStage


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
