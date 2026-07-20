from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class DecodingVocabulary(str, Enum):
    RESTRICTED = "restricted"
    UNRESTRICTED = "unrestricted"


@dataclass(frozen=True)
class MaskingConfig:
    budget: int = 4
    entropy_quantile: float = 0.5
    entropy_bandwidth: float | None = None
    entropy_chunk_size: int = 16
    seed_grow_alpha: float = 1.0
    seed_grow_beta: float = 1.0
    seed_grow_coupling_length: float = 4.0

    def __post_init__(self) -> None:
        if self.budget < 1:
            raise ValueError("Mask budget must be positive.")
        if not 0.0 <= self.entropy_quantile <= 1.0:
            raise ValueError("Entropy quantile must be between zero and one.")


@dataclass(frozen=True)
class OnlineAdaptationConfig:
    epochs: int = 5
    learning_rate: float = 3e-5
    kl_coefficient: float = 2.0
    batch_size: int = 1
    maximum_absolute_advantage: float = 2.0
    negative_advantage_scale: float = 0.25

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("Adaptation epochs and batch size must be positive.")
        if self.learning_rate <= 0.0 or self.kl_coefficient < 0.0:
            raise ValueError("Invalid adaptation learning rate or KL coefficient.")


@dataclass(frozen=True)
class ModelConfig:
    task: str
    device: str
    masking: MaskingConfig
    decoding_vocabulary: DecodingVocabulary
    substitution_alphabet: str
    adaptation: OnlineAdaptationConfig | None


@dataclass(frozen=True)
class ProSSTConfig(ModelConfig):
    model_name_or_path: str
    structure_tokens_directory: Path
    structure_vocabulary_size: str = "2048"


@dataclass(frozen=True)
class EvoDiffConfig(ModelConfig):
    pass


@dataclass(frozen=True)
class CampaignConfig:
    output_directory: Path
    task: str
    seed: int
    query_budget: int
    rounds: int
    candidate_batch_size: int
    write_traces: bool
    deterministic_algorithms: bool

    def __post_init__(self) -> None:
        if self.query_budget < 1 or self.rounds < 1:
            raise ValueError("Query budget and rounds must be positive.")
        if self.candidate_batch_size < 1:
            raise ValueError("Candidate batch size must be positive.")
