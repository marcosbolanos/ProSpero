#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from prospero.experiments_config import ALPHABETS
from prospero.optimization.campaign import run_optimization_campaign
from prospero.optimization.prosst import (
    STRUCTURE_TOKENS_DIR,
    SUPPORTED_TASKS,
    ProSSTModel,
)
from prospero.optimization.types import (
    CampaignConfig,
    DecodingVocabulary,
    MaskingConfig,
    OnlineAdaptationConfig,
    ProSSTConfig,
)


LOGGER = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PLM-guided protein optimization with ProSST."
    )
    parser.add_argument(
        "--results-directory",
        required=True,
    )
    parser.add_argument("--task", default="AAV", choices=SUPPORTED_TASKS)
    parser.add_argument("--seed", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--query-budget", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=64,
    )
    parser.add_argument("--mask-budget", type=int, default=4)
    parser.add_argument(
        "--entropy-quantile", dest="entropy_quantile", type=float, default=0.5
    )
    parser.add_argument("--entropy-sigma", dest="entropy_sigma", type=float)
    parser.add_argument(
        "--entropy-chunk-size", dest="entropy_chunk_size", type=int, default=16
    )
    parser.add_argument(
        "--seed-grow-alpha", dest="seed_grow_alpha", type=float, default=1.0
    )
    parser.add_argument(
        "--seed-grow-beta", dest="seed_grow_beta", type=float, default=1.0
    )
    parser.add_argument(
        "--seed-grow-coupling-tau",
        dest="seed_grow_coupling_tau",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--substitution-alphabet",
        dest="substitution_alphabet",
        default="CHARGE",
        choices=list(ALPHABETS),
    )
    parser.add_argument(
        "--decoding-vocabulary",
        dest="decoding_vocabulary",
        choices=["restricted", "unrestricted"],
        default="restricted",
    )
    parser.add_argument(
        "--model-name-or-path",
        dest="model_name_or_path",
        default="AI4Protein/ProSST-2048",
    )
    parser.add_argument(
        "--structure-tokens-directory",
        dest="structure_tokens_directory",
        default=str(STRUCTURE_TOKENS_DIR),
    )
    parser.add_argument(
        "--structure-vocabulary-size", dest="structure_vocabulary_size", default="2048"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--write-traces",
        dest="write_traces",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--online-adaptation", dest="online_adaptation", action="store_true"
    )
    parser.add_argument(
        "--adaptation-epochs", dest="adaptation_epochs", type=int, default=5
    )
    parser.add_argument(
        "--adaptation-learning-rate",
        dest="adaptation_learning_rate",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--kl-coefficient", dest="kl_coefficient", type=float, default=2.0
    )
    parser.add_argument(
        "--adaptation-batch-size", dest="adaptation_batch_size", type=int, default=16
    )
    parser.add_argument(
        "--maximum-absolute-advantage",
        dest="maximum_absolute_advantage",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--negative-advantage-scale",
        dest="negative_advantage_scale",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--deterministic-algorithms",
        dest="deterministic_algorithms",
        action="store_true",
    )
    return parser


def build_configs(
    args: argparse.Namespace,
) -> tuple[CampaignConfig, ProSSTConfig]:
    adaptation = (
        OnlineAdaptationConfig(
            epochs=args.adaptation_epochs,
            learning_rate=args.adaptation_learning_rate,
            kl_coefficient=args.kl_coefficient,
            batch_size=args.adaptation_batch_size,
            maximum_absolute_advantage=args.maximum_absolute_advantage,
            negative_advantage_scale=args.negative_advantage_scale,
        )
        if args.online_adaptation
        else None
    )
    masking = MaskingConfig(
        budget=args.mask_budget,
        entropy_quantile=args.entropy_quantile,
        entropy_bandwidth=args.entropy_sigma,
        entropy_chunk_size=args.entropy_chunk_size,
        seed_grow_alpha=args.seed_grow_alpha,
        seed_grow_beta=args.seed_grow_beta,
        seed_grow_coupling_length=args.seed_grow_coupling_tau,
    )
    campaign = CampaignConfig(
        output_directory=Path(args.results_directory),
        task=args.task,
        seed=args.seed,
        query_budget=args.query_budget,
        rounds=args.rounds,
        candidate_batch_size=args.candidate_batch_size,
        write_traces=args.write_traces,
        deterministic_algorithms=args.deterministic_algorithms,
    )
    model = ProSSTConfig(
        task=args.task,
        device=args.device,
        masking=masking,
        decoding_vocabulary=DecodingVocabulary(args.decoding_vocabulary),
        substitution_alphabet=args.substitution_alphabet,
        adaptation=adaptation,
        model_name_or_path=args.model_name_or_path,
        structure_tokens_directory=Path(args.structure_tokens_directory),
        structure_vocabulary_size=args.structure_vocabulary_size,
    )
    return campaign, model


def run_seed(args: argparse.Namespace) -> Path:
    campaign_config, model_config = build_configs(args)
    model = ProSSTModel(model_config)
    metadata = {
        "task": args.task,
        "model": args.model_name_or_path,
        "structure": {
            "source": "ProSST precomputed structure tokens",
            "structure_token_name": model.mapping.structure_token_name,
            "offset": model.mapping.offset,
            "covered_length": model.covered_length,
            "identity": model.mapping.identity,
            "note": model.mapping.note,
        },
        "generation": {
            "mode": "sequential_masked_decoding",
            "mask_strategy": "mixed_position_entropy_anticollapse",
            "mask_budget": args.mask_budget,
            "decoding_vocabulary": args.decoding_vocabulary,
            "score": "MMS from one unmasked incumbent context",
            "substitution_alphabet": args.substitution_alphabet,
            "candidate_batch_size": args.candidate_batch_size,
        },
        "prosst_online_adaptation": {
            "enabled": model.adaptation_enabled,
            "objective": "signed_advantage_weighted_reconstruction_plus_kl",
            "training_corruption": "uniform random fixed-budget masks",
            "mask_budget": args.mask_budget,
            "epochs_per_round": args.adaptation_epochs,
            "learning_rate": args.adaptation_learning_rate,
            "kl_coefficient": args.kl_coefficient,
            "batch_size": args.adaptation_batch_size,
            "replay": "all_online_queries",
            "negative_advantage_scale": args.negative_advantage_scale,
            "maximum_absolute_advantage": args.maximum_absolute_advantage,
        },
    }
    return run_optimization_campaign(
        campaign_config,
        model,
        metadata,
        logger=LOGGER,
        artifact_prefix="prosst_optimization",
        adaptation_result_label="ProSST online adaptation",
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = get_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    result_path = run_seed(args)
    print(
        f"complete task={args.task} seed={args.seed} path={result_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
