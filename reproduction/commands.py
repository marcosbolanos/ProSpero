from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

from prospero.reproduction.types import EpistasisStage, ProSperoStage, ReproductionContext, Stage, ZeroShotStage


def csv_ints(values: Iterable[int]) -> str:
    return ",".join(str(value) for value in values)


def seed_done(run_root: Path, task: str, seed: int) -> bool:
    return (run_root / task / f"seed_{seed}.pkl").is_file()


def prospero_commands(stage: ProSperoStage, context: ReproductionContext) -> list[tuple[str, list[str]]]:
    commands = []
    for task in stage.tasks:
        out = context.results / "prospero" / f"{task}_cnn"
        commands.append((
            f"prospero_{task}",
            [
                sys.executable,
                "-m",
                "prospero.runners.run_variable_k",
                str(out),
                "--task",
                task,
                "--surrogate-arch",
                stage.surrogate_arch,
                "--n-samples",
                csv_ints(stage.budgets),
                "--seeds",
                csv_ints(stage.seeds),
                "--n-iters",
                str(stage.n_iters),
                "--max-workers",
                str(stage.max_workers),
                "--safe",
            ],
        ))
    return commands


def zero_shot_base(stage: ZeroShotStage, out: Path, task: str, seed: int, budget: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "prospero.runners.run_zero_shot_prosst",
        "--task",
        task,
        "--results_dirpath",
        str(out),
        "--seed",
        str(seed),
        "--n_queries",
        str(budget),
        "--n_iters",
        str(stage.n_iters),
        "--batch_size",
        str(stage.batch_size),
        "--mask_budget",
        str(stage.mask_budget),
        "--mask_strategy",
        stage.mask_strategy,
        "--structure_tokens_dir",
        stage.structure_tokens_dir,
        "--device",
        "cuda",
        "--debug_generation_trace",
        "--smc_vocab",
        stage.smc_vocab,
    ]


def zero_shot_commands(stage: ZeroShotStage, context: ReproductionContext) -> list[tuple[str, list[str]]]:
    commands = []
    for budget in stage.budgets:
        out = context.results / f"0shotprot_prosst_{stage.name}_n{budget}"
        for task in stage.tasks:
            for seed in stage.seeds:
                if context.skip_existing and seed_done(out, task, seed):
                    continue
                cmd = zero_shot_base(stage, out, task, seed, budget)
                if stage.finetune:
                    cmd.extend([
                        "--finetune_prosst",
                        "--finetune_epochs",
                        str(stage.finetune_epochs),
                        "--finetune_lr",
                        str(stage.finetune_lr),
                        "--lambda_kl",
                        str(stage.lambda_kl),
                        "--finetune_batch_size",
                        str(stage.finetune_batch_size),
                        "--finetune_replay",
                        "all",
                        "--reward_mode",
                        stage.loss,
                    ])
                commands.append((f"{stage.name}_n{budget}_{task}_seed{seed}", cmd))
    return commands


def epistasis_commands(stage: EpistasisStage, context: ReproductionContext) -> list[tuple[str, list[str]]]:
    out = context.results / "epistasis"
    return [(
        stage.name,
        [
            sys.executable,
            "-m",
            "prospero.runners.run_epistasis_additivity_test",
            "--tasks",
            *stage.tasks,
            "--seed",
            str(stage.seed),
            "--samples-per-pair-type",
            str(stage.samples_per_pair_type),
            "--oracle-batch-size",
            str(stage.oracle_batch_size),
            "--output-dir",
            str(out),
        ],
    )]


def stage_commands(stage: Stage, context: ReproductionContext) -> list[tuple[str, list[str]]]:
    if isinstance(stage, ProSperoStage):
        return prospero_commands(stage, context)
    if isinstance(stage, ZeroShotStage):
        return zero_shot_commands(stage, context)
    if isinstance(stage, EpistasisStage):
        return epistasis_commands(stage, context)
    raise TypeError(f"Unsupported stage type: {type(stage)!r}")


def plot_commands(stages: list[Stage], context: ReproductionContext) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    stage_names = {stage.name for stage in stages}
    tasks = tuple(dict.fromkeys(task for stage in stages for task in getattr(stage, "tasks", ())))
    budgets = sorted({budget for stage in stages for budget in getattr(stage, "budgets", ())})

    main_zero_shot_stage = None
    if "grpo_cluster" in stage_names:
        main_zero_shot_stage = "grpo_cluster"
    elif "rank_cluster" in stage_names:
        main_zero_shot_stage = "rank_cluster"

    if main_zero_shot_stage is not None and "prospero_cnn_variable_k" in stage_names:
        commands.append((
            f"plot_main_{main_zero_shot_stage}",
            [
                sys.executable,
                "-m",
                "prospero.runners.plot_simplified_zero_shotprot_mean_max",
                "--output-dir",
                str(context.plots / f"main_{main_zero_shot_stage}"),
                "--prosst-k8-root",
                str(context.results / f"0shotprot_prosst_{main_zero_shot_stage}_n8"),
                "--prosst-k128-root",
                str(context.results / f"0shotprot_prosst_{main_zero_shot_stage}_n128"),
                "--prospero-results-dir",
                str(context.results / "prospero"),
                "--no-evodiff",
            ],
        ))

    for stage in stages:
        if not isinstance(stage, ZeroShotStage):
            continue
        for budget in stage.budgets:
            run_dir = context.results / f"0shotprot_prosst_{stage.name}_n{budget}"
            if context.dry_run or run_dir.exists():
                commands.append((
                    f"plot_hist_{stage.name}_n{budget}",
                    [
                        sys.executable,
                        "-m",
                        "prospero.runners.plot_zero_shot_round_fitness_histograms",
                        "--run_dir",
                        str(run_dir),
                        "--method_label",
                        f"0shotProt ProSST {stage.name} K={budget}",
                        "--output_dir",
                        str(context.plots / "round_histograms" / stage.name / f"k{budget}"),
                        "--tasks",
                        *stage.tasks,
                    ],
                ))

    epi_json = context.results / "epistasis" / "epistasis_additivity_all_tasks.json"
    if "epistasis_additivity" in stage_names and (context.dry_run or epi_json.exists()):
        commands.append((
            "plot_epistasis",
            [
                sys.executable,
                "-m",
                "prospero.runners.plot_epistasis_additivity_styled",
                "--source",
                str(epi_json),
                "--output-dir",
                str(context.plots / "epistasis"),
            ],
        ))

    if {"grpo_cluster", "grpo_unrestricted_vocab"}.issubset(stage_names) and 128 in budgets:
        commands.append((
            "plot_vocab_ablation_k128",
            [
                sys.executable,
                "-m",
                "prospero.runners.plot_prosst_vocab_ablation_k128",
                "--output-dir",
                str(context.plots / "vocab_ablation_k128"),
                "--restricted-root",
                str(context.results / "0shotprot_prosst_grpo_cluster_n128"),
                "--unrestricted-root",
                str(context.results / "0shotprot_prosst_grpo_unrestricted_vocab_n128"),
                "--tasks",
                *(tasks or ("AAV", "LGK")),
            ],
        ))

    return commands
