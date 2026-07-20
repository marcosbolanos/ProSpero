from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

from prospero.reproduction.types import (
    EpistasisStage,
    PlmOptimizationStage,
    ProSperoStage,
    ProteinLanguageModel,
    ReproductionContext,
    ScoringBenchmarkStage,
    Stage,
)


def csv_ints(values: Iterable[int]) -> str:
    return ",".join(str(value) for value in values)


def seed_done(run_root: Path, task: str, seed: int) -> bool:
    return (run_root / task / f"seed_{seed}.pkl").is_file() and (
        run_root / task / f"seed_{seed}.complete.json"
    ).is_file()


def prospero_commands(
    stage: ProSperoStage, context: ReproductionContext
) -> list[tuple[str, list[str]]]:
    commands = []
    for task in stage.tasks:
        out = context.results / "prospero" / f"{task}_cnn"
        commands.append(
            (
                f"prospero_{task}",
                [
                    sys.executable,
                    "-m",
                    "prospero.runners.run_variable_k",
                    str(out),
                    "--task",
                    task,
                    "--query-budgets",
                    csv_ints(stage.query_budgets),
                    "--seeds",
                    csv_ints(stage.seeds),
                    "--rounds",
                    str(stage.rounds),
                    "--max-workers",
                    str(stage.max_workers),
                    "--resume",
                ],
            )
        )
    return commands


def optimization_command(
    stage: PlmOptimizationStage,
    output_directory: Path,
    task: str,
    seed: int,
    query_budget: int,
) -> list[str]:
    module = (
        "prospero.runners.run_prosst_optimization"
        if stage.model is ProteinLanguageModel.PROSST
        else "prospero.runners.run_evodiff_optimization"
    )
    command = [
        sys.executable,
        "-m",
        module,
        "--task",
        task,
        "--results-directory",
        str(output_directory),
        "--seed",
        str(seed),
        "--query-budget",
        str(query_budget),
        "--rounds",
        str(stage.rounds),
        "--candidate-batch-size",
        str(stage.candidate_batch_size),
        "--mask-budget",
        str(stage.mask_budget),
        "--device",
        "cuda",
        "--write-traces",
        "--deterministic-algorithms",
        "--decoding-vocabulary",
        stage.decoding_vocabulary.value,
    ]
    if stage.model is ProteinLanguageModel.PROSST:
        command.extend(
            [
                "--structure-tokens-directory",
                stage.structure_tokens_directory,
            ]
        )
    return command


def optimization_root(
    stage: PlmOptimizationStage,
    context: ReproductionContext,
    query_budget: int,
) -> Path:
    return context.results / stage.name / f"k_{query_budget}"


def optimization_commands(
    stage: PlmOptimizationStage, context: ReproductionContext
) -> list[tuple[str, list[str]]]:
    commands = []
    for budget in stage.query_budgets:
        out = optimization_root(stage, context, budget)
        for task in stage.tasks:
            for seed in stage.seeds:
                if context.skip_existing and seed_done(out, task, seed):
                    continue
                cmd = optimization_command(stage, out, task, seed, budget)
                if stage.adaptation is not None:
                    cmd.extend(
                        [
                            "--online-adaptation",
                            "--adaptation-epochs",
                            str(stage.adaptation.epochs),
                            "--adaptation-learning-rate",
                            str(stage.adaptation.learning_rate),
                            "--kl-coefficient",
                            str(stage.adaptation.kl_coefficient),
                            "--adaptation-batch-size",
                            str(stage.adaptation.batch_size),
                        ]
                    )
                commands.append((f"{stage.name}_k{budget}_{task}_seed{seed}", cmd))
    return commands


def scoring_benchmark_commands(
    stage: ScoringBenchmarkStage, context: ReproductionContext
) -> list[tuple[str, list[str]]]:
    return [
        (
            stage.name,
            [
                sys.executable,
                "-m",
                "prospero.runners.run_plm_scoring_benchmark",
                "--output-directory",
                str(context.results / "plm_scoring_benchmark"),
                "--tasks",
                *stage.tasks,
                "--models",
                *stage.models,
                "--max-sequences",
                str(stage.max_sequences),
                "--chunk-size",
                str(stage.chunk_size),
                "--seed",
                str(stage.seed),
                "--esm-model",
                stage.esm_model,
                "--prosst-model",
                stage.prosst_model,
                "--structure-tokens-directory",
                stage.structure_tokens_directory,
                "--device",
                "cuda",
            ],
        )
    ]


def epistasis_commands(
    stage: EpistasisStage, context: ReproductionContext
) -> list[tuple[str, list[str]]]:
    out = context.results / "epistasis"
    return [
        (
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
        )
    ]


def stage_commands(
    stage: Stage, context: ReproductionContext
) -> list[tuple[str, list[str]]]:
    if isinstance(stage, ProSperoStage):
        return prospero_commands(stage, context)
    if isinstance(stage, PlmOptimizationStage):
        return optimization_commands(stage, context)
    if isinstance(stage, EpistasisStage):
        return epistasis_commands(stage, context)
    if isinstance(stage, ScoringBenchmarkStage):
        return scoring_benchmark_commands(stage, context)
    raise TypeError(f"Unsupported stage type: {type(stage)!r}")


def plot_commands(
    stages: list[Stage], context: ReproductionContext
) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    stage_names = {stage.name for stage in stages}
    tasks = tuple(
        dict.fromkeys(task for stage in stages for task in getattr(stage, "tasks", ()))
    )
    budgets = sorted(
        {budget for stage in stages for budget in getattr(stage, "query_budgets", ())}
    )

    optimization_by_name = {
        stage.name: stage for stage in stages if isinstance(stage, PlmOptimizationStage)
    }
    main_prosst = optimization_by_name.get("prosst_online_adaptation")
    main_evodiff = optimization_by_name.get("evodiff_online_adaptation")

    main_plot_budgets_available = (
        main_prosst is not None
        and {8, 128}.issubset(main_prosst.query_budgets)
        and (main_evodiff is None or {8, 128}.issubset(main_evodiff.query_budgets))
    )
    if main_plot_budgets_available and "prospero_cnn" in stage_names:
        assert main_prosst is not None
        main_plot_command = [
            sys.executable,
            "-m",
            "prospero.runners.plot_combined_budget_mean_max",
            "--output-dir",
            str(context.plots / "main_optimization"),
            "--prosst-k8-root",
            str(optimization_root(main_prosst, context, 8)),
            "--prosst-k128-root",
            str(optimization_root(main_prosst, context, 128)),
            "--prospero-results-dir",
            str(context.results / "prospero"),
        ]
        if main_evodiff is not None:
            main_plot_command.extend(
                [
                    "--evodiff-k8-root",
                    str(optimization_root(main_evodiff, context, 8)),
                    "--evodiff-k128-root",
                    str(optimization_root(main_evodiff, context, 128)),
                ]
            )
        commands.append(
            (
                "plot_main_optimization",
                main_plot_command,
            )
        )

    for stage in stages:
        if not isinstance(stage, PlmOptimizationStage):
            continue
        for budget in stage.query_budgets:
            run_dir = optimization_root(stage, context, budget)
            if context.dry_run or run_dir.exists():
                commands.append(
                    (
                        f"plot_hist_{stage.name}_k{budget}",
                        [
                            sys.executable,
                            "-m",
                            "prospero.runners.plot_round_fitness_histograms",
                            "--run-directory",
                            str(run_dir),
                            "--method-label",
                            f"{stage.plot_label}, K={budget}",
                            "--output-directory",
                            str(
                                context.plots
                                / "round_histograms"
                                / stage.model.value
                                / stage.name
                                / f"k{budget}"
                            ),
                            "--tasks",
                            *stage.tasks,
                            "--seeds",
                            str(stage.seeds[0]),
                        ],
                    )
                )

    epi_json = context.results / "epistasis" / "epistasis_additivity_all_tasks.json"
    if "epistasis_additivity" in stage_names and (context.dry_run or epi_json.exists()):
        commands.append(
            (
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
            )
        )

    if {
        "prosst_online_adaptation",
        "prosst_unrestricted_vocabulary",
    }.issubset(stage_names):
        for budget in budgets:
            commands.append(
                (
                    f"plot_vocab_ablation_k{budget}",
                    [
                        sys.executable,
                        "-m",
                        "prospero.runners.plot_vocabulary_ablation",
                        "--output-dir",
                        str(context.plots / f"vocab_ablation_k{budget}"),
                        "--restricted-root",
                        str(
                            optimization_root(
                                optimization_by_name["prosst_online_adaptation"],
                                context,
                                budget,
                            )
                        ),
                        "--unrestricted-root",
                        str(
                            optimization_root(
                                optimization_by_name["prosst_unrestricted_vocabulary"],
                                context,
                                budget,
                            )
                        ),
                        "--budget",
                        str(budget),
                        "--tasks",
                        *(tasks or ("AAV", "LGK")),
                    ],
                )
            )

    commands.append(
        (
            "summarize_paper_results",
            [
                sys.executable,
                "-m",
                "prospero.runners.summarize_reproduction",
                "--results-root",
                str(context.results),
                "--output-dir",
                str(context.root / "tables"),
                "--tasks",
                *(
                    tasks
                    or ("AAV", "LGK", "GFP", "Pab1", "AMIE", "E4B", "TEM", "UBE2I")
                ),
                "--budgets",
                *(str(budget) for budget in (budgets or [8, 128])),
            ],
        )
    )

    return commands
