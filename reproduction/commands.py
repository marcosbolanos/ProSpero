from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

from prospero.reproduction.types import AlignmentStage, EpistasisStage, ProSperoStage, ReproductionContext, Stage, ZeroShotStage


def csv_ints(values: Iterable[int]) -> str:
    return ",".join(str(value) for value in values)


def seed_done(run_root: Path, task: str, seed: int) -> bool:
    return (
        (run_root / task / f"seed_{seed}.pkl").is_file()
        and (run_root / task / f"seed_{seed}.complete.json").is_file()
    )


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
                "--resume-missing-seeds",
            ],
        ))
    return commands


def zero_shot_base(stage: ZeroShotStage, out: Path, task: str, seed: int, budget: int) -> list[str]:
    if stage.plm not in {"prosst", "evodiff"}:
        raise ValueError(f"Unsupported optimization PLM: {stage.plm}")
    module = (
        "prospero.runners.run_zero_shot_prosst"
        if stage.plm == "prosst"
        else "prospero.runners.run_zero_shot_evodiff"
    )
    command = [
        sys.executable,
        "-m",
        module,
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
        "--device",
        "cuda",
        "--debug_generation_trace",
        "--full_deterministic",
        "--decoding_vocab",
        stage.decoding_vocab,
    ]
    if stage.plm == "prosst":
        command.extend(["--structure_tokens_dir", stage.structure_tokens_dir])
    return command


def zero_shot_root(stage: ZeroShotStage, context: ReproductionContext, budget: int) -> Path:
    return context.results / f"0shotprot_{stage.plm}_{stage.name}_n{budget}"


def zero_shot_commands(stage: ZeroShotStage, context: ReproductionContext) -> list[tuple[str, list[str]]]:
    commands = []
    for budget in stage.budgets:
        out = zero_shot_root(stage, context, budget)
        for task in stage.tasks:
            for seed in stage.seeds:
                if context.skip_existing and seed_done(out, task, seed):
                    continue
                cmd = zero_shot_base(stage, out, task, seed, budget)
                if stage.finetune:
                    cmd.extend([
                        f"--finetune_{stage.plm}",
                        "--finetune_epochs",
                        str(stage.finetune_epochs),
                        "--finetune_lr",
                        str(stage.finetune_lr),
                        "--lambda_kl",
                        str(stage.lambda_kl),
                        "--finetune_batch_size",
                        str(stage.finetune_batch_size),
                    ])
                commands.append((f"{stage.name}_n{budget}_{task}_seed{seed}", cmd))
    return commands


def alignment_commands(stage: AlignmentStage, context: ReproductionContext) -> list[tuple[str, list[str]]]:
    return [(
        stage.name,
        [
            sys.executable,
            "-m",
            "prospero.runners.run_plm_full_pll_alignment",
            "--out_dir",
            str(context.results / "plm_mms_pll_alignment"),
            "--tasks",
            *stage.tasks,
            "--plms",
            *stage.plms,
            "--max_sequences",
            str(stage.max_sequences),
            "--chunk_size",
            str(stage.chunk_size),
            "--seed",
            str(stage.seed),
            "--esm_model",
            stage.esm_model,
            "--prosst_model",
            stage.prosst_model,
            "--structure_tokens_dir",
            stage.structure_tokens_dir,
            "--device",
            "cuda",
        ],
    )]


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
    if isinstance(stage, AlignmentStage):
        return alignment_commands(stage, context)
    raise TypeError(f"Unsupported stage type: {type(stage)!r}")


def plot_commands(stages: list[Stage], context: ReproductionContext) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    stage_names = {stage.name for stage in stages}
    tasks = tuple(dict.fromkeys(task for stage in stages for task in getattr(stage, "tasks", ())))
    budgets = sorted({budget for stage in stages for budget in getattr(stage, "budgets", ())})

    zero_shot_by_name = {
        stage.name: stage for stage in stages if isinstance(stage, ZeroShotStage)
    }
    main_prosst = zero_shot_by_name.get("prosst_finetuned")
    main_evodiff = zero_shot_by_name.get("evodiff_finetuned")

    if main_prosst is not None and "prospero_cnn_variable_k" in stage_names:
        main_plot_command = [
            sys.executable,
            "-m",
            "prospero.runners.plot_combined_budget_mean_max",
            "--output-dir",
            str(context.plots / "main_optimization"),
            "--prosst-k8-root",
            str(zero_shot_root(main_prosst, context, 8)),
            "--prosst-k128-root",
            str(zero_shot_root(main_prosst, context, 128)),
            "--prospero-results-dir",
            str(context.results / "prospero"),
        ]
        if main_evodiff is not None:
            main_plot_command.extend([
                "--evodiff-k8-root",
                str(zero_shot_root(main_evodiff, context, 8)),
                "--evodiff-k128-root",
                str(zero_shot_root(main_evodiff, context, 128)),
            ])
        else:
            main_plot_command.append("--no-evodiff")
        commands.append((
            "plot_main_optimization",
            main_plot_command,
        ))

    for stage in stages:
        if not isinstance(stage, ZeroShotStage):
            continue
        for budget in stage.budgets:
            run_dir = zero_shot_root(stage, context, budget)
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
                        f"{stage.plot_label}, K={budget}",
                        "--output_dir",
                    str(context.plots / "round_histograms" / stage.plm / stage.name / f"k{budget}"),
                        "--tasks",
                        *stage.tasks,
                        "--seeds",
                        str(stage.seeds[0]),
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

    if {"prosst_finetuned", "prosst_finetuned_unrestricted"}.issubset(stage_names):
        for budget in budgets:
            commands.append((
                f"plot_vocab_ablation_k{budget}",
                [
                    sys.executable,
                    "-m",
                    "prospero.runners.plot_prosst_vocab_ablation_k128",
                    "--output-dir",
                    str(context.plots / f"vocab_ablation_k{budget}"),
                    "--restricted-root",
                    str(zero_shot_root(zero_shot_by_name["prosst_finetuned"], context, budget)),
                    "--unrestricted-root",
                    str(zero_shot_root(zero_shot_by_name["prosst_finetuned_unrestricted"], context, budget)),
                    "--budget",
                    str(budget),
                    "--tasks",
                    *(tasks or ("AAV", "LGK")),
                ],
            ))

    commands.append((
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
            *(tasks or ("AAV", "LGK", "GFP", "Pab1", "AMIE", "E4B", "TEM", "UBE2I")),
            "--budgets",
            *(str(budget) for budget in (budgets or [8, 128])),
        ],
    ))

    return commands
