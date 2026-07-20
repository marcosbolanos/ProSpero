from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from prospero.reproduction.commands import plot_commands, stage_commands
from prospero.reproduction.types import (
    ScoringBenchmarkStage,
    EpistasisStage,
    ProSperoStage,
    ReproductionContext,
    ReproductionRecipe,
    RuntimeOptions,
    Stage,
    PlmOptimizationStage,
)


class CommandRunner:
    def __init__(self, context: ReproductionContext) -> None:
        self.context = context
        self.commands: list[dict[str, object]] = []

    def run(self, name: str, cmd: list[str]) -> None:
        log_path = self.context.logs / f"{len(self.commands):04d}_{name}.log"
        self.commands.append({"name": name, "cmd": cmd, "log": str(log_path)})
        print("[reproduce]", name)
        print(" ".join(cmd))
        if self.context.dry_run:
            return
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(cmd) + "\n\n")
            log.flush()
            subprocess.run(
                cmd,
                cwd=self.context.repo_root,
                env=self.context.env,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )

    def write_manifest(
        self, recipe: ReproductionRecipe, options: RuntimeOptions, stages: list[Stage]
    ) -> None:
        manifest = self.context.root / "manifest.json"
        payload = {
            "recipe": _json_safe(asdict(recipe)),
            "selected_stages": [_json_safe(asdict(stage)) for stage in stages],
            "runtime_options": _json_safe(asdict(options)),
            "commands": self.commands,
        }
        manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[reproduce] manifest: {manifest}")


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


def _filter_tuple(values: tuple, allowed: tuple | None) -> tuple:
    if allowed is None:
        return values
    allowed_set = set(allowed)
    return tuple(value for value in values if value in allowed_set)


def apply_filters(stage: Stage, options: RuntimeOptions) -> Stage | None:
    if (
        options.selected_stages is not None
        and stage.name not in options.selected_stages
    ):
        return None
    if isinstance(stage, PlmOptimizationStage):
        filtered = replace(
            stage,
            tasks=_filter_tuple(stage.tasks, options.task_filter),
            seeds=_filter_tuple(stage.seeds, options.seed_filter),
            query_budgets=_filter_tuple(stage.query_budgets, options.budget_filter),
        )
        return (
            filtered
            if filtered.tasks and filtered.seeds and filtered.query_budgets
            else None
        )
    if isinstance(stage, ProSperoStage):
        filtered = replace(
            stage,
            tasks=_filter_tuple(stage.tasks, options.task_filter),
            seeds=_filter_tuple(stage.seeds, options.seed_filter),
            query_budgets=_filter_tuple(stage.query_budgets, options.budget_filter),
        )
        return (
            filtered
            if filtered.tasks and filtered.seeds and filtered.query_budgets
            else None
        )
    if isinstance(stage, EpistasisStage):
        filtered = replace(stage, tasks=_filter_tuple(stage.tasks, options.task_filter))
        return filtered if filtered.tasks else None
    if isinstance(stage, ScoringBenchmarkStage):
        filtered = replace(stage, tasks=_filter_tuple(stage.tasks, options.task_filter))
        return filtered if filtered.tasks else None
    raise TypeError(type(stage))


def make_context(options: RuntimeOptions) -> ReproductionContext:
    repo_root = Path.cwd()
    timestamp = options.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = repo_root / options.output_root / timestamp
    results = root / "results"
    plots = root / "plots"
    logs = root / "logs"
    for path in (root, results, plots, logs):
        path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if options.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = options.gpu
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return ReproductionContext(
        repo_root=repo_root,
        root=root,
        results=results,
        plots=plots,
        logs=logs,
        env=env,
        dry_run=options.dry_run,
        skip_existing=options.skip_existing,
    )


def print_plan(
    context: ReproductionContext, stages: list[Stage], options: RuntimeOptions
) -> None:
    print(f"[reproduce] root: {context.root}")
    print("[reproduce] stages:")
    for idx, stage in enumerate(stages, start=1):
        if isinstance(stage, ProSperoStage):
            print(
                f"  {idx}. {stage.name}: ProSpero CNN, "
                f"query_budgets={stage.query_budgets}, seeds={stage.seeds}, "
                f"tasks={stage.tasks}"
            )
        elif isinstance(stage, PlmOptimizationStage):
            if stage.adaptation is None:
                adaptation = "no online adaptation"
            else:
                adaptation = (
                    f"adaptation={stage.adaptation.epochs} epochs, "
                    f"lr={stage.adaptation.learning_rate:g}, "
                    f"KL={stage.adaptation.kl_coefficient:g}"
                )
            print(
                f"  {idx}. {stage.name}: {stage.plot_label}, "
                f"model={stage.model.value}, "
                f"vocabulary={stage.decoding_vocabulary.value}, "
                f"mask_budget={stage.mask_budget}, {adaptation}, "
                f"query_budgets={stage.query_budgets}, "
                f"seeds={stage.seeds}, tasks={stage.tasks}"
            )
        elif isinstance(stage, EpistasisStage):
            print(
                f"  {idx}. {stage.name}: samples_per_pair_type={stage.samples_per_pair_type}, tasks={stage.tasks}"
            )
        elif isinstance(stage, ScoringBenchmarkStage):
            print(
                f"  {idx}. {stage.name}: models={stage.models}, "
                f"n={stage.max_sequences}, "
                f"tasks={stage.tasks}"
            )
    if options.plots_only:
        print("[reproduce] plots-only: experiment commands will be skipped")
    if options.no_plots:
        print("[reproduce] no-plots: plot commands will be skipped")
    if options.dry_run:
        print("[reproduce] dry-run: commands will be printed but not executed")


def run_reproduction(
    recipe: ReproductionRecipe, options: RuntimeOptions
) -> ReproductionContext:
    context = make_context(options)
    stages = [
        stage
        for stage in (apply_filters(stage, options) for stage in recipe.stages)
        if stage is not None
    ]
    print_plan(context, stages, options)
    runner = CommandRunner(context)

    config_path = context.root / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "runtime_options": _json_safe(asdict(options)),
                "stages": [_json_safe(asdict(s)) for s in stages],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not options.plots_only:
        for stage in stages:
            for name, cmd in stage_commands(stage, context):
                runner.run(name, cmd)

    if recipe.plot_after_runs and not options.no_plots:
        for name, cmd in plot_commands(stages, context):
            runner.run(name, cmd)

    runner.write_manifest(recipe, options, stages)
    return context
