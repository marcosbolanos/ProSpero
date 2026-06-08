import argparse
import importlib
from dataclasses import dataclass

from prospero.dataset import RegressionDataset
from prospero.experiments_config import WT_SEQUENCES
from prospero.surrogate import CNN


@dataclass
class TaskFlopsBreakdown:
    task: str
    sequence_length: int
    model_params: int
    forward_flops_per_sequence: float
    train_flops_total: float
    inference_flops_total: float
    total_flops: float


def parse_tasks(raw_tasks):
    if not raw_tasks:
        return list(WT_SEQUENCES.keys())
    tasks = [task.strip() for task in raw_tasks.split(",") if task.strip()]
    invalid = [task for task in tasks if task not in WT_SEQUENCES]
    if invalid:
        raise ValueError(
            f"Unknown tasks: {invalid}. Available tasks: {list(WT_SEQUENCES.keys())}"
        )
    return tasks


def resolve_num_iters(task, default_iters, dshift_iters):
    if task.startswith("D_SHIFT"):
        return dshift_iters
    return default_iters


def profile_forward_flops_per_sequence(seq_length, profile_batch_size):
    try:
        module = importlib.import_module("deepspeed.profiling.flops_profiler")
        get_model_profile = module.get_model_profile
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "deepspeed is required for FLOPs profiling. Install it with `pip install deepspeed`."
        ) from exc

    model = CNN(num_input_channels=20, seq_length=seq_length)
    flops, _, params = get_model_profile(
        model=model,
        input_shape=(profile_batch_size, 20, seq_length),
        print_profile=False,
        detailed=False,
        warm_up=0,
        as_string=False,
    )
    return flops / profile_batch_size, params


def estimate_training_flops(
    dataset,
    n_iters,
    n_queries,
    ensemble_size,
    forward_flops_per_sequence,
    avg_train_epochs,
    avg_val_epochs,
    train_step_multiplier,
):
    train_per_iter = int(round(0.9 * n_queries))
    val_per_iter = n_queries - train_per_iter

    train_total = 0.0
    valid_total = 0.0

    for retrain_idx in range(n_iters):
        train_size = len(dataset.train) + retrain_idx * train_per_iter
        valid_size = len(dataset.valid) + retrain_idx * val_per_iter

        train_total += train_size * avg_train_epochs
        valid_total += valid_size * avg_val_epochs

    per_model = (
        train_total * forward_flops_per_sequence * train_step_multiplier
        + valid_total * forward_flops_per_sequence
    )
    return per_model * ensemble_size


def count_resampling_calls(max_corruptions, resampling_steps):
    if isinstance(resampling_steps, int):
        if resampling_steps <= 0:
            raise ValueError("resampling_steps must be >= 1")
        return max_corruptions // resampling_steps

    values = [int(step.strip()) for step in resampling_steps.split(",") if step.strip()]
    if not values:
        raise ValueError("resampling_steps list must not be empty")
    return len([step for step in values if 1 <= step <= max_corruptions])


def estimate_inference_flops(
    n_iters,
    ensemble_size,
    forward_flops_per_sequence,
    batch_size,
    n_checks_multiplier,
    max_corruptions,
    resampling_steps,
    expected_generation_loops,
):
    scan_samples = batch_size * n_checks_multiplier
    resampling_calls = count_resampling_calls(max_corruptions, resampling_steps)
    ucb_samples = resampling_calls * batch_size
    samples_per_generation_call = scan_samples + ucb_samples
    total_samples = n_iters * expected_generation_loops * samples_per_generation_call
    return total_samples * forward_flops_per_sequence * ensemble_size


def estimate_task_flops(task, args):
    dataset = RegressionDataset(task)
    seq_length = len(WT_SEQUENCES[task])
    n_iters = resolve_num_iters(task, args.n_iters, args.dshift_n_iters)
    forward_flops_per_sequence, model_params = profile_forward_flops_per_sequence(
        seq_length=seq_length,
        profile_batch_size=args.profile_batch_size,
    )

    train_flops_total = estimate_training_flops(
        dataset=dataset,
        n_iters=n_iters,
        n_queries=args.n_queries,
        ensemble_size=args.ensemble_size,
        forward_flops_per_sequence=forward_flops_per_sequence,
        avg_train_epochs=args.avg_train_epochs,
        avg_val_epochs=args.avg_val_epochs,
        train_step_multiplier=args.train_step_multiplier,
    )

    inference_flops_total = estimate_inference_flops(
        n_iters=n_iters,
        ensemble_size=args.ensemble_size,
        forward_flops_per_sequence=forward_flops_per_sequence,
        batch_size=args.batch_size,
        n_checks_multiplier=args.n_checks_multiplier,
        max_corruptions=args.max_corruptions,
        resampling_steps=args.resampling_steps,
        expected_generation_loops=args.expected_generation_loops,
    )

    total_flops = train_flops_total + inference_flops_total
    return TaskFlopsBreakdown(
        task=task,
        sequence_length=seq_length,
        model_params=model_params,
        forward_flops_per_sequence=forward_flops_per_sequence,
        train_flops_total=train_flops_total,
        inference_flops_total=inference_flops_total,
        total_flops=total_flops,
    )


def humanize_flops(value):
    units = ["FLOPs", "KFLOPs", "MFLOPs", "GFLOPs", "TFLOPs", "PFLOPs"]
    idx = 0
    while value >= 1000 and idx < len(units) - 1:
        value /= 1000.0
        idx += 1
    return f"{value:.3f} {units[idx]}"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Estimate ProSpero surrogate FLOPs with DeepSpeed profiler"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help="Comma-separated list of tasks. Defaults to all tasks from WT_SEQUENCES.",
    )
    parser.add_argument(
        "--n-iters", type=int, default=10, help="Iterations for non D_SHIFT tasks"
    )
    parser.add_argument(
        "--dshift-n-iters", type=int, default=4, help="Iterations for D_SHIFT tasks"
    )
    parser.add_argument("--n-queries", type=int, default=128)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256, help="SMC batch size")
    parser.add_argument("--n-checks-multiplier", type=int, default=16)
    parser.add_argument("--max-corruptions", type=int, default=10)
    parser.add_argument(
        "--resampling-steps",
        type=str,
        default="1",
        help="Either an int step interval (e.g. '1' or '2') or explicit comma-separated steps (e.g. '1,3,7').",
    )
    parser.add_argument(
        "--expected-generation-loops",
        type=float,
        default=1.0,
        help="Expected while-loop calls to generate_raa_from_alanine_scan per AL iteration.",
    )
    parser.add_argument(
        "--avg-train-epochs",
        type=float,
        default=11.0,
        help="Average train epochs per retrain pass (defaults to patience+1 style estimate).",
    )
    parser.add_argument(
        "--avg-val-epochs",
        type=float,
        default=11.0,
        help="Average validation epochs per retrain pass.",
    )
    parser.add_argument(
        "--train-step-multiplier",
        type=float,
        default=3.0,
        help="Multiplier converting forward FLOPs to train-step FLOPs (forward+backward+optimizer).",
    )
    parser.add_argument(
        "--profile-batch-size",
        type=int,
        default=256,
        help="Batch size used for DeepSpeed forward FLOPs profiling.",
    )
    return parser


def parse_resampling_steps(raw):
    try:
        return int(raw)
    except ValueError:
        return raw


def print_report(results):
    total = 0.0
    print(
        "task,seq_len,params,forward_flops_per_seq,train_flops,inference_flops,total_flops"
    )
    for row in results:
        total += row.total_flops
        print(
            f"{row.task},{row.sequence_length},{row.model_params},"
            f"{row.forward_flops_per_sequence:.6e},{row.train_flops_total:.6e},"
            f"{row.inference_flops_total:.6e},{row.total_flops:.6e}"
        )

    print()
    print(f"Total FLOPs across selected tasks: {total:.6e} ({humanize_flops(total)})")


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.resampling_steps = parse_resampling_steps(args.resampling_steps)

    tasks = parse_tasks(args.tasks)
    results = [estimate_task_flops(task, args) for task in tasks]
    results.sort(key=lambda item: item.total_flops, reverse=True)
    print_report(results)


if __name__ == "__main__":
    main()
