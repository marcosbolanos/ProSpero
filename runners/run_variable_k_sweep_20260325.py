import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def build_experiments():
    return [
        {
            "name": "lgk_frozen_esm_cnn",
            "task": "LGK",
            "surrogate_arch": "frozen_esm_cnn",
            "extra_args": [],
        },
        {
            "name": "aav_frozen_esm_cnn_proj64_ln",
            "task": "AAV",
            "surrogate_arch": "frozen_esm_cnn",
            "extra_args": [
                "--esm-cnn-projection-dim",
                "64",
                "--esm-cnn-use-layernorm",
            ],
        },
        {
            "name": "lgk_frozen_esm_cnn_proj64_ln",
            "task": "LGK",
            "surrogate_arch": "frozen_esm_cnn",
            "extra_args": [
                "--esm-cnn-projection-dim",
                "64",
                "--esm-cnn-use-layernorm",
            ],
        },
        {
            "name": "aav_frozen_esm_cnn_proj64_ln_onehot",
            "task": "AAV",
            "surrogate_arch": "frozen_esm_cnn",
            "extra_args": [
                "--esm-cnn-projection-dim",
                "64",
                "--esm-cnn-use-layernorm",
                "--esm-cnn-concat-one-hot",
            ],
        },
        {
            "name": "lgk_frozen_esm_cnn_proj64_ln_onehot",
            "task": "LGK",
            "surrogate_arch": "frozen_esm_cnn",
            "extra_args": [
                "--esm-cnn-projection-dim",
                "64",
                "--esm-cnn-use-layernorm",
                "--esm-cnn-concat-one-hot",
            ],
        },
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Run the requested variable-k ESM sweep sequentially."
    )
    parser.add_argument("--n-samples", default="8,16,32,64,128")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--n-iters", type=int, default=10)
    parser.add_argument("--min-corruptions", type=int, default=3)
    parser.add_argument("--max-corruptions", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--disable-esm-cache", action="store_true", default=False)
    parser.add_argument("--n-queries-base", type=int, default=None)
    parser.add_argument("--uv-cache-dir", default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("outputs") / f"variable_k_sweep_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to {results_dir}")

    for experiment in build_experiments():
        experiment_dir = results_dir / experiment["name"]
        command = [
            sys.executable,
            "src/prospero/runners/run_variable_k.py",
            str(experiment_dir),
            "--task",
            experiment["task"],
            "--surrogate-arch",
            experiment["surrogate_arch"],
            "--n-samples",
            args.n_samples,
            "--seeds",
            args.seeds,
            "--n-iters",
            str(args.n_iters),
            "--min-corruptions",
            str(args.min_corruptions),
            "--max-corruptions",
            str(args.max_corruptions),
            "--max-workers",
            str(args.max_workers),
        ]
        if args.n_queries_base is not None:
            command.extend(["--n-queries-base", str(args.n_queries_base)])
        if args.disable_esm_cache:
            command.append("--disable-esm-cache")
        if args.uv_cache_dir is not None:
            command.extend(["--uv-cache-dir", args.uv_cache_dir])
        command.extend(experiment["extra_args"])

        print(f"Running {experiment['name']}")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
