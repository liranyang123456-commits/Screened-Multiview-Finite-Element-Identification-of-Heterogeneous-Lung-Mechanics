"""Run reproducible validation-selected lung training across fixed seeds."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[2024, 2025, 2026, 2027, 2028]
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed after '--' to train_lung_mesh_gnn.py",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    predictions = []
    extra = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
    for seed in args.seeds:
        seed_dir = args.results / f"seed_{seed}"
        metrics = seed_dir / "metrics_gnn.json"
        if not (args.resume_completed and metrics.exists()):
            command = [
                sys.executable,
                str(root / "experiments" / "train_lung_mesh_gnn.py"),
                "--dataset",
                str(args.dataset),
                "--results",
                str(seed_dir),
                "--model",
                "gnn",
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--seed",
                str(seed),
                *extra,
            ]
            subprocess.run(command, check=True)
        prediction = seed_dir / "predictions_gnn_test.json"
        if not prediction.exists():
            raise FileNotFoundError(prediction)
        predictions.append(prediction)
    aggregate = args.results / "multiseed_aggregate.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "evaluation" / "aggregate_multiseed.py"),
            "--inputs",
            *[str(path) for path in predictions],
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--out",
            str(aggregate),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
