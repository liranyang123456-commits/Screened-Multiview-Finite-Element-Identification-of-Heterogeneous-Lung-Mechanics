"""Finite validation-only model-selection cycle for lung MeshGNN experiments."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--run-fem-correction", action="store_true")
    parser.add_argument("--fem-patient-limit", type=int, default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    python = sys.executable
    args.results.mkdir(parents=True, exist_ok=True)
    status_path = args.results / "cycle_status.json"
    status = {"dataset": str(args.dataset), "rounds": [], "complete": False}

    population_dir = args.results / "baselines"
    run(
        [
            python,
            str(root / "experiments" / "evaluate_lung_population_prior.py"),
            "--dataset",
            str(args.dataset),
            "--split",
            "val",
            "--out",
            str(population_dir / "population_val.json"),
        ]
    )
    mlp_dir = args.results / "mlp"
    run(
        [
            python,
            str(root / "experiments" / "train_lung_mesh_gnn.py"),
            "--dataset",
            str(args.dataset),
            "--results",
            str(mlp_dir),
            "--model",
            "mlp",
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--skip-test",
        ]
    )

    configurations = [
        {"hidden_dim": 64, "layers": 3, "dropout": 0.05, "lr": 8e-4},
        {"hidden_dim": 128, "layers": 4, "dropout": 0.10, "lr": 3e-4},
        {"hidden_dim": 192, "layers": 5, "dropout": 0.15, "lr": 2e-4},
    ][: args.max_rounds]
    best_score = float("inf")
    stale = 0
    best_directory: Path | None = None
    for index, config in enumerate(configurations):
        directory = args.results / f"round_{index}"
        run(
            [
                python,
                str(root / "experiments" / "train_lung_mesh_gnn.py"),
                "--dataset",
                str(args.dataset),
                "--results",
                str(directory),
                "--model",
                "gnn",
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--hidden-dim",
                str(config["hidden_dim"]),
                "--layers",
                str(config["layers"]),
                "--dropout",
                str(config["dropout"]),
                "--lr",
                str(config["lr"]),
                "--skip-test",
            ]
        )
        metrics = json.loads(
            (directory / "metrics_gnn.json").read_text(encoding="utf-8")
        )
        validation = metrics["validation"]
        score = (
            validation["E_background_median_relative_error"]
            + validation["inclusion_ratio_median_relative_error"]
            + 0.25 * validation["heterogeneity"]["brier_score"]
        )
        status["rounds"].append(
            {"round": index, "config": config, "score": score, "validation": validation}
        )
        if score < best_score:
            best_score, stale, best_directory = score, 0, directory
        else:
            stale += 1
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if stale >= 2:
            break
    if best_directory is None:
        raise RuntimeError("No GNN training round completed")
    selected_checkpoint = args.results / "best_gnn.pt"
    shutil.copy2(best_directory / "best_gnn.pt", selected_checkpoint)
    test_predictions = args.results / "predictions_gnn_test.json"
    run(
        [
            python,
            str(root / "evaluation" / "evaluate_lung_mesh_gnn.py"),
            "--dataset",
            str(args.dataset),
            "--checkpoint",
            str(selected_checkpoint),
            "--out",
            str(test_predictions),
            "--split",
            "test",
        ]
    )
    test_result = json.loads(test_predictions.read_text(encoding="utf-8"))
    status["selected_round"] = int(best_directory.name.rsplit("_", 1)[1])
    status["test"] = test_result["metrics"]
    status["validation_gate_passed"] = bool(
        status["rounds"][status["selected_round"]]["validation"][
            "E_background_median_relative_error"
        ]
        <= 0.15
        and status["rounds"][status["selected_round"]]["validation"][
            "inclusion_ratio_median_relative_error"
        ]
        <= 0.35
    )
    if args.run_fem_correction:
        fem_results = args.results / "fem_correction"
        base = [
            python,
            str(root / "lung_inverse_rendering" / "evaluate_sim_lung_v2.py"),
            "--dataset",
            str(args.dataset),
            "--results",
            str(fem_results),
            "--patient-limit",
            str(args.fem_patient_limit),
            "--max-nfev",
            "24",
        ]
        run([*base, "--output-tag", "fixed_init"])
        run(
            [
                *base,
                "--initial-predictions",
                str(test_predictions),
                "--output-tag",
                "learned_init",
            ]
        )
    status["complete"] = True
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
