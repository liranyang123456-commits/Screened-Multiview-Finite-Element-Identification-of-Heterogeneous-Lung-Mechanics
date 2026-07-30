"""Train multiple expanded-cohort MeshGNN seeds and select on validation only."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "experiments" / "train_lung_mesh_gnn.py"
DEFAULT_DATASET = ROOT / "dataset" / "ion_ct_synthetic_mechanics540"
DEFAULT_RESULTS = ROOT / "results" / "ion_ct_expanded_training"


def _validation_score(metrics: dict) -> float:
    validation = metrics["validation"]
    score = (
        float(validation["E_background_median_relative_error"])
        + float(validation["inclusion_ratio_median_relative_error"])
        + 0.25 * float(validation["heterogeneity"]["brier_score"])
    )
    for key, weight in (
        ("center_error_normalized_median", 0.5),
        ("radius_relative_error_median", 0.5),
    ):
        if validation.get(key) is not None:
            score += weight * float(validation[key])
    if validation.get("partition_soft_dice_mean") is not None:
        score += 0.5 * (1.0 - float(validation["partition_soft_dice_mean"]))
    return score


def train_seed(args: argparse.Namespace, seed: int) -> dict:
    result_dir = args.results / f"seed_{seed}"
    metrics_path = result_dir / "metrics_gnn.json"
    if metrics_path.exists() and not args.overwrite:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    if result_dir.exists() and args.overwrite:
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True)
    command = [
        sys.executable,
        str(TRAINER.relative_to(ROOT)),
        "--dataset",
        str(args.dataset.relative_to(ROOT)),
        "--results",
        str(result_dir.relative_to(ROOT)),
        "--model",
        "gnn",
        "--observation",
        "image_tracks",
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--layers",
        str(args.layers),
        "--workers",
        str(args.loader_workers),
        "--seed",
        str(seed),
        "--ratio-loss-weight",
        "2.5",
        "--sdf-weight",
        "1.0",
        "--partition-weight",
        "1.5",
    ]
    with (result_dir / "training.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    args.results = args.results.resolve()
    args.results.mkdir(parents=True, exist_ok=True)
    candidates = []
    for seed in args.seeds:
        metrics = train_seed(args, seed)
        candidates.append(
            {
                "seed": seed,
                "validation_score": _validation_score(metrics),
                "selected_epoch": metrics["selected_epoch"],
                "validation": metrics["validation"],
                "test": metrics["test"],
            }
        )
    selected = min(candidates, key=lambda item: item["validation_score"])
    selected_dir = args.results / "selected"
    selected_dir.mkdir(exist_ok=True)
    source_dir = args.results / f"seed_{selected['seed']}"
    for name in (
        "best_gnn.pt",
        "metrics_gnn.json",
        "predictions_gnn_val.json",
        "predictions_gnn_test.json",
    ):
        shutil.copy2(source_dir / name, selected_dir / name)
    summary = {
        "schema_version": 1,
        "selection": "minimum composite validation score; test metrics not used",
        "seed_count": len(candidates),
        "selected_seed": selected["seed"],
        "candidates": candidates,
    }
    (args.results / "multiseed_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_seed": selected["seed"],
                "validation_score": selected["validation_score"],
                "test": selected["test"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
