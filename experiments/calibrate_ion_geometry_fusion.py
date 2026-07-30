"""Select global-head versus node-partition geometry fusion on validation only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train_lung_mesh_gnn import (  # noqa: E402
    build_model,
    evaluate,
    make_loaders,
)


DEFAULT_DATASET = ROOT / "dataset" / "ion_ct_synthetic_mechanics540"
DEFAULT_RESULTS = ROOT / "results" / "ion_ct_expanded_training" / "selected"


def geometry_score(metrics: dict) -> float:
    return (
        float(metrics["center_error_normalized_median"])
        + float(metrics["radius_relative_error_median"])
        + 0.5 * (1.0 - float(metrics["partition_soft_dice_mean"]))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    args = parser.parse_args()
    checkpoint = args.results / "best_gnn.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    config = state["config"]
    _, validation_loader, test_loader, input_dim = make_loaders(
        args.dataset,
        batch_size=8,
        workers=0,
        experiments_limit=config.get("experiments_limit"),
        observation=config.get("observation", "image_tracks"),
        track_noise_px=float(config.get("track_noise_px", 0.0)),
        single_view=bool(config.get("single_view", False)),
        no_depth=bool(config.get("no_depth", False)),
        no_confidence=bool(config.get("no_confidence", False)),
        peak_only=bool(config.get("peak_only", False)),
    )
    model = build_model(
        config["model"],
        input_dim,
        config["hidden_dim"],
        config["layers"],
        config["dropout"],
        config.get("dynamic_dim"),
    ).to(device)
    model.load_state_dict(state["model"])
    candidates = []
    for weight in args.weights:
        metrics, _ = evaluate(
            model,
            validation_loader,
            device,
            geometry_fusion_weight=weight,
        )
        candidates.append(
            {
                "weight": weight,
                "validation_geometry_score": geometry_score(metrics),
                "validation": metrics,
            }
        )
    selected = min(candidates, key=lambda row: row["validation_geometry_score"])
    test_metrics, test_records = evaluate(
        model,
        test_loader,
        device,
        geometry_fusion_weight=float(selected["weight"]),
    )
    state["config"]["geometry_fusion_weight"] = selected["weight"]
    torch.save(state, checkpoint)
    result = {
        "schema_version": 1,
        "selection": "validation-only composite geometry score",
        "selected_weight": selected["weight"],
        "candidates": candidates,
        "test": test_metrics,
    }
    (args.results / "geometry_fusion_calibration.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    (args.results / "predictions_gnn_test_geometry_calibrated.json").write_text(
        json.dumps(
            {
                "model": "gnn",
                "split": "test",
                "geometry_fusion_weight": selected["weight"],
                "metrics": test_metrics,
                "records": test_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_weight": selected["weight"],
                "test": test_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
