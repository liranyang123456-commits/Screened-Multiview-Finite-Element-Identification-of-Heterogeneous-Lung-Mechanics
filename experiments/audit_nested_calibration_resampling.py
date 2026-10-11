#!/usr/bin/env python3
"""Repeat nested geometry calibration resampling on training geometries only.

This audit never opens validation or locked-test predictions. It records the
input/script hashes and writes an isolated, fully reproducible JSON artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import run_ct_reviewer_drivers as driver  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--heldout-geometries", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "ct_reviewer_drivers_v6" / "repeated_outer_audit_seed20261008_1000.json",
    )
    parser.add_argument(
        "--loo-refit-only",
        action="store_true",
        help=(
            "instead of repeated resampling, write driver.calibration_stability() "
            "(deterministic leave-one-training-geometry-out refit scored once on the "
            "validation geometries; archived as calibration_stability_loo_refit.json)"
        ),
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    if args.loo_refit_only:
        result = driver.calibration_stability()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({
            "output": str(args.output),
            "outer_geometry_count": result["outer_geometry_count"],
            "outer_policy_selection_counts": result["outer_policy_selection_counts"],
            "outer_validation_median_error": result["outer_validation"]["median_error"],
        }, indent=2))
        return

    predictions = driver.RATIO_PREDICTIONS / "gnn_train.json"
    train = driver.load_json(predictions)["records"]
    # Explicitly do not call dataset_splits(), calibration_stability(), or read
    # gnn_val/gnn_test: this is a training-only nested-resampling audit.
    result = driver.repeated_outer_geometry_resampling(
        train,
        repeats=args.repeats,
        heldout_geometry_count=args.heldout_geometries,
        seed=args.seed,
    )
    result["audit_metadata"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_data": "gnn_train.json only",
        "validation_or_test_predictions_opened": False,
        "train_prediction_path": str(predictions.relative_to(ROOT)),
        "train_prediction_sha256": sha256(predictions),
        "driver_path": str(Path(driver.__file__).relative_to(ROOT)),
        "driver_sha256": sha256(Path(driver.__file__)),
        "resampling_script_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "repeats": result["repeats"],
        "training_geometries": result["training_geometry_count"],
        "selection_counts": result["selection_counts"],
        "outer_median_error_quantiles": result["outer_median_error_quantiles"],
        "outer_geometry_median_error_quantiles": result["outer_geometry_median_error_quantiles"],
    }, indent=2))


if __name__ == "__main__":
    main()
