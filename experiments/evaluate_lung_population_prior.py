"""Patient-level population-prior baseline for lung material prediction."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.material_uncertainty_metrics import (  # noqa: E402
    bootstrap_median_ci,
    gaussian_interval_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = (
        args.dataset if args.dataset.is_file() else args.dataset / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train = [row for row in manifest["patients"] if row["split"] == "train"]
    target = [row for row in manifest["patients"] if row["split"] == args.split]
    if not train or not target:
        raise ValueError("Population baseline requires non-empty train and target splits")
    train_log_E = np.log([row["E_background"] for row in train])
    train_log_ratio = np.log([row["inclusion_ratio"] for row in train])
    mean_log_E, mean_log_ratio = float(train_log_E.mean()), float(train_log_ratio.mean())
    std_log_E = max(float(train_log_E.std(ddof=1)), 0.05)
    std_log_ratio = max(float(train_log_ratio.std(ddof=1)), 0.05)
    records = []
    for row in target:
        predicted_E = math.exp(mean_log_E)
        predicted_ratio = math.exp(mean_log_ratio)
        records.append(
            {
                "patient_id": row["patient_id"],
                "E_background_true": row["E_background"],
                "E_background_estimated": predicted_E,
                "inclusion_ratio_true": row["inclusion_ratio"],
                "inclusion_ratio_estimated": predicted_ratio,
                "log_E_std": std_log_E,
                "log_ratio_std": std_log_ratio,
                "E_background_relative_error": abs(predicted_E - row["E_background"])
                / row["E_background"],
                "inclusion_ratio_relative_error": abs(
                    predicted_ratio - row["inclusion_ratio"]
                )
                / row["inclusion_ratio"],
            }
        )
    E_errors = [row["E_background_relative_error"] for row in records]
    ratio_errors = [row["inclusion_ratio_relative_error"] for row in records]
    result = {
        "model": "population_log_normal_prior",
        "dataset": manifest["version"],
        "split": args.split,
        "patient_count": len(records),
        "E_background_median_relative_error": float(np.median(E_errors)),
        "E_background_median_bootstrap_95_ci": bootstrap_median_ci(E_errors),
        "inclusion_ratio_median_relative_error": float(np.median(ratio_errors)),
        "inclusion_ratio_median_bootstrap_95_ci": bootstrap_median_ci(
            ratio_errors, seed=2027
        ),
        "log_E_uncertainty": gaussian_interval_metrics(
            np.log([row["E_background_true"] for row in records]),
            np.full(len(records), mean_log_E),
            np.full(len(records), std_log_E),
        ),
        "log_ratio_uncertainty": gaussian_interval_metrics(
            np.log([row["inclusion_ratio_true"] for row in records]),
            np.full(len(records), mean_log_ratio),
            np.full(len(records), std_log_ratio),
        ),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
