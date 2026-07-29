"""Aggregate patient-level material metrics across independent training seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap_patient_median(values: np.ndarray, seed: int) -> list[float]:
    generator = np.random.default_rng(seed)
    draws = np.asarray(
        [
            np.median(generator.choice(values, size=len(values), replace=True))
            for _ in range(5000)
        ]
    )
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def records_from(payload: dict) -> list[dict]:
    if "records" in payload:
        return payload["records"]
    if "test_records" in payload:
        return payload["test_records"]
    raise ValueError("Input has no patient-level records")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.inputs) != len(args.seeds):
        raise ValueError("--inputs and --seeds must have the same length")
    payloads = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.inputs
    ]
    records_by_seed = [records_from(payload) for payload in payloads]
    patient_sets = [
        {row["patient_id"] for row in records} for records in records_by_seed
    ]
    if any(patient_set != patient_sets[0] for patient_set in patient_sets[1:]):
        raise ValueError("All seeds must evaluate the identical patient cohort")
    patient_ids = sorted(patient_sets[0])
    per_seed = []
    for seed, records in zip(args.seeds, records_by_seed):
        indexed = {row["patient_id"]: row for row in records}
        E_errors = np.asarray(
            [indexed[patient]["E_background_relative_error"] for patient in patient_ids]
        )
        ratio_errors = np.asarray(
            [indexed[patient]["inclusion_ratio_relative_error"] for patient in patient_ids]
        )
        per_seed.append(
            {
                "seed": seed,
                "E_background_median_relative_error": float(np.median(E_errors)),
                "inclusion_ratio_median_relative_error": float(
                    np.median(ratio_errors)
                ),
            }
        )
    E_seed = np.asarray(
        [row["E_background_median_relative_error"] for row in per_seed]
    )
    ratio_seed = np.asarray(
        [row["inclusion_ratio_median_relative_error"] for row in per_seed]
    )
    ensemble_records = []
    for patient_id in patient_ids:
        rows = [
            next(row for row in records if row["patient_id"] == patient_id)
            for records in records_by_seed
        ]
        ensemble_records.append(
            {
                "patient_id": patient_id,
                "E_background_relative_error": float(
                    np.mean([row["E_background_relative_error"] for row in rows])
                ),
                "inclusion_ratio_relative_error": float(
                    np.mean([row["inclusion_ratio_relative_error"] for row in rows])
                ),
            }
        )
    ensemble_E = np.asarray(
        [row["E_background_relative_error"] for row in ensemble_records]
    )
    ensemble_ratio = np.asarray(
        [row["inclusion_ratio_relative_error"] for row in ensemble_records]
    )
    result = {
        "seed_count": len(args.seeds),
        "seeds": args.seeds,
        "patient_count": len(patient_ids),
        "per_seed": per_seed,
        "E_background_seed_mean": float(E_seed.mean()),
        "E_background_seed_std": float(E_seed.std(ddof=1)),
        "inclusion_ratio_seed_mean": float(ratio_seed.mean()),
        "inclusion_ratio_seed_std": float(ratio_seed.std(ddof=1)),
        "ensemble_patient_E_median": float(np.median(ensemble_E)),
        "ensemble_patient_E_bootstrap_95_ci": bootstrap_patient_median(
            ensemble_E, 2026
        ),
        "ensemble_patient_ratio_median": float(np.median(ensemble_ratio)),
        "ensemble_patient_ratio_bootstrap_95_ci": bootstrap_patient_median(
            ensemble_ratio, 2027
        ),
        "records": ensemble_records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
