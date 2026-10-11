#!/usr/bin/env python3
"""Validation-only selection of a CT inclusion-ratio head.

The locked test geometries are scored only after a policy is frozen on the
three validation geometries. No test label is used for selection.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "results" / "ion_ct_expanded_training"
SEEDS = (2026, 2027, 2028)
GATE = 0.25
OUT = ROOT / "results" / "ion_ct_ratio_optimization"


def load_records(seed: int, split: str) -> list[dict]:
    if seed == "selected":
        path = TRAINING / "selected" / f"predictions_gnn_{split}.json"
    else:
        path = TRAINING / f"seed_{seed}" / f"predictions_gnn_{split}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["records"]


def geometry_id(row: dict) -> str:
    patient = str(row.get("patient_id", row.get("entry_id", "")))
    if "geometry_id" in row:
        return str(row["geometry_id"])
    if patient.startswith("geom_"):
        return patient.split("_scenario_")[0].replace("geom_", "geom_", 1)
    return patient


def ratio_error(true: float, pred: float) -> float:
    return abs(pred / true - 1.0)


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def summarize(rows: list[dict], pred_key: str = "inclusion_ratio_estimated") -> dict:
    errors = [
        ratio_error(float(row["inclusion_ratio_true"]), float(row[pred_key]))
        for row in rows
        if row.get("inclusion_ratio_true")
    ]
    by_geom: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if not row.get("inclusion_ratio_true"):
            continue
        by_geom[geometry_id(row)].append(
            ratio_error(float(row["inclusion_ratio_true"]), float(row[pred_key]))
        )
    geom_medians = {geom: median(vals) for geom, vals in sorted(by_geom.items())}
    return {
        "n": len(errors),
        "median": median(errors),
        "pass_rate": float(np.mean(np.asarray(errors) <= GATE)) if errors else float("nan"),
        "geometry_medians": geom_medians,
        "geometries_passing": int(sum(m <= GATE for m in geom_medians.values())),
        "geometry_count": len(geom_medians),
    }


def affine_fit(rows: list[dict]) -> tuple[float, float]:
    log_true = np.log([float(r["inclusion_ratio_true"]) for r in rows])
    log_pred = np.log([max(float(r["inclusion_ratio_estimated"]), 1e-6) for r in rows])
    design = np.column_stack([np.ones(len(log_pred)), log_pred])
    coef, *_ = np.linalg.lstsq(design, log_true, rcond=None)
    return float(coef[0]), float(coef[1])


def apply_affine(rows: list[dict], intercept: float, slope: float) -> list[dict]:
    out = []
    for row in rows:
        pred = float(row["inclusion_ratio_estimated"])
        calibrated = float(np.exp(intercept + slope * np.log(max(pred, 1e-6))))
        new = dict(row)
        new["inclusion_ratio_estimated"] = calibrated
        new["inclusion_ratio_relative_error"] = ratio_error(
            float(row["inclusion_ratio_true"]), calibrated
        )
        out.append(new)
    return out


def index_by_patient(rows: list[dict]) -> dict[str, dict]:
    return {str(row["patient_id"]): row for row in rows}


def ensemble_rows(seed_rows: dict[int, list[dict]], how: str) -> list[dict]:
    indexed = {seed: index_by_patient(rows) for seed, rows in seed_rows.items()}
    patients = sorted(set.intersection(*(set(m) for m in indexed.values())))
    out = []
    for patient in patients:
        base = dict(indexed[SEEDS[0]][patient])
        preds = [float(indexed[seed][patient]["inclusion_ratio_estimated"]) for seed in SEEDS]
        base["inclusion_ratio_estimated"] = float(np.median(preds) if how == "median" else np.mean(preds))
        base["inclusion_ratio_relative_error"] = ratio_error(
            float(base["inclusion_ratio_true"]),
            float(base["inclusion_ratio_estimated"]),
        )
        out.append(base)
    return out


def size_bias_fit(rows: list[dict]) -> tuple[float, float]:
    """Fit log-ratio residual vs log predicted radius on validation."""
    residual = []
    radius = []
    for row in rows:
        true = float(row["inclusion_ratio_true"])
        pred = max(float(row["inclusion_ratio_estimated"]), 1e-6)
        residual.append(np.log(true) - np.log(pred))
        radius.append(np.log(max(float(row["radius_fraction_estimated"]), 1e-6)))
    design = np.column_stack([np.ones(len(radius)), radius])
    coef, *_ = np.linalg.lstsq(design, residual, rcond=None)
    return float(coef[0]), float(coef[1])


def apply_size_bias(rows: list[dict], intercept: float, slope: float) -> list[dict]:
    out = []
    for row in rows:
        pred = max(float(row["inclusion_ratio_estimated"]), 1e-6)
        radius = np.log(max(float(row["radius_fraction_estimated"]), 1e-6))
        calibrated = float(np.exp(np.log(pred) + intercept + slope * radius))
        new = dict(row)
        new["inclusion_ratio_estimated"] = calibrated
        new["inclusion_ratio_relative_error"] = ratio_error(
            float(row["inclusion_ratio_true"]), calibrated
        )
        out.append(new)
    return out


def main() -> None:
    val = {seed: load_records(seed, "val") for seed in SEEDS}
    test = {seed: load_records(seed, "test") for seed in SEEDS}
    selected_seed = 2027

    policies = []

    def add(name: str, val_rows: list[dict], test_rows: list[dict], extra: dict | None = None) -> None:
        policies.append(
            {
                "name": name,
                "validation": summarize(val_rows),
                "test_after_freeze": summarize(test_rows),
                **(extra or {}),
            }
        )

    for seed in SEEDS:
        add(f"seed_{seed}_raw", val[seed], test[seed])

    for how in ("median", "mean"):
        add(f"ensemble_{how}", ensemble_rows(val, how), ensemble_rows(test, how))

    intercept, slope = affine_fit(val[selected_seed])
    add(
        "affine_on_selected_seed",
        apply_affine(val[selected_seed], intercept, slope),
        apply_affine(test[selected_seed], intercept, slope),
        {"affine": {"intercept": intercept, "slope": slope, "fit_on": "val seed 2027"}},
    )

    best_val_seed = min(SEEDS, key=lambda seed: summarize(val[seed])["median"])
    intercept_b, slope_b = affine_fit(val[best_val_seed])
    add(
        f"affine_on_best_val_ratio_seed_{best_val_seed}",
        apply_affine(val[best_val_seed], intercept_b, slope_b),
        apply_affine(test[best_val_seed], intercept_b, slope_b),
        {"affine": {"intercept": intercept_b, "slope": slope_b, "fit_on": f"val seed {best_val_seed}"}},
    )

    ens_val = ensemble_rows(val, "median")
    ens_test = ensemble_rows(test, "median")
    intercept_e, slope_e = affine_fit(ens_val)
    add(
        "affine_on_median_ensemble",
        apply_affine(ens_val, intercept_e, slope_e),
        apply_affine(ens_test, intercept_e, slope_e),
        {"affine": {"intercept": intercept_e, "slope": slope_e, "fit_on": "val median ensemble"}},
    )

    a0, a1 = size_bias_fit(val[selected_seed])
    add(
        "size_bias_on_selected_seed",
        apply_size_bias(val[selected_seed], a0, a1),
        apply_size_bias(test[selected_seed], a0, a1),
        {"size_bias": {"intercept": a0, "slope": a1, "fit_on": "val seed 2027"}},
    )

    a0b, a1b = size_bias_fit(val[best_val_seed])
    add(
        f"size_bias_on_best_val_ratio_seed_{best_val_seed}",
        apply_size_bias(val[best_val_seed], a0b, a1b),
        apply_size_bias(test[best_val_seed], a0b, a1b),
        {"size_bias": {"intercept": a0b, "slope": a1b, "fit_on": f"val seed {best_val_seed}"}},
    )

    # Selection uses validation median, then geometries passing, then pass rate.
    ranked = sorted(
        policies,
        key=lambda row: (
            row["validation"]["median"],
            -row["validation"]["geometries_passing"],
            -row["validation"]["pass_rate"],
        ),
    )
    selected = ranked[0]
    report = {
        "schema_version": 1,
        "gate": GATE,
        "selection": (
            "minimum validation median ratio error; "
            "test_after_freeze is unused for selection"
        ),
        "selected_policy": selected["name"],
        "policies": ranked,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ratio_head_selection.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(
        {
            "selected_policy": selected["name"],
            "validation_median": selected["validation"]["median"],
            "validation_geometries_passing": (
                f"{selected['validation']['geometries_passing']}/"
                f"{selected['validation']['geometry_count']}"
            ),
            "test_median_after_freeze": selected["test_after_freeze"]["median"],
            "test_geometries_passing": (
                f"{selected['test_after_freeze']['geometries_passing']}/"
                f"{selected['test_after_freeze']['geometry_count']}"
            ),
            "leaderboard_val_median": [
                {"name": row["name"], "val": row["validation"]["median"], "test": row["test_after_freeze"]["median"]}
                for row in ranked
            ],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
