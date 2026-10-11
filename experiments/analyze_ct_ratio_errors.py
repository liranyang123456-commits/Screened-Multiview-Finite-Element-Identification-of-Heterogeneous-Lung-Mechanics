#!/usr/bin/env python3
"""Diagnose CT inclusion-ratio errors on already-written prediction JSON."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "results" / "ion_ct_expanded_training" / "seed_2028" / "predictions_gnn_val.json"
TEST = ROOT / "results" / "ion_ct_expanded_training" / "seed_2028" / "predictions_gnn_test.json"
GATE = 0.25
OUT = ROOT / "results" / "ion_ct_ratio_optimization" / "error_diagnosis.json"


def geometry_id(row: dict) -> str:
    patient = str(row.get("patient_id", ""))
    if "_scenario_" in patient:
        return patient.split("_scenario_", 1)[0]
    return patient


def summarize(rows: list[dict]) -> dict:
    err = np.asarray([float(r["inclusion_ratio_relative_error"]) for r in rows])
    true = np.asarray([float(r["inclusion_ratio_true"]) for r in rows])
    pred = np.asarray([float(r["inclusion_ratio_estimated"]) for r in rows])
    radius = np.asarray([float(r["radius_fraction_true"]) for r in rows])
    hetero = np.asarray([bool(r["heterogeneous_true"]) for r in rows])
    over = pred > true
    signed = (pred - true) / np.maximum(true, 1e-6)

    def median(mask: np.ndarray) -> float | None:
        return float(np.median(err[mask])) if mask.any() else None

    tertile = np.quantile(true, [1 / 3, 2 / 3])
    radius_t = np.quantile(radius[hetero], [1 / 3, 2 / 3]) if hetero.any() else (0.0, 0.0)
    by_geom: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, err):
        by_geom[geometry_id(row)].append(float(value))
    return {
        "n": int(len(rows)),
        "median": float(np.median(err)),
        "pass_rate": float(np.mean(err <= GATE)),
        "homo_n": int((~hetero).sum()),
        "hetero_n": int(hetero.sum()),
        "median_homo": median(~hetero),
        "median_hetero": median(hetero),
        "overestimate_rate": float(np.mean(over)),
        "median_signed_relative": float(np.median(signed)),
        "median_true_ratio": float(np.median(true)),
        "median_pred_ratio": float(np.median(pred)),
        "spearman_err_vs_true": float(_spearman(true, err)),
        "spearman_err_vs_radius": float(_spearman(radius, err)),
        "median_by_true_tertile": {
            "low": median(true <= tertile[0]),
            "mid": median((true > tertile[0]) & (true <= tertile[1])),
            "high": median(true > tertile[1]),
        },
        "median_by_radius_tertile_hetero": {
            "small": median(hetero & (radius <= radius_t[0])),
            "mid": median(hetero & (radius > radius_t[0]) & (radius <= radius_t[1])),
            "large": median(hetero & (radius > radius_t[1])),
        },
        "geometry_medians": {
            geom: float(np.median(vals)) for geom, vals in sorted(by_geom.items())
        },
        "geometries_passing": int(
            sum(np.median(vals) <= GATE for vals in by_geom.values())
        ),
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def shrink_oracle(rows: list[dict]) -> dict:
    """Diagnostic only: best global shrink-to-1 on this split."""
    true = np.asarray([float(r["inclusion_ratio_true"]) for r in rows])
    pred = np.asarray([float(r["inclusion_ratio_estimated"]) for r in rows])
    best_alpha, best = 1.0, float("inf")
    for alpha in np.linspace(0.3, 1.3, 41):
        calibrated = 1.0 + alpha * (pred - 1.0)
        calibrated = np.clip(calibrated, 1.0, 5.0)
        med = float(np.median(np.abs(calibrated / true - 1.0)))
        if med < best:
            best, best_alpha = med, float(alpha)
    return {"best_alpha": best_alpha, "median": best}


def main() -> None:
    report = {}
    for name, path in (("val", VAL), ("test", TEST)):
        rows = json.loads(path.read_text(encoding="utf-8"))["records"]
        report[name] = {
            "raw": summarize(rows),
            "oracle_shrink_on_this_split": shrink_oracle(rows),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
