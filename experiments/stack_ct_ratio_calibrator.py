"""Train-fit / val-select / test-once stacking for the CT ratio head.

Uses locked bins3_regularized as the official baseline. Additional policies
are either parameter-free (equal-weight seeds) or fit on 2028 training
predictions only. Validation chooses; test is scored once.
Does not overwrite official_lock.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.calibrate_ct_ratio_v2 import (
    apply_bins,
    apply_linear,
    apply_pred,
    clip_ratio,
    fit_linear,
    geometry_id,
    row_key,
    summarize,
)
from lung_inverse_rendering.pinn_inverse import official_bins3_regularized

GNN = ROOT / "results" / "ion_ct_ratio_optimization" / "calibrate_v2"
SEEDS = ROOT / "results" / "ion_ct_expanded_training"
LOCK = ROOT / "results" / "ion_ct_ratio_optimization" / "fair_calibrate" / "official_lock.json"
OUT = ROOT / "results" / "ion_ct_ratio_optimization" / "stack_covariate_v1"
GATE = 0.25


def _load_records(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def _index(rows: list[dict]) -> dict[str, dict]:
    return {row_key(row): row for row in rows}


def _official_selection_key(summary: dict) -> tuple:
    return (-int(summary["geometries_passing"]), float(summary["median"]))


def _mean_ratio(rows_by_seed: list[dict]) -> float:
    return clip_ratio(float(np.mean([float(row["inclusion_ratio_estimated"]) for row in rows_by_seed])))


def _covariate_row(row: dict, calibrated: float) -> np.ndarray:
    return np.asarray(
        [
            math.log(max(calibrated, 1e-6)),
            math.log(max(float(row["radius_fraction_estimated"]), 1e-6)),
            math.log(max(float(row.get("flow_contrast", 1e-12)), 1e-12)),
            float(row.get("heterogeneity_probability", 0.5)),
            math.log(max(float(row["E_background_estimated"]), 1.0)),
        ],
        dtype=np.float64,
    )


def _fit_residual(train_rows: list[dict], edges: list[float], multipliers: list[float]) -> dict:
    calibrated = [apply_bins(float(row["inclusion_ratio_estimated"]), edges, multipliers) for row in train_rows]
    X = np.stack([_covariate_row(row, value) for row, value in zip(train_rows, calibrated)])
    y = np.log([float(row["inclusion_ratio_true"]) for row in train_rows]) - np.log(calibrated)
    mean = X.mean(0)
    std = np.maximum(X.std(0), 1e-6)
    Z = np.column_stack([np.ones(len(X)), (X - mean) / std])
    eye = np.eye(Z.shape[1])
    eye[0, 0] = 0.0
    coef = np.linalg.solve(Z.T @ Z + 1.0 * eye, Z.T @ y)
    return {"coef": coef.tolist(), "mean": mean.tolist(), "std": std.tolist()}


def _apply_residual(row: dict, fit: dict, edges: list[float], multipliers: list[float]) -> float:
    calibrated = apply_bins(float(row["inclusion_ratio_estimated"]), edges, multipliers)
    x = (_covariate_row(row, calibrated) - np.asarray(fit["mean"])) / np.asarray(fit["std"])
    z = np.concatenate([[1.0], x])
    return clip_ratio(float(calibrated * math.exp(float(z @ np.asarray(fit["coef"])))))


def main() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    edges = lock["fit"]["edges"]
    multipliers = lock["fit"]["multipliers"]
    gnn = {split: _load_records(GNN / f"gnn_{split}.json") for split in ("train", "val", "test")}
    seeds = {}
    for seed in (2026, 2027, 2028):
        seeds[seed] = {
            split: _index(_load_records(SEEDS / f"seed_{seed}" / f"predictions_gnn_{split}.json"))
            for split in ("val", "test")
        }
    residual = _fit_residual(gnn["train"], edges, multipliers)
    linear = fit_linear(gnn["train"])

    def policy_maps(name: str) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for split, rows in gnn.items():
            mapping = {}
            for row in rows:
                key = row_key(row)
                if name == "official_2028_bins3":
                    mapping[key] = official_bins3_regularized(row["inclusion_ratio_estimated"])
                elif name == "stack_linear_2028":
                    mapping[key] = apply_linear(row, linear)
                elif name == "covariate_residual_2028":
                    mapping[key] = _apply_residual(row, residual, edges, multipliers)
                elif name in {"seed_equal_mean_raw", "seed_equal_mean_bins3"}:
                    if split == "train":
                        mapping[key] = official_bins3_regularized(row["inclusion_ratio_estimated"])
                        continue
                    members = []
                    for seed in (2026, 2027, 2028):
                        member = seeds[seed][split].get(key)
                        if member is None:
                            member = seeds[seed][split].get(row["patient_id"])
                        if member is None:
                            raise KeyError(f"missing {seed} {split} {key}")
                        value = float(member["inclusion_ratio_estimated"])
                        if name == "seed_equal_mean_bins3":
                            value = official_bins3_regularized(value)
                        members.append({"inclusion_ratio_estimated": value})
                    mapping[key] = _mean_ratio(members)
                else:
                    raise KeyError(name)
            out[split] = mapping
        return out

    names = (
        "official_2028_bins3",
        "stack_linear_2028",
        "covariate_residual_2028",
        "seed_equal_mean_raw",
        "seed_equal_mean_bins3",
    )
    leaderboard = []
    for name in names:
        maps = policy_maps(name)
        scored = {split: apply_pred(gnn[split], maps[split]) for split in ("val", "test")}
        val = summarize(scored["val"])
        test = summarize(scored["test"])
        leaderboard.append({"policy": name, "validation": val, "test_after_freeze": test})

    selected = min(leaderboard, key=lambda row: _official_selection_key(row["validation"]))
    payload = {
        "schema_version": 1,
        "do_not_overwrite_official_lock": True,
        "official_rule": "validation_geometry_pass_then_median",
        "locked_bins3": {"edges": edges, "multipliers": multipliers},
        "residual_fit": residual,
        "selected_by_val_geometries": selected["policy"],
        "leaderboard": leaderboard,
        "paper_reference": {
            "official_hybrid_ratio": 0.17767180073003352,
            "official_geometry_pass": "6/6",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": selected["policy"],
                "validation": selected["validation"],
                "test_after_freeze": selected["test_after_freeze"],
                "wrote": str(OUT / "selection.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
