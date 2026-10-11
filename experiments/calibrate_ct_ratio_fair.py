#!/usr/bin/env python3
"""Fair, train-fit / validation-select ratio calibration for GNN and baselines.

Each method gets the same procedure:
  1. fit calibrators on its own training predictions
  2. pick one calibrator on validation
  3. score test once after that choice is frozen

Robust bin variants shrink multipliers toward 1 or average them across
leave-one-training-geometry-out folds so a single aggressive bin cannot
dominate selection.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.sim_lung_graph import SimLungGraphDataset  # noqa: E402
from experiments.benchmark_ion_ct_synthetic_mechanics import (  # noqa: E402
    guarded_features,
    guarded_prediction,
)
from experiments.benchmark_lung_response_baselines import _prediction  # noqa: E402
from experiments.calibrate_ct_ratio_v2 import (  # noqa: E402
    apply_bins,
    apply_pred,
    clip_ratio,
    geometry_id,
    row_key,
    selection_key,
    summarize,
)
from experiments.train_lung_response_calibrator import features  # noqa: E402

DATASET = ROOT / "dataset" / "ion_ct_synthetic_mechanics540"
GNN_DUMP = ROOT / "results" / "ion_ct_ratio_optimization" / "calibrate_v2"
BENCH = ROOT / "results" / "ion_ct_expanded120_final" / "benchmark.json"
ARTIFACT = ROOT / "results" / "ion_ct_expanded30_physics16" / "baseline_models.joblib"
SEED2027 = ROOT / "results" / "ion_ct_expanded_training" / "seed_2027"
OUT = ROOT / "results" / "ion_ct_ratio_optimization" / "fair_calibrate"
GATE = 0.25


def load_gnn(split: str) -> list[dict]:
    path = GNN_DUMP / f"gnn_{split}.json"
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def load_seed2027(split: str) -> list[dict]:
    path = SEED2027 / f"predictions_gnn_{split}.json"
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def extract_split_features(split: str) -> tuple[np.ndarray, list[dict]]:
    dataset = SimLungGraphDataset(
        DATASET,
        split=split,
        observation_key="image_tracks",
        cache_graphs=True,
    )
    X = []
    meta = []
    print(f"extracting features on {split} ({len(dataset)})", flush=True)
    for graph in dataset:
        label = graph["labels"]
        X.append(features(graph))
        meta.append(
            {
                "patient_id": str(graph["patient_id"]),
                "inclusion_ratio_true": float(math.exp(float(label["log_ratio"]))),
                "E_background_true": float(math.exp(float(label["log_E_background"]))),
                "radius_fraction_true": float(label["radius_fraction"]),
                "center_fraction_true": label["center_fraction"].tolist(),
                "heterogeneous_true": bool(float(label["heterogeneous"]) > 0.5),
            }
        )
    return np.asarray(X), meta


def baseline_records_from_features(
    family: str,
    artifact: dict[str, Any],
    X: np.ndarray,
    meta: list[dict],
) -> list[dict]:
    Xg = guarded_features(X, artifact)
    pred = guarded_prediction(_prediction(artifact["models"][family]["main"], Xg), artifact)
    radius = guarded_prediction(
        _prediction(artifact["models"][family]["radius"], Xg), artifact
    )
    rows = []
    for info, estimate, radius_row in zip(meta, pred, radius, strict=True):
        true_ratio = info["inclusion_ratio_true"]
        est_ratio = clip_ratio(float(math.exp(float(estimate[1]))))
        true_e = info["E_background_true"]
        est_e = float(math.exp(float(estimate[0])))
        est_radius = float(radius_row.reshape(-1)[0])
        true_radius = info["radius_fraction_true"]
        rows.append(
            {
                **info,
                "inclusion_ratio_estimated": est_ratio,
                "inclusion_ratio_relative_error": abs(est_ratio / true_ratio - 1.0),
                "E_background_estimated": est_e,
                "E_background_relative_error": abs(est_e / true_e - 1.0),
                "radius_fraction_estimated": est_radius,
                "radius_relative_error": (
                    abs(est_radius / true_radius - 1.0)
                    if info["heterogeneous_true"]
                    else None
                ),
                "center_fraction_estimated": estimate[2:5].tolist(),
                "center_error_normalized": (
                    float(np.linalg.norm(estimate[2:5] - np.asarray(info["center_fraction_true"])))
                    if info["heterogeneous_true"]
                    else None
                ),
                "method": family,
            }
        )
    return rows


def overlay_benchmark_test(rows: list[dict], family: str) -> list[dict]:
    """Prefer the frozen 120-scene benchmark numbers for the official test split."""
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    official = {
        row_key(row): row
        for row in bench["records"]
        if row["method"] == family
    }
    if not official:
        return rows
    merged = []
    for row in rows:
        src = official.get(row_key(row))
        if src is None:
            merged.append(row)
            continue
        new = dict(row)
        new["inclusion_ratio_estimated"] = float(src["inclusion_ratio_estimated"])
        new["inclusion_ratio_relative_error"] = float(src["inclusion_ratio_relative_error"])
        new["E_background_estimated"] = float(src["E_background_estimated"])
        new["E_background_relative_error"] = float(src["E_background_relative_error"])
        if src.get("radius_relative_error") is not None:
            new["radius_fraction_estimated"] = src.get(
                "radius_fraction_estimated", new.get("radius_fraction_estimated")
            )
            new["radius_relative_error"] = src["radius_relative_error"]
        if src.get("center_error_normalized") is not None:
            new["center_error_normalized"] = src["center_error_normalized"]
        merged.append(new)
    return merged


def fit_bins(
    train_rows: list[dict],
    bins: int = 4,
    l2: float = 0.0,
    lo: float = 0.55,
    hi: float = 1.25,
) -> tuple[list[float], list[float]]:
    pred = np.asarray([float(r["inclusion_ratio_estimated"]) for r in train_rows])
    true = np.asarray([float(r["inclusion_ratio_true"]) for r in train_rows])
    edges = np.quantile(pred, np.linspace(0.0, 1.0, bins + 1))
    edges[0] = 0.0
    edges[-1] = 10.0
    multipliers = []
    for index in range(bins):
        if index == bins - 1:
            mask = pred >= edges[index]
        else:
            mask = (pred >= edges[index]) & (pred < edges[index + 1])
        if not mask.any():
            multipliers.append(1.0)
            continue
        best_m, best = 1.0, float("inf")
        for mult in np.linspace(lo, hi, 29):
            calibrated = np.clip(pred[mask] * mult, 1.0, 5.0)
            med = float(np.median(np.abs(calibrated / np.maximum(true[mask], 1e-8) - 1.0)))
            score = med + l2 * (mult - 1.0) ** 2
            if score < best:
                best, best_m = score, float(mult)
        multipliers.append(best_m)
    return edges.tolist(), multipliers


def logo_bins(train_rows: list[dict], bins: int = 4, l2: float = 0.08) -> tuple[list[float], list[float]]:
    edges, _ = fit_bins(train_rows, bins=bins, l2=l2)
    geoms = sorted({geometry_id(row) for row in train_rows})
    collected = []
    for held in geoms:
        complement = [row for row in train_rows if geometry_id(row) != held]
        if len(complement) < 20:
            continue
        _, multipliers = fit_bins(complement, bins=bins, l2=l2)
        if len(multipliers) == bins:
            collected.append(multipliers)
    if not collected:
        return edges, [1.0] * bins
    return edges, np.mean(np.asarray(collected), axis=0).tolist()


def fit_isotonic(train_rows: list[dict]) -> IsotonicRegression:
    pred = np.asarray([float(r["inclusion_ratio_estimated"]) for r in train_rows])
    true = np.asarray([float(r["inclusion_ratio_true"]) for r in train_rows])
    model = IsotonicRegression(increasing=True, y_min=1.0, y_max=5.0, out_of_bounds="clip")
    model.fit(pred, true)
    return model


def shrink_alpha(train_rows: list[dict]) -> float:
    pred = np.asarray([float(r["inclusion_ratio_estimated"]) for r in train_rows])
    true = np.asarray([float(r["inclusion_ratio_true"]) for r in train_rows])
    best_a, best = 1.0, float("inf")
    for alpha in np.linspace(0.4, 1.25, 35):
        calibrated = np.clip(1.0 + alpha * (pred - 1.0), 1.0, 5.0)
        med = float(np.median(np.abs(calibrated / np.maximum(true, 1e-8) - 1.0)))
        if med < best:
            best, best_a = med, float(alpha)
    return best_a


def maps_from(splits: dict[str, list[dict]], fn: Callable[[dict], float]) -> dict[str, dict[str, float]]:
    return {
        name: {row_key(row): fn(row) for row in rows}
        for name, rows in splits.items()
    }


def evaluate_policies(
    name: str,
    splits: dict[str, list[dict]],
) -> dict[str, Any]:
    train = splits["train"]
    raw = {
        split: {row_key(row): float(row["inclusion_ratio_estimated"]) for row in rows}
        for split, rows in splits.items()
    }
    edges4, mult4 = fit_bins(train, bins=4, l2=0.0)
    edges_reg, mult_reg = fit_bins(train, bins=4, l2=0.12)
    edges_con, mult_con = fit_bins(train, bins=4, l2=0.0, lo=0.80, hi=1.15)
    edges3, mult3 = fit_bins(train, bins=3, l2=0.08)
    edges_logo, mult_logo = logo_bins(train, bins=4, l2=0.08)
    iso = fit_isotonic(train)
    alpha = shrink_alpha(train)

    policies = {
        "raw": raw,
        "shrink": maps_from(
            splits,
            lambda row, a=alpha: clip_ratio(
                1.0 + a * (float(row["inclusion_ratio_estimated"]) - 1.0)
            ),
        ),
        "bins4": maps_from(
            splits,
            lambda row, e=edges4, m=mult4: apply_bins(
                float(row["inclusion_ratio_estimated"]), e, m
            ),
        ),
        "bins4_regularized": maps_from(
            splits,
            lambda row, e=edges_reg, m=mult_reg: apply_bins(
                float(row["inclusion_ratio_estimated"]), e, m
            ),
        ),
        "bins4_conservative": maps_from(
            splits,
            lambda row, e=edges_con, m=mult_con: apply_bins(
                float(row["inclusion_ratio_estimated"]), e, m
            ),
        ),
        "bins3_regularized": maps_from(
            splits,
            lambda row, e=edges3, m=mult3: apply_bins(
                float(row["inclusion_ratio_estimated"]), e, m
            ),
        ),
        "bins4_logo": maps_from(
            splits,
            lambda row, e=edges_logo, m=mult_logo: apply_bins(
                float(row["inclusion_ratio_estimated"]), e, m
            ),
        ),
        "isotonic": maps_from(
            splits,
            lambda row, model=iso: clip_ratio(
                float(model.predict([float(row["inclusion_ratio_estimated"])])[0])
            ),
        ),
    }
    leaderboard = []
    for policy, mapping in policies.items():
        leaderboard.append(
            {
                "policy": policy,
                "validation": summarize(apply_pred(splits["val"], mapping["val"])),
                "test_after_freeze": summarize(apply_pred(splits["test"], mapping["test"])),
            }
        )
    by_median = sorted(leaderboard, key=lambda row: selection_key(row["validation"]))
    by_geometry = sorted(
        leaderboard,
        key=lambda row: (
            -int(row["validation"]["geometries_passing"]),
            float(row["validation"]["median"]),
            -float(row["validation"]["pass_rate"]),
        ),
    )
    return {
        "method": name,
        "raw": {
            "train": summarize(splits["train"]),
            "validation": summarize(splits["val"]),
            "test": summarize(splits["test"]),
        },
        "fit": {
            "shrink_alpha": alpha,
            "bins4": {"edges": edges4, "multipliers": mult4},
            "bins4_regularized": {"edges": edges_reg, "multipliers": mult_reg},
            "bins4_conservative": {"edges": edges_con, "multipliers": mult_con},
            "bins3_regularized": {"edges": edges3, "multipliers": mult3},
            "bins4_logo": {"edges": edges_logo, "multipliers": mult_logo},
        },
        "selected_by_val_median": by_median[0]["policy"],
        "selected_by_val_geometries": by_geometry[0]["policy"],
        "leaderboard": by_median,
        "policies": policies,
    }


def compose_hybrid(
    e_rows: list[dict],
    ratio_map: dict[str, float],
    radius_rows: list[dict],
) -> list[dict]:
    radius = {row_key(row): row for row in radius_rows}
    out = []
    for row in e_rows:
        key = row_key(row)
        if key not in ratio_map or key not in radius:
            continue
        new = dict(row)
        true_ratio = float(row["inclusion_ratio_true"])
        pred_ratio = clip_ratio(ratio_map[key])
        new["inclusion_ratio_estimated"] = pred_ratio
        new["inclusion_ratio_relative_error"] = abs(pred_ratio / true_ratio - 1.0)
        rad = radius[key]
        new["radius_fraction_estimated"] = rad.get("radius_fraction_estimated")
        new["radius_relative_error"] = rad.get("radius_relative_error")
        out.append(new)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gnn = {split: load_gnn(split) for split in ("train", "val", "test")}
    print("loaded GNN seed 2028 dumps", flush=True)

    cache = OUT / "baseline_records.json"
    if cache.exists():
        baseline = json.loads(cache.read_text(encoding="utf-8"))
        print("loaded cached baseline records", flush=True)
    else:
        artifact = joblib.load(ARTIFACT)
        features_by_split = {
            split: extract_split_features(split) for split in ("train", "val", "test")
        }
        baseline = {}
        for family in ("ridge", "pca_ridge"):
            baseline[family] = {
                split: baseline_records_from_features(
                    family, artifact, features_by_split[split][0], features_by_split[split][1]
                )
                for split in ("train", "val", "test")
            }
            baseline[family]["test"] = overlay_benchmark_test(
                baseline[family]["test"], family
            )
        cache.write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    methods = {
        "mesh_gnn_seed2028": gnn,
        "ridge": baseline["ridge"],
        "pca_ridge": baseline["pca_ridge"],
    }
    reports = {}
    selected_maps = {}
    for name, splits in methods.items():
        print(f"calibrating {name}", flush=True)
        report = evaluate_policies(name, splits)
        selected = report["selected_by_val_geometries"]
        selected_maps[name] = report["policies"][selected]
        slim = {key: value for key, value in report.items() if key != "policies"}
        reports[name] = slim
        print(
            json.dumps(
                {
                    "method": name,
                    "selected": selected,
                    "selection_rule": "validation_geometry_pass_then_median",
                    "val": next(
                        row["validation"]
                        for row in slim["leaderboard"]
                        if row["policy"] == selected
                    ),
                    "test_after_freeze": next(
                        row["test_after_freeze"]
                        for row in slim["leaderboard"]
                        if row["policy"] == selected
                    ),
                    "raw_val": slim["raw"]["validation"]["median"],
                    "raw_test": slim["raw"]["test"]["median"],
                },
                indent=2,
            ),
            flush=True,
        )

    e_val = load_seed2027("val")
    e_test = load_seed2027("test")
    gnn_policy = reports["mesh_gnn_seed2028"]["selected_by_val_geometries"]
    gnn_maps = selected_maps["mesh_gnn_seed2028"]
    hybrid_val = compose_hybrid(e_val, gnn_maps["val"], baseline["pca_ridge"]["val"])
    hybrid_test = compose_hybrid(e_test, gnn_maps["test"], baseline["pca_ridge"]["test"])
    hybrid_raw_val = compose_hybrid(
        e_val,
        {row_key(row): float(row["inclusion_ratio_estimated"]) for row in gnn["val"]},
        baseline["pca_ridge"]["val"],
    )
    hybrid_raw_test = compose_hybrid(
        e_test,
        {row_key(row): float(row["inclusion_ratio_estimated"]) for row in gnn["test"]},
        baseline["pca_ridge"]["test"],
    )
    summary = {
        "schema_version": 1,
        "gate": GATE,
        "protocol": (
            "identical train-fit / validation-select / test-after-freeze "
            "procedure for MeshGNN, Ridge, and PCA-Ridge"
        ),
        "methods": reports,
        "hybrid": {
            "components": {
                "E_center": "MeshGNN seed 2027",
                "ratio": f"MeshGNN seed 2028 + {gnn_policy}",
                "radius": "PCA-Ridge",
            },
            "raw_ratio": {
                "validation": summarize(hybrid_raw_val),
                "test_after_freeze": summarize(hybrid_raw_test),
            },
            "calibrated_ratio": {
                "validation": summarize(hybrid_val),
                "test_after_freeze": summarize(hybrid_test),
            },
        },
    }
    (OUT / "selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "hybrid_raw_val": summary["hybrid"]["raw_ratio"]["validation"],
                "hybrid_cal_val": summary["hybrid"]["calibrated_ratio"]["validation"],
                "hybrid_raw_test": summary["hybrid"]["raw_ratio"]["test_after_freeze"],
                "hybrid_cal_test": summary["hybrid"]["calibrated_ratio"]["test_after_freeze"],
                "official_rule": "validation_geometry_pass_then_median",
                "winners": {
                    name: report["selected_by_val_geometries"]
                    for name, report in reports.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
