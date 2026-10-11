#!/usr/bin/env python3
"""Validation-selected Ridge / PCA–Ridge ensemble on the frozen synthetic split."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.benchmark_lung_response_baselines import (  # noqa: E402
    _prediction,
    _score,
    bootstrap_ci,
    family_candidates,
    select_main,
    select_radius,
    summarize,
)
from experiments.train_lung_response_calibrator import arrays  # noqa: E402
from dataset.sim_lung_graph import SimLungGraphDataset  # noqa: E402


def blend(a: np.ndarray, b: np.ndarray, weight: float) -> np.ndarray:
    return (1.0 - weight) * a + weight * b


def main() -> None:
    dataset = ROOT / "dataset" / "sim_lung_ai_v2_multiview250"
    out = ROOT / "results" / "lung_mesh_ai_v2" / "ridge_pcaridge_ensemble.json"
    train = arrays(dataset, "train")
    val = arrays(dataset, "val")
    test = arrays(dataset, "test")
    families = family_candidates()
    ridge_main, ridge_main_p, _ = select_main(families["ridge"], *train)
    ridge_rad, ridge_rad_p, _ = select_radius(families["ridge"], *train)
    pca_main, pca_main_p, _ = select_main(families["pca_ridge"], *train)
    pca_rad, pca_rad_p, _ = select_radius(families["pca_ridge"], *train)

    def predict(split_x: np.ndarray, main, radius) -> np.ndarray:
        pred = _prediction(main, split_x)
        pred[:, 5] = _prediction(radius, split_x).reshape(-1)
        return pred

    ridge_val = predict(val[0], ridge_main, ridge_rad)
    pca_val = predict(val[0], pca_main, pca_rad)
    ridge_test = predict(test[0], ridge_main, ridge_rad)
    pca_test = predict(test[0], pca_main, pca_rad)

    weights = (0.0, 0.25, 0.5, 0.75, 1.0)
    blend_board = []
    best_blend = None
    for weight in weights:
        pred = blend(ridge_val, pca_val, weight)
        metric = {
            "E_background_median_relative_error": float(
                np.median(np.abs(np.exp(pred[:, 0] - val[1][:, 0]) - 1.0))
            ),
            "inclusion_ratio_median_relative_error": float(
                np.median(np.abs(np.exp(pred[:, 1] - val[1][:, 1]) - 1.0))
            ),
            "center_error_normalized_median": float(
                np.median(np.linalg.norm(pred[val[2], 2:5] - val[1][val[2], 2:5], axis=1))
            ),
            "radius_relative_error_median": float(
                np.median(np.abs(pred[val[2], 5] / val[1][val[2], 5] - 1.0))
            ),
        }
        score = _score(metric)
        blend_board.append({"pca_weight": weight, "score": list(score), "validation": metric})
        if best_blend is None or score < best_blend[0]:
            best_blend = (score, weight, metric)
    assert best_blend is not None

    # Per-endpoint pick on validation: 0 = Ridge, 1 = PCA–Ridge.
    endpoint_choice = {}
    for name, sl, use_het in (
        ("E", (0, 0), False),
        ("ratio", (1, 1), False),
        ("center", (2, 5), True),
        ("radius", (5, 6), True),
    ):
        if name == "center":
            ridge_err = float(np.median(np.linalg.norm(ridge_val[val[2], 2:5] - val[1][val[2], 2:5], axis=1)))
            pca_err = float(np.median(np.linalg.norm(pca_val[val[2], 2:5] - val[1][val[2], 2:5], axis=1)))
        elif name == "radius":
            ridge_err = float(np.median(np.abs(ridge_val[val[2], 5] / val[1][val[2], 5] - 1.0)))
            pca_err = float(np.median(np.abs(pca_val[val[2], 5] / val[1][val[2], 5] - 1.0)))
        else:
            col = sl[0]
            ridge_err = float(np.median(np.abs(np.exp(ridge_val[:, col] - val[1][:, col]) - 1.0)))
            pca_err = float(np.median(np.abs(np.exp(pca_val[:, col] - val[1][:, col]) - 1.0)))
        endpoint_choice[name] = {
            "ridge": ridge_err,
            "pca_ridge": pca_err,
            "selected": "pca_ridge" if pca_err < ridge_err else "ridge",
        }

    mixed_test = ridge_test.copy()
    if endpoint_choice["E"]["selected"] == "pca_ridge":
        mixed_test[:, 0] = pca_test[:, 0]
    if endpoint_choice["ratio"]["selected"] == "pca_ridge":
        mixed_test[:, 1] = pca_test[:, 1]
    if endpoint_choice["center"]["selected"] == "pca_ridge":
        mixed_test[:, 2:5] = pca_test[:, 2:5]
    if endpoint_choice["radius"]["selected"] == "pca_ridge":
        mixed_test[:, 5] = pca_test[:, 5]

    blended_test = blend(ridge_test, pca_test, best_blend[1])
    patient_ids = [row["patient_id"] for row in SimLungGraphDataset(dataset, split="test")]
    methods = [
        summarize("ridge", ridge_test, test[1], test[2], patient_ids, ridge_main_p, ridge_rad_p),
        summarize("pca_ridge", pca_test, test[1], test[2], patient_ids, pca_main_p, pca_rad_p),
        summarize(
            "blend_val_selected",
            blended_test,
            test[1],
            test[2],
            patient_ids,
            {"pca_weight": best_blend[1]},
            None,
        ),
        summarize(
            "endpoint_mix_val_selected",
            mixed_test,
            test[1],
            test[2],
            patient_ids,
            endpoint_choice,
            None,
        ),
    ]
    report = {
        "schema_version": 1,
        "selection": "validation-only blend weight and per-endpoint pick; test unused",
        "selected_blend_pca_weight": best_blend[1],
        "blend_leaderboard": blend_board,
        "endpoint_choice": endpoint_choice,
        "methods": [
            {
                "name": row["name"],
                "test_metrics": row["test_metrics"],
                "bootstrap_95_ci": row["bootstrap_95_ci"],
            }
            for row in methods
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["methods"], indent=2))


if __name__ == "__main__":
    main()
