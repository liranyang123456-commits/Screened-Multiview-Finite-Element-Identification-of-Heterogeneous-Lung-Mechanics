"""Fair train-selected response-regression baselines for the frozen lung cohort.

Every learned family uses the same temporal mean/maximum track summaries, the
same patient split, and five-fold train-only model selection.  The held-out test
split is scored only after each family and its radius head have been fitted.
These comparators are implementations in the present pipeline, not claimed
reproductions of any external publication.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.sim_lung_graph import SimLungGraphDataset  # noqa: E402
from experiments.train_lung_response_calibrator import (  # noqa: E402
    FOLDS,
    SEED,
    _safe_components,
    _score,
    arrays,
    errors,
    pca_ridge,
)


def family_candidates() -> dict[str, list[dict[str, Any]]]:
    """Prespecified model families and within-family grids."""
    return {
        "ridge": [
            {
                "params": {"alpha": alpha},
                "estimator": make_pipeline(StandardScaler(), Ridge(alpha=alpha)),
            }
            for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0)
        ],
        "pls": [
            {
                "params": {"components": components},
                "estimator": make_pipeline(
                    StandardScaler(),
                    PLSRegression(
                        n_components=components,
                        scale=False,
                        max_iter=2000,
                    ),
                ),
            }
            for components in (2, 4, 8, 12, 16)
        ],
        "extra_trees": [
            {
                "params": {
                    "max_features": max_features,
                    "min_samples_leaf": min_samples_leaf,
                },
                "estimator": ExtraTreesRegressor(
                    n_estimators=500,
                    max_features=max_features,
                    min_samples_leaf=min_samples_leaf,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            }
            for max_features in (0.5, 1.0)
            for min_samples_leaf in (1, 2, 4)
        ],
        "pca_ridge": [
            {
                "params": {"components": components, "alpha": alpha},
                "estimator": pca_ridge(components, alpha),
            }
            for components in (16, 32, 64, 80, 96)
            for alpha in (0.1, 1.0, 10.0, 100.0)
        ],
    }


def _prediction(estimator: Any, X: np.ndarray) -> np.ndarray:
    value = np.asarray(estimator.predict(X))
    return value.reshape(len(X), -1)


def select_main(
    candidates: list[dict[str, Any]],
    X: np.ndarray,
    target: np.ndarray,
    heterogeneous: np.ndarray,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    folds = list(KFold(FOLDS, shuffle=True, random_state=SEED).split(X))
    leaderboard: list[dict[str, Any]] = []
    best: tuple[tuple[float, float], dict[str, Any]] | None = None
    for candidate in candidates:
        out_of_fold = np.empty_like(target)
        for train_indices, held_out_indices in folds:
            estimator = _safe_components(
                candidate["estimator"], len(train_indices), X.shape[1]
            )
            estimator.fit(X[train_indices], target[train_indices])
            out_of_fold[held_out_indices] = _prediction(
                estimator, X[held_out_indices]
            )
        metric = errors(out_of_fold, target, heterogeneous)
        score = _score(metric)
        leaderboard.append(
            {
                "params": candidate["params"],
                "score": list(score),
                "out_of_fold_metrics": metric,
            }
        )
        if best is None or score < best[0]:
            best = (score, candidate)
    assert best is not None
    selected = best[1]
    estimator = _safe_components(selected["estimator"], len(X), X.shape[1])
    estimator.fit(X, target)
    return estimator, selected["params"], leaderboard


def select_radius(
    candidates: list[dict[str, Any]],
    X: np.ndarray,
    target: np.ndarray,
    heterogeneous: np.ndarray,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    indices = np.flatnonzero(heterogeneous)
    X_radius, y_radius = X[indices], target[indices, 5]
    folds = list(KFold(FOLDS, shuffle=True, random_state=SEED).split(X_radius))
    leaderboard: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any]] | None = None
    for candidate in candidates:
        out_of_fold = np.empty_like(y_radius)
        for train_indices, held_out_indices in folds:
            estimator = _safe_components(
                candidate["estimator"], len(train_indices), X.shape[1]
            )
            estimator.fit(X_radius[train_indices], y_radius[train_indices])
            out_of_fold[held_out_indices] = _prediction(
                estimator, X_radius[held_out_indices]
            ).reshape(-1)
        score = float(np.median(np.abs(out_of_fold / y_radius - 1.0)))
        leaderboard.append(
            {
                "params": candidate["params"],
                "out_of_fold_radius_relative_error_median": score,
            }
        )
        if best is None or score < best[0]:
            best = (score, candidate)
    assert best is not None
    selected = best[1]
    estimator = _safe_components(
        selected["estimator"], len(X_radius), X.shape[1]
    )
    estimator.fit(X_radius, y_radius)
    return estimator, selected["params"], leaderboard


def patient_errors(
    prediction: np.ndarray, target: np.ndarray, heterogeneous: np.ndarray
) -> dict[str, list[float]]:
    output = {
        "E_background_relative_error": np.abs(
            np.exp(prediction[:, 0] - target[:, 0]) - 1.0
        ).tolist(),
        "inclusion_ratio_relative_error": np.abs(
            np.exp(prediction[:, 1] - target[:, 1]) - 1.0
        ).tolist(),
        "center_error_normalized": np.linalg.norm(
            prediction[heterogeneous, 2:5] - target[heterogeneous, 2:5],
            axis=1,
        ).tolist(),
        "radius_relative_error": np.abs(
            prediction[heterogeneous, 5] / target[heterogeneous, 5] - 1.0
        ).tolist(),
    }
    return output


def bootstrap_ci(values: list[float], seed: int, replicates: int = 5000) -> list[float]:
    array = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    medians = np.median(
        generator.choice(array, size=(replicates, len(array)), replace=True),
        axis=1,
    )
    return np.quantile(medians, [0.025, 0.975]).tolist()


def paired_comparisons(
    methods: list[dict[str, Any]], reference_name: str = "pca_ridge"
) -> dict[str, Any]:
    """Patient-aligned reference-minus-comparator error differences."""
    reference = next(row for row in methods if row["name"] == reference_name)
    reference_records = {
        row["patient_id"]: row for row in reference["records"]
    }
    metric_names = (
        "E_background_relative_error",
        "inclusion_ratio_relative_error",
        "center_error_normalized",
        "radius_relative_error",
    )
    output: dict[str, Any] = {}
    for method_index, method in enumerate(methods):
        if method["name"] == reference_name:
            continue
        rows = []
        for record in method["records"]:
            reference_record = reference_records[record["patient_id"]]
            rows.append((reference_record, record))
        comparison = {}
        for metric_index, metric in enumerate(metric_names):
            differences = np.asarray(
                [
                    reference_record[metric] - comparator_record[metric]
                    for reference_record, comparator_record in rows
                    if metric in reference_record and metric in comparator_record
                ],
                dtype=float,
            )
            comparison[metric] = {
                "paired_median_difference_reference_minus_comparator": float(
                    np.median(differences)
                ),
                "bootstrap_95_ci": bootstrap_ci(
                    differences.tolist(),
                    SEED + 100 + 10 * method_index + metric_index,
                ),
                "reference_win_rate": float(np.mean(differences < 0.0)),
                "patient_count": int(len(differences)),
            }
        output[method["name"]] = comparison
    return output


def summarize(
    name: str,
    prediction: np.ndarray,
    target: np.ndarray,
    heterogeneous: np.ndarray,
    patient_ids: list[str],
    selected_main: dict[str, Any] | None,
    selected_radius: dict[str, Any] | None,
) -> dict[str, Any]:
    metric = errors(prediction, target, heterogeneous)
    per_patient = patient_errors(prediction, target, heterogeneous)
    intervals = {
        key: bootstrap_ci(value, SEED + index)
        for index, (key, value) in enumerate(per_patient.items())
    }
    records = []
    heterogeneous_index = 0
    for index, patient_id in enumerate(patient_ids):
        row = {
            "patient_id": patient_id,
            "E_background_relative_error": per_patient[
                "E_background_relative_error"
            ][index],
            "inclusion_ratio_relative_error": per_patient[
                "inclusion_ratio_relative_error"
            ][index],
            "heterogeneous": bool(heterogeneous[index]),
        }
        if heterogeneous[index]:
            row["center_error_normalized"] = per_patient[
                "center_error_normalized"
            ][heterogeneous_index]
            row["radius_relative_error"] = per_patient[
                "radius_relative_error"
            ][heterogeneous_index]
            heterogeneous_index += 1
        records.append(row)
    return {
        "name": name,
        "selected_main": selected_main,
        "selected_radius": selected_radius,
        "test_metrics": metric,
        "bootstrap_95_ci": intervals,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train = arrays(args.dataset, "train")
    test = arrays(args.dataset, "test")
    patient_ids = [
        row["patient_id"]
        for row in SimLungGraphDataset(args.dataset, split="test")
    ]
    results: list[dict[str, Any]] = []

    population = np.tile(np.mean(train[1], axis=0), (len(test[0]), 1))
    population[:, 5] = np.mean(train[1][train[2], 5])
    results.append(
        summarize(
            "training_population_prior",
            population,
            test[1],
            test[2],
            patient_ids,
            None,
            None,
        )
    )

    selection: dict[str, Any] = {}
    for family, candidates in family_candidates().items():
        main_estimator, main_params, main_board = select_main(candidates, *train)
        radius_estimator, radius_params, radius_board = select_radius(
            candidates, *train
        )
        prediction = _prediction(main_estimator, test[0])
        prediction[:, 5] = _prediction(radius_estimator, test[0]).reshape(-1)
        results.append(
            summarize(
                family,
                prediction,
                test[1],
                test[2],
                patient_ids,
                main_params,
                radius_params,
            )
        )
        selection[family] = {
            "main_leaderboard": main_board,
            "radius_leaderboard": radius_board,
        }

    output = {
        "schema_version": 1,
        "protocol": {
            "dataset": str(args.dataset),
            "split_counts": {"train": 150, "validation": 50, "test": 50},
            "features": "identical visible-flow temporal mean/maximum summaries",
            "selection": "five-fold shuffled train-only CV; seed 2026",
            "test_policy": (
                "Test labels score the train-selected models only. Comparator "
                "families were added post hoc and do not alter the frozen "
                "primary estimator."
            ),
            "comparison_scope": (
                "In-project implementations of standard regression families; "
                "not external-paper reproductions."
            ),
        },
        "methods": results,
        "paired_against_pca_ridge": paired_comparisons(results),
        "train_cv": selection,
        "evidence_boundary": (
            "All material targets and accuracy statistics are synthetic. "
            "No result in this file measures real-patient material accuracy."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                row["name"]: row["test_metrics"]
                for row in results
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
