"""Leakage-free node-response calibration for patient material inference.

Model selection is performed exclusively by fixed five-fold cross-validation
inside the training split.  The complete validation split is reserved for
split-conformal calibration, and the test split is evaluated exactly once.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.sim_lung_graph import SimLungGraphDataset  # noqa: E402


def features(graph: dict) -> np.ndarray:
    dynamic = graph["dynamic_seq"].numpy()
    visible = dynamic[..., 4] > 0.5
    flow = np.linalg.norm(dynamic[..., :3], axis=-1) * visible
    count = visible.sum(axis=2).clip(1)
    temporal_mean = flow.sum(axis=2) / count
    temporal_maximum = flow.max(axis=2)
    return np.concatenate((temporal_mean.ravel(), temporal_maximum.ravel()))


def arrays(dataset: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = SimLungGraphDataset(dataset, split=split)
    X, targets, heterogeneous = [], [], []
    for graph in rows:
        label = graph["labels"]
        X.append(features(graph))
        targets.append(
            [
                float(label["log_E_background"]),
                float(label["log_ratio"]),
                *label["center_fraction"].tolist(),
                float(label["radius_fraction"]),
            ]
        )
        heterogeneous.append(bool(label["heterogeneous"]))
    return np.asarray(X), np.asarray(targets), np.asarray(heterogeneous)


def errors(
    prediction: np.ndarray, target: np.ndarray, heterogeneous: np.ndarray
) -> dict[str, float]:
    return {
        "E_background_median_relative_error": float(
            np.median(np.abs(np.exp(prediction[:, 0] - target[:, 0]) - 1.0))
        ),
        "inclusion_ratio_median_relative_error": float(
            np.median(np.abs(np.exp(prediction[:, 1] - target[:, 1]) - 1.0))
        ),
        "center_error_normalized_median": float(
            np.median(
                np.linalg.norm(
                    prediction[heterogeneous, 2:5] - target[heterogeneous, 2:5],
                    axis=1,
                )
            )
        ),
        "radius_relative_error_median": float(
            np.median(
                np.abs(
                    prediction[heterogeneous, 5] / target[heterogeneous, 5] - 1.0
                )
            )
        ),
    }


SEED = 2026
FOLDS = 5
TARGET_NAMES = (
    "log_E_background",
    "log_ratio",
    "center_x",
    "center_y",
    "center_z",
    "radius_fraction",
)


def pca_ridge(components: int, alpha: float):
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=components, random_state=SEED),
        Ridge(alpha=alpha),
    )


def main_candidates() -> list[dict[str, Any]]:
    return [
        {
            "name": "pca_ridge",
            "params": {"components": components, "alpha": alpha},
            "estimator": pca_ridge(components, alpha),
        }
        for components in (16, 32, 64, 80, 96)
        for alpha in (0.1, 1.0, 10.0, 100.0)
    ]


def radius_candidates() -> list[dict[str, Any]]:
    candidates = [
        {
            "name": "pca_ridge",
            "params": {"components": components, "alpha": alpha},
            "estimator": pca_ridge(components, alpha),
        }
        for components in (16, 32, 64, 80, 96)
        for alpha in (0.1, 1.0, 10.0, 100.0)
    ]
    candidates.extend(
        {
            "name": "pls",
            "params": {"components": components},
            "estimator": make_pipeline(
                StandardScaler(),
                PLSRegression(n_components=components, scale=False, max_iter=1000),
            ),
        }
        for components in (2, 4, 8, 12, 16)
    )
    candidates.extend(
        {
            "name": "extra_trees",
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
    )
    return candidates


def _safe_components(estimator: Any, train_size: int, feature_count: int) -> Any:
    """Keep latent dimensions valid in every training fold."""
    estimator = clone(estimator)
    limit = min(train_size - 1, feature_count)
    params = estimator.get_params()
    if "pca__n_components" in params:
        estimator.set_params(pca__n_components=min(params["pca__n_components"], limit))
    if "plsregression__n_components" in params:
        estimator.set_params(
            plsregression__n_components=min(
                params["plsregression__n_components"], limit
            )
        )
    return estimator


def _score(metric: dict[str, float]) -> tuple[float, float]:
    normalized = (
        metric["E_background_median_relative_error"] / 0.15,
        metric["inclusion_ratio_median_relative_error"] / 0.25,
        metric["center_error_normalized_median"] / 0.10,
        metric["radius_relative_error_median"] / 0.12,
    )
    return max(normalized), sum(normalized)


def select_main_model(
    X: np.ndarray, target: np.ndarray, heterogeneous: np.ndarray
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    folds = list(KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(X))
    leaderboard = []
    best: tuple[tuple[float, float], dict[str, Any]] | None = None
    for candidate in main_candidates():
        out_of_fold = np.empty_like(target)
        for train_indices, held_out_indices in folds:
            estimator = _safe_components(
                candidate["estimator"], len(train_indices), X.shape[1]
            )
            estimator.fit(X[train_indices], target[train_indices])
            out_of_fold[held_out_indices] = estimator.predict(X[held_out_indices])
        metric = errors(out_of_fold, target, heterogeneous)
        score = _score(metric)
        row = {
            "name": candidate["name"],
            "params": candidate["params"],
            "score": list(score),
            "out_of_fold_metrics": metric,
        }
        leaderboard.append(row)
        if best is None or score < best[0]:
            best = (score, candidate)
    assert best is not None
    selected = best[1]
    estimator = _safe_components(selected["estimator"], len(X), X.shape[1])
    estimator.fit(X, target)
    return estimator, {"name": selected["name"], "params": selected["params"]}, leaderboard


def select_radius_model(
    X: np.ndarray, target: np.ndarray, heterogeneous: np.ndarray
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    """Select a radius-only head from training OOF predictions.

    Radius is identifiable only for heterogeneous patients, so both fitting and
    scoring of this head use heterogeneous training rows.  Fold assignment is
    nevertheless fixed before examining any candidate.
    """
    indices = np.flatnonzero(heterogeneous)
    X_radius, y_radius = X[indices], target[indices, 5]
    folds = list(
        KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(X_radius)
    )
    leaderboard = []
    best: tuple[float, dict[str, Any]] | None = None
    for candidate in radius_candidates():
        out_of_fold = np.empty_like(y_radius)
        for train_indices, held_out_indices in folds:
            estimator = _safe_components(
                candidate["estimator"], len(train_indices), X.shape[1]
            )
            estimator.fit(X_radius[train_indices], y_radius[train_indices])
            out_of_fold[held_out_indices] = np.asarray(
                estimator.predict(X_radius[held_out_indices])
            ).reshape(-1)
        relative_errors = np.abs(out_of_fold / y_radius - 1.0)
        score = float(np.median(relative_errors))
        row = {
            "name": candidate["name"],
            "params": candidate["params"],
            "out_of_fold_radius_relative_error_median": score,
        }
        leaderboard.append(row)
        if best is None or score < best[0]:
            best = (score, candidate)
    assert best is not None
    selected = best[1]
    estimator = _safe_components(selected["estimator"], len(X_radius), X.shape[1])
    estimator.fit(X_radius, y_radius)
    return estimator, {"name": selected["name"], "params": selected["params"]}, leaderboard


def predict(main_estimator: Any, radius_estimator: Any, X: np.ndarray) -> np.ndarray:
    prediction = np.asarray(main_estimator.predict(X))
    prediction[:, 5] = np.asarray(radius_estimator.predict(X)).reshape(-1)
    return prediction


def conformal_quantile(residuals: np.ndarray, level: float = 0.9) -> float:
    """Finite-sample split-conformal absolute-residual quantile."""
    residuals = np.asarray(residuals, dtype=float)
    rank = min(math.ceil((len(residuals) + 1) * level), len(residuals))
    return float(np.partition(residuals, rank - 1)[rank - 1])


def conformal_metrics(
    target: np.ndarray, prediction: np.ndarray, half_width: float
) -> dict[str, Any]:
    covered = np.abs(np.asarray(target) - np.asarray(prediction)) <= half_width
    return {
        "coverage": {"90": float(np.mean(covered))},
        "mean_interval_width": {"90": float(2.0 * half_width)},
        "half_width": float(half_width),
        "calibration": "validation_split_conformal_absolute_residual",
    }


def prediction_records(
    dataset: Path,
    split: str,
    prediction: np.ndarray,
    target: np.ndarray,
    conformal_half_widths: np.ndarray,
    prior_log_stds: np.ndarray,
) -> list[dict]:
    graphs = SimLungGraphDataset(dataset, split=split)
    records = []
    for index, (estimate, truth) in enumerate(
        zip(prediction, target, strict=True)
    ):
        log_E_lower = float(estimate[0] - conformal_half_widths[0])
        log_E_upper = float(estimate[0] + conformal_half_widths[0])
        log_ratio_lower = float(estimate[1] - conformal_half_widths[1])
        log_ratio_upper = float(estimate[1] + conformal_half_widths[1])
        records.append({
            "patient_id": graphs[index]["patient_id"],
            "patient_index": index,
            "E_background_true": float(np.exp(truth[0])),
            "E_background_estimated": float(np.exp(estimate[0])),
            "inclusion_ratio_true": float(np.exp(truth[1])),
            "inclusion_ratio_estimated": float(np.exp(estimate[1])),
            "log_E_std": float(prior_log_stds[0]),
            "log_ratio_std": float(prior_log_stds[1]),
            "log_E_interval_90": [log_E_lower, log_E_upper],
            "E_background_interval_90": [
                float(np.exp(log_E_lower)),
                float(np.exp(log_E_upper)),
            ],
            "log_E_covered_90": bool(log_E_lower <= truth[0] <= log_E_upper),
            "log_ratio_interval_90": [log_ratio_lower, log_ratio_upper],
            "inclusion_ratio_interval_90": [
                float(np.exp(log_ratio_lower)),
                float(np.exp(log_ratio_upper)),
            ],
            "log_ratio_covered_90": bool(
                log_ratio_lower <= truth[1] <= log_ratio_upper
            ),
            "center_fraction_true": truth[2:5].tolist(),
            "center_fraction_estimated": estimate[2:5].tolist(),
            "radius_fraction_true": float(truth[5]),
            "radius_fraction_estimated": float(estimate[5]),
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    train = arrays(args.dataset, "train")
    validation = arrays(args.dataset, "val")
    test = arrays(args.dataset, "test")
    estimator, selected_main, main_leaderboard = select_main_model(*train)
    radius_estimator, selected_radius, radius_leaderboard = select_radius_model(*train)

    validation_prediction = predict(estimator, radius_estimator, validation[0])
    validation_metrics = errors(
        validation_prediction, validation[1], validation[2]
    )
    conformal_half_widths = np.asarray(
        [
            conformal_quantile(
                np.abs(validation_prediction[:, index] - validation[1][:, index])
            )
            for index in (0, 1)
        ]
    )
    prior_log_stds = np.std(
        validation_prediction[:, :2] - validation[1][:, :2],
        axis=0,
        ddof=1,
    )
    validation_metrics["log_E_uncertainty"] = conformal_metrics(
        validation[1][:, 0], validation_prediction[:, 0], conformal_half_widths[0]
    )
    validation_metrics["log_ratio_uncertainty"] = conformal_metrics(
        validation[1][:, 1], validation_prediction[:, 1], conformal_half_widths[1]
    )

    # The test split is touched only after every model and interval parameter is frozen.
    test_prediction = predict(estimator, radius_estimator, test[0])
    test_metrics = errors(test_prediction, test[1], test[2])
    test_metrics["log_E_uncertainty"] = conformal_metrics(
        test[1][:, 0], test_prediction[:, 0], conformal_half_widths[0]
    )
    test_metrics["log_ratio_uncertainty"] = conformal_metrics(
        test[1][:, 1], test_prediction[:, 1], conformal_half_widths[1]
    )
    records = prediction_records(
        args.dataset,
        "test",
        test_prediction,
        test[1],
        conformal_half_widths,
        prior_log_stds,
    )
    validation_records = prediction_records(
        args.dataset,
        "val",
        validation_prediction,
        validation[1],
        conformal_half_widths,
        prior_log_stds,
    )
    result = {
        "model": "train_cv_selected_node_response_with_radius_head",
        "protocol": {
            "selection": (
                "Fixed 5-fold shuffled KFold on train only; seed 2026. "
                "Main head minimizes lexicographic (worst normalized gate metric, "
                "sum normalized gate metrics). Radius head minimizes OOF median "
                "relative error among heterogeneous train patients."
            ),
            "uncertainty": (
                "After train-only model selection, the full validation split "
                "sets finite-sample 90% split-conformal quantiles and global "
                "log-residual standard deviations used by the FEM material prior."
            ),
            "test_policy": "Test evaluated once after model and intervals were frozen.",
            "folds": FOLDS,
            "seed": SEED,
            "split_counts": {
                "train": len(train[0]),
                "validation": len(validation[0]),
                "test": len(test[0]),
            },
        },
        "selected_main_model": selected_main,
        "selected_radius_model": selected_radius,
        "validation_residual_log_std_for_fem_prior": {
            "log_E_background": float(prior_log_stds[0]),
            "log_ratio": float(prior_log_stds[1]),
        },
        "train_cv": {
            "main_leaderboard": main_leaderboard,
            "radius_leaderboard": radius_leaderboard,
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "test_patient_count": len(test_prediction),
        "records": records,
        "validation_records": validation_records,
        "evidence_boundary": (
            "Point accuracy and uncertainty coverage are measured on the frozen "
            "synthetic 250-patient multiview cohort. After train-only selection, "
            "validation calibrates interval widths and the global FEM material-prior "
            "residual scale; test labels never select models or hyperparameters. These "
            "results do not establish real-patient material accuracy."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    artifact = {
        "main_estimator": estimator,
        "radius_estimator": radius_estimator,
        "selected_main_model": selected_main,
        "selected_radius_model": selected_radius,
        "conformal_half_widths": {
            TARGET_NAMES[0]: float(conformal_half_widths[0]),
            TARGET_NAMES[1]: float(conformal_half_widths[1]),
        },
        "validation_residual_log_std_for_fem_prior": {
            TARGET_NAMES[0]: float(prior_log_stds[0]),
            TARGET_NAMES[1]: float(prior_log_stds[1]),
        },
        "feature_definition": "visible_flow_temporal_mean_and_max",
        "target_names": TARGET_NAMES,
        "seed": SEED,
        "folds": FOLDS,
    }
    joblib.dump(artifact, args.out.with_suffix(".joblib"))
    summary = {key: value for key, value in result.items() if key not in {
        "records", "validation_records", "train_cv"
    }}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
