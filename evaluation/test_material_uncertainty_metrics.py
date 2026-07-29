from __future__ import annotations

import numpy as np

from evaluation.material_uncertainty_metrics import (
    binary_calibration,
    bootstrap_median_ci,
    gaussian_interval_metrics,
    gaussian_std_calibration_scale,
)


def test_uncertainty_metrics_reward_calibrated_predictions() -> None:
    targets = np.linspace(-1.0, 1.0, 101)
    metrics = gaussian_interval_metrics(targets, targets, np.full_like(targets, 0.2))
    assert metrics["gaussian_nll"] < 0.0
    assert metrics["coverage"]["90"] == 1.0
    assert metrics["mean_interval_width"]["95"] > metrics["mean_interval_width"]["50"]


def test_bootstrap_and_binary_metrics_are_deterministic() -> None:
    first = bootstrap_median_ci([0.1, 0.2, 0.3], draws=100)
    second = bootstrap_median_ci([0.1, 0.2, 0.3], draws=100)
    assert first == second
    classification = binary_calibration(
        np.asarray([0, 1]), np.asarray([0.1, 0.9])
    )
    assert classification["accuracy"] == 1.0
    assert classification["brier_score"] < 0.02


def test_small_validation_set_never_shrinks_posterior() -> None:
    scale = gaussian_std_calibration_scale(
        np.asarray([0.0, 0.1]),
        np.asarray([0.0, 0.1]),
        np.asarray([1.0, 1.0]),
    )
    assert scale == 1.0
