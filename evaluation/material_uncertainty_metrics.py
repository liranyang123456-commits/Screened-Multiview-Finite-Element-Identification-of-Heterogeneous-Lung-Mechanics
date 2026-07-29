"""Uncertainty and patient-level bootstrap metrics for material regression."""
from __future__ import annotations

import math

import numpy as np
import torch


def gaussian_interval_metrics(
    targets: np.ndarray,
    means: np.ndarray,
    standard_deviations: np.ndarray,
    levels: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
) -> dict:
    targets = np.asarray(targets, dtype=float)
    means = np.asarray(means, dtype=float)
    standard_deviations = np.maximum(
        np.asarray(standard_deviations, dtype=float), 1e-8
    )
    normal = torch.distributions.Normal(0.0, 1.0)
    coverage, widths = {}, {}
    calibration_terms = []
    for level in levels:
        quantile = float(
            normal.icdf(torch.tensor((1.0 + level) / 2.0, dtype=torch.float64))
        )
        half_width = quantile * standard_deviations
        empirical = float(np.mean(np.abs(targets - means) <= half_width))
        key = f"{int(round(100 * level))}"
        coverage[key] = empirical
        widths[key] = float(np.mean(2.0 * half_width))
        calibration_terms.append(abs(empirical - level))
    variance = standard_deviations**2
    nll = 0.5 * (
        np.log(2.0 * math.pi * variance) + (targets - means) ** 2 / variance
    )
    return {
        "gaussian_nll": float(np.mean(nll)),
        "coverage": coverage,
        "mean_interval_width": widths,
        "mean_absolute_calibration_error": float(np.mean(calibration_terms)),
    }


def bootstrap_median_ci(
    values: list[float] | np.ndarray,
    *,
    seed: int = 2026,
    draws: int = 5000,
) -> list[float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return [float("nan"), float("nan")]
    generator = np.random.default_rng(seed)
    medians = np.asarray(
        [
            np.median(generator.choice(array, size=len(array), replace=True))
            for _ in range(draws)
        ]
    )
    return [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))]


def binary_calibration(targets: np.ndarray, probabilities: np.ndarray) -> dict:
    targets = np.asarray(targets, dtype=float)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    predictions = probabilities >= 0.5
    return {
        "accuracy": float(np.mean(predictions == targets.astype(bool))),
        "brier_score": float(np.mean((probabilities - targets) ** 2)),
        "binary_nll": float(
            -np.mean(
                targets * np.log(probabilities)
                + (1.0 - targets) * np.log(1.0 - probabilities)
            )
        ),
    }


def gaussian_std_calibration_scale(
    targets: np.ndarray,
    means: np.ndarray,
    standard_deviations: np.ndarray,
    *,
    target_level: float = 0.9,
) -> float:
    """Return a validation-only scalar making a Gaussian interval calibrated."""
    standard_deviations = np.maximum(
        np.asarray(standard_deviations, dtype=float), 1e-8
    )
    normalized_error = np.abs(
        np.asarray(targets, dtype=float) - np.asarray(means, dtype=float)
    ) / standard_deviations
    normal = torch.distributions.Normal(0.0, 1.0)
    gaussian_quantile = float(
        normal.icdf(
            torch.tensor((1.0 + target_level) / 2.0, dtype=torch.float64)
        )
    )
    # Small validation splits can look artificially easy. Never shrink a learned
    # posterior from this post-hoc step; only inflate intervals when warranted.
    return max(float(np.quantile(normalized_error, target_level) / gaussian_quantile), 1.0)
