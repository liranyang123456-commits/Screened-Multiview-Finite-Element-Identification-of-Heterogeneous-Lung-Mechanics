from __future__ import annotations

import numpy as np

from experiments.train_lung_response_calibrator import (
    conformal_metrics,
    conformal_quantile,
    predict,
)


class _FixedEstimator:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.values[: len(features)]


def test_conformal_quantile_uses_finite_sample_higher_rank() -> None:
    residuals = np.arange(1.0, 11.0)
    assert conformal_quantile(residuals, level=0.9) == 10.0


def test_conformal_metrics_expose_publication_gate_shape() -> None:
    metrics = conformal_metrics(
        np.asarray([0.0, 1.0]),
        np.asarray([0.1, 1.3]),
        half_width=0.2,
    )
    assert metrics["coverage"]["90"] == 0.5
    assert metrics["mean_interval_width"]["90"] == 0.4


def test_radius_head_replaces_only_radius_prediction() -> None:
    main = np.arange(12.0).reshape(2, 6)
    prediction = predict(
        _FixedEstimator(main),
        _FixedEstimator(np.asarray([0.25, 0.5])),
        np.zeros((2, 3)),
    )
    np.testing.assert_array_equal(prediction[:, :5], main[:, :5])
    np.testing.assert_array_equal(prediction[:, 5], np.asarray([0.25, 0.5]))
