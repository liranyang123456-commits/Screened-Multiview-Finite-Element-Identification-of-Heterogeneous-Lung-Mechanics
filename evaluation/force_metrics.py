"""Force-regression metrics with recording-level bootstrap intervals."""
from __future__ import annotations

import numpy as np


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - target
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    force_range = float(target.max() - target.min())
    nrmse = rmse / force_range if force_range > 0 else float("nan")
    denominator = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - float(np.sum(error**2)) / denominator if denominator > 0 else float("nan")
    if target.std() > 0 and prediction.std() > 0:
        pearson = float(np.corrcoef(target, prediction)[0, 1])
    else:
        pearson = float("nan")
    return {"mae_n": mae, "rmse_n": rmse, "nrmse": nrmse, "r2": r2, "pearson_r": pearson}


def recording_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    recording: np.ndarray,
) -> list[dict[str, float | int]]:
    rows = []
    for recording_id in sorted(set(np.asarray(recording, dtype=int).tolist())):
        mask = np.asarray(recording, dtype=int) == recording_id
        rows.append(
            {
                "recording": recording_id,
                "n": int(mask.sum()),
                **regression_metrics(target[mask], prediction[mask]),
            }
        )
    return rows


def bootstrap_recording_ci(
    rows: list[dict[str, float | int]],
    metric: str,
    seed: int = 2026,
    samples: int = 10_000,
) -> list[float]:
    values = np.asarray([float(row[metric]) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    estimates = np.median(
        rng.choice(values, size=(samples, len(values)), replace=True), axis=1
    )
    return np.percentile(estimates, [2.5, 97.5]).tolist()
