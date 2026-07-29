"""External mechanics validation on the public ÉTS indentation phantom.

This experiment estimates the *modulus contrast*, not absolute Young's
modulus. Dense incremental ultrasound registration yields axial strain in the
known inclusion and surrounding background. Under the shared-stress
elastography approximation, E_bg / E_inc ~= strain_inc / strain_bg. The result
is compared with independent Bose ElectroForce compression-test measurements.

The absolute-E inverse problem is deliberately not reported because the public
files do not fully specify probe contact width, out-of-plane boundary
conditions, or Poisson's ratio. Inventing those quantities would make an
apparently quantitative FEM result non-auditable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.ets_indentation.loader import (  # noqa: E402
    DATASET_DOI,
    ETSIndentationDataset,
    MechanicalTruth,
)


RESULTS = ROOT / "results"
FIGURES = ROOT / "paper_tbme" / "figures"
BOOTSTRAP_SEED = 2026
N_BOOTSTRAP = 10_000


def _active_crop(image: np.ndarray) -> tuple[np.ndarray, slice]:
    column_energy = image.mean(axis=0)
    active = np.flatnonzero(column_energy > 1.0)
    if len(active) == 0:
        raise ValueError("No active ultrasound columns found")
    crop = slice(int(active[0]), int(active[-1]) + 1)
    return image[:, crop], crop


def _preprocess(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    return cv2.GaussianBlur(clahe.apply(image), (5, 5), 0.8)


def _cumulative_dis_flow(
    images: list[np.ndarray],
    gradient_iterations: int = 40,
) -> tuple[np.ndarray, float]:
    """Compose pairwise DIS flows on the first-frame (Lagrangian) grid."""
    height, width = images[0].shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x, map_y = grid_x.copy(), grid_y.copy()
    cumulative = np.zeros((height, width, 2), dtype=np.float32)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(0)
    dis.setGradientDescentIterations(gradient_iterations)
    dis.setVariationalRefinementIterations(10)

    previous = _preprocess(images[0])
    for image in images[1:]:
        current = _preprocess(image)
        flow = dis.calc(previous, current, None)
        sampled_x = cv2.remap(
            flow[..., 0], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        sampled_y = cv2.remap(
            flow[..., 1], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        cumulative[..., 0] += sampled_x
        cumulative[..., 1] += sampled_y
        map_x += sampled_x
        map_y += sampled_y
        previous = current

    warped = cv2.remap(
        images[-1],
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    # Both images are in uint8 intensity units. This residual is a registration
    # quality-control measure, not a photometric material metric.
    residual = float(np.mean(np.abs(warped.astype(np.float32) - images[0])) / 255.0)
    return cumulative, residual


def _strain_masks(
    height: int,
    width: int,
    inclusion_scale: float = 0.70,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference-space ROIs from the published 20-mm inclusion geometry.

    The background ROI is on the same central load path immediately below the
    inclusion. This series arrangement is the region where the shared-stress
    approximation is physically appropriate; lateral background is excluded
    because the finite probe creates a different stress path there.
    """
    yy, xx = np.mgrid[:height, :width]
    center_x = 0.50 * width
    center_y = 0.50 * height
    radius_x = 0.245 * width
    radius_y = (10.0 / 60.0) * height
    normalized_radius = (
        ((xx - center_x) / radius_x) ** 2
        + ((yy - center_y) / radius_y) ** 2
    )
    inclusion = normalized_radius <= inclusion_scale**2
    central_column = np.abs(xx - center_x) <= 0.55 * radius_x
    below_inclusion = (yy >= center_y + 1.05 * radius_y) & (
        yy <= center_y + 1.75 * radius_y
    )
    background = central_column & below_inclusion
    return inclusion, background


def _sequence_images(acquisition: int, end_fraction: float = 1.0):
    dataset = ETSIndentationDataset(acquisition=acquisition)
    raw_images = [
        np.rint(dataset[index]["image"][0] * 255.0).astype(np.uint8)
        for index in range(len(dataset))
    ]
    _, crop = _active_crop(raw_images[0])
    images = [image[:, crop] for image in raw_images]
    end_index = max(2, int(round((len(images) - 1) * end_fraction)) + 1)
    return dataset, images[:end_index]


def _ratio_from_flow(
    cumulative: np.ndarray,
    smooth_sigma: float = 3.0,
    inclusion_scale: float = 0.70,
) -> tuple[float, float, float]:
    axial_displacement = cv2.GaussianBlur(
        cumulative[..., 1],
        (0, 0),
        sigmaX=smooth_sigma,
        sigmaY=smooth_sigma,
    )
    compressive_strain = -np.gradient(axial_displacement, axis=0)
    inclusion_mask, background_mask = _strain_masks(
        *compressive_strain.shape,
        inclusion_scale=inclusion_scale,
    )

    strain_inclusion = float(np.median(compressive_strain[inclusion_mask]))
    strain_background = float(np.median(compressive_strain[background_mask]))
    if strain_inclusion <= 0 or strain_background <= 0:
        modulus_ratio = float("nan")
    else:
        modulus_ratio = strain_inclusion / strain_background
    return modulus_ratio, strain_inclusion, strain_background


def evaluate_sequence(
    acquisition: int,
    *,
    smooth_sigma: float = 3.0,
    inclusion_scale: float = 0.70,
    end_fraction: float = 1.0,
    gradient_iterations: int = 40,
) -> dict[str, float | int]:
    dataset, images = _sequence_images(acquisition, end_fraction=end_fraction)
    cumulative, residual = _cumulative_dis_flow(
        images,
        gradient_iterations=gradient_iterations,
    )
    modulus_ratio, strain_inclusion, strain_background = _ratio_from_flow(
        cumulative,
        smooth_sigma=smooth_sigma,
        inclusion_scale=inclusion_scale,
    )

    start_force = float(np.median([dataset[index]["force_n"] for index in range(5)]))
    final_index = len(images) - 1
    final_force = float(dataset[final_index]["force_n"])
    return {
        "acquisition": acquisition,
        "frames": len(images),
        "final_indentation_mm": float(dataset[final_index]["indentation_mm"]),
        "incremental_force_n": final_force - start_force,
        "inclusion_compressive_strain": strain_inclusion,
        "background_compressive_strain": strain_background,
        "estimated_modulus_ratio_bg_over_inc": modulus_ratio,
        "registration_residual": residual,
    }


def _bootstrap_median_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(values, size=(N_BOOTSTRAP, len(values)), replace=True)
    return tuple(np.percentile(np.median(samples, axis=1), [2.5, 97.5]).tolist())


def _write_outputs(rows: list[dict[str, float | int]]) -> dict[str, object]:
    truth = MechanicalTruth()
    measured_ratio = truth.background_mean_kpa / truth.inclusion_mean_kpa
    ratios = np.asarray(
        [row["estimated_modulus_ratio_bg_over_inc"] for row in rows], dtype=float
    )
    valid = np.isfinite(ratios)
    valid_ratios = ratios[valid]
    if len(valid_ratios) == 0:
        raise RuntimeError("No sequence produced a positive strain contrast")

    median_ratio = float(np.median(valid_ratios))
    ci = _bootstrap_median_ci(valid_ratios)
    relative_error = abs(median_ratio - measured_ratio) / measured_ratio * 100.0
    summary: dict[str, object] = {
        "dataset": DATASET_DOI,
        "validation_scope": "ultrasound-registration modulus contrast; mechanics only",
        "ground_truth": {
            "background_kpa": [truth.background_mean_kpa, truth.background_sd_kpa],
            "inclusion_kpa": [truth.inclusion_mean_kpa, truth.inclusion_sd_kpa],
            "modulus_ratio_bg_over_inc": measured_ratio,
            "specimens_per_region": truth.specimens_per_region,
        },
        "sequences": rows,
        "valid_sequences": int(valid.sum()),
        "median_estimated_ratio": median_ratio,
        "bootstrap_95_ci_of_median": list(ci),
        "relative_error_of_median_ratio_percent": relative_error,
        "limitations": [
            "This is a modulus-contrast validation, not absolute-E recovery.",
            "The shared-stress approximation and manually specified geometry-derived ROIs are used.",
            "The dataset is ultrasound, not endoscopic RGB; the optical renderer is not evaluated.",
        ],
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "ets_phantom_eval.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# ÉTS public-phantom mechanics validation",
        "",
        f"Dataset: {DATASET_DOI}",
        "",
        "| Acq. | Frames | Final indentation (mm) | Δ force (N) | "
        "Inclusion strain | Background strain | Estimated E_bg/E_inc | Reg. residual |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ratio = float(row["estimated_modulus_ratio_bg_over_inc"])
        ratio_text = f"{ratio:.3f}" if np.isfinite(ratio) else "invalid"
        lines.append(
            f"| {row['acquisition']} | {row['frames']} | "
            f"{row['final_indentation_mm']:.1f} | {row['incremental_force_n']:.3f} | "
            f"{row['inclusion_compressive_strain']:.4f} | "
            f"{row['background_compressive_strain']:.4f} | {ratio_text} | "
            f"{row['registration_residual']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"- Independent compression-test ratio: {measured_ratio:.3f} "
            f"(background {truth.background_mean_kpa:.2f}±{truth.background_sd_kpa:.2f} kPa; "
            f"inclusion {truth.inclusion_mean_kpa:.2f}±{truth.inclusion_sd_kpa:.2f} kPa; "
            f"n={truth.specimens_per_region} per region).",
            f"- Median image-derived ratio: {median_ratio:.3f}; bootstrap 95% CI "
            f"[{ci[0]:.3f}, {ci[1]:.3f}].",
            f"- Relative error of the median ratio: {relative_error:.1f}%.",
            "",
            "This external experiment validates measured stiffness contrast only. "
            "It does not recover absolute E and does not evaluate the endoscopic renderer.",
            "",
        ]
    )
    (RESULTS / "ets_phantom_eval.md").write_text("\n".join(lines), encoding="utf-8")

    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    acquisitions = np.arange(1, 7)
    ax.plot(acquisitions[valid], ratios[valid], "o-", color="#1f77b4", label="US registration")
    ax.axhline(measured_ratio, color="#d62728", linestyle="--", label="Compression-test GT")
    ax.set_xlabel("Acquisition sequence")
    ax.set_ylabel(r"$E_{\mathrm{bg}}/E_{\mathrm{inc}}$")
    ax.set_xticks(acquisitions)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig6_ets_phantom.png", dpi=300)
    plt.close(fig)
    return summary


def _write_sensitivity(
    baseline_rows: list[dict[str, float | int]],
) -> dict[str, object]:
    truth = MechanicalTruth()
    measured_ratio = truth.background_mean_kpa / truth.inclusion_mean_kpa
    configurations = [
        ("baseline", {}),
        ("roi_0.60", {"inclusion_scale": 0.60}),
        ("roi_0.80", {"inclusion_scale": 0.80}),
        ("smooth_2", {"smooth_sigma": 2.0}),
        ("smooth_4", {"smooth_sigma": 4.0}),
        ("end_0.80", {"end_fraction": 0.80}),
        ("end_0.90", {"end_fraction": 0.90}),
        ("iterations_20", {"gradient_iterations": 20}),
        ("iterations_60", {"gradient_iterations": 60}),
    ]
    rows = []
    for name, kwargs in configurations:
        sequence_rows = (
            baseline_rows
            if name == "baseline"
            else [
                evaluate_sequence(acquisition, **kwargs)
                for acquisition in range(1, 7)
            ]
        )
        ratios = np.asarray(
            [row["estimated_modulus_ratio_bg_over_inc"] for row in sequence_rows],
            dtype=float,
        )
        ratios = ratios[np.isfinite(ratios)]
        median = float(np.median(ratios))
        rows.append(
            {
                "configuration": name,
                "median_ratio": median,
                "relative_error_percent": abs(median - measured_ratio)
                / measured_ratio
                * 100.0,
                "bootstrap_95_ci": list(_bootstrap_median_ci(ratios)),
                "sequence_ratios": ratios.tolist(),
            }
        )
        print(f"ETS sensitivity {name}: median ratio={median:.3f}", flush=True)

    payload: dict[str, object] = {
        "ground_truth_ratio": measured_ratio,
        "configurations": rows,
        "median_ratio_range": [
            min(row["median_ratio"] for row in rows),
            max(row["median_ratio"] for row in rows),
        ],
        "relative_error_range_percent": [
            min(row["relative_error_percent"] for row in rows),
            max(row["relative_error_percent"] for row in rows),
        ],
    }
    (RESULTS / "ets_phantom_sensitivity.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# ÉTS sensitivity analysis",
        "",
        "| Configuration | Median E_bg/E_inc | Relative error | Bootstrap 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        ci = row["bootstrap_95_ci"]
        lines.append(
            f"| {row['configuration']} | {row['median_ratio']:.3f} | "
            f"{row['relative_error_percent']:.1f}% | [{ci[0]:.3f}, {ci[1]:.3f}] |"
        )
    (RESULTS / "ets_phantom_sensitivity.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    values = [row["median_ratio"] for row in rows]
    ax.bar(np.arange(len(rows)), values, color="#4c78a8")
    ax.axhline(measured_ratio, color="#d62728", linestyle="--", label="Compression-test GT")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(
        [row["configuration"] for row in rows],
        rotation=35,
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel(r"Median $E_{\mathrm{bg}}/E_{\mathrm{inc}}$")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_ets_sensitivity.png", dpi=300)
    plt.close(fig)
    return payload


def main() -> None:
    rows = [evaluate_sequence(acquisition) for acquisition in range(1, 7)]
    summary = _write_outputs(rows)
    sensitivity = _write_sensitivity(rows)
    print(json.dumps({"primary": summary, "sensitivity": sensitivity}, indent=2))


if __name__ == "__main__":
    main()
