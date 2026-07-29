"""Coarse-to-fine recovery when the heterogeneous region is not given."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lung_inverse_rendering.evaluate_sim_lung_v2 import (
    check_patient_consistency,
    fit_patient,
    load_patient,
    relative_error,
)
from physics.fem import make_heterogeneous_E_field, solve_nh_heterogeneous


DATASET = ROOT / "dataset" / "sim_lung_v2"
RESULTS = ROOT / "results" / "sim_lung_v2"


def candidate_cost(
    scene: dict,
    experiments: list[dict],
    center: torch.Tensor,
    radius: float,
) -> float:
    """Score a region with population material priors before joint refinement."""
    E_nodes = make_heterogeneous_E_field(
        scene["nodes"], center, radius, 5000.0, 12500.0
    )
    losses = []
    for experiment in experiments:
        observation = experiment["surface_motion_observed"][2].to(torch.float64)
        scale = max(float(torch.sqrt(observation.square().mean())), 1e-6)
        displacement = solve_nh_heterogeneous(
            scene["nodes"],
            scene["elems"],
            E_nodes,
            scene["nu_true"],
            experiment["forces"][2],
            scene["fixed"],
        )
        prediction = displacement.view(-1, 3)[scene["surface_node_ids"]]
        # Region scoring should depend on the spatial response pattern, not on
        # the globally unknown force/E scale. Eliminate one scalar amplitude
        # analytically before comparing candidate shapes.
        denominator = float(prediction.square().sum())
        amplitude = (
            float((prediction * observation).sum()) / denominator
            if denominator > 1e-16
            else 1.0
        )
        losses.append(
            float((((amplitude * prediction - observation) / scale) ** 2).mean())
        )
    return float(np.mean(losses))


def localize_region(
    scene: dict, experiments: list[dict]
) -> tuple[float, torch.Tensor, float, int]:
    minimum = scene["nodes"].min(dim=0).values
    extent = scene["nodes"].max(dim=0).values - minimum
    candidates = []
    for x_fraction in (0.36, 0.50, 0.64):
        for y_fraction in (0.38, 0.50, 0.62):
            for radius_fraction in (0.14, 0.18, 0.22):
                fraction = torch.tensor(
                    [x_fraction, y_fraction, 0.50], dtype=scene["nodes"].dtype
                )
                center = minimum + fraction * extent
                radius = radius_fraction * min(
                    float(extent[0]), float(extent[1])
                )
                cost = candidate_cost(scene, experiments, center, radius)
                candidates.append((cost, center, radius))
    coarse_cost, coarse_center, coarse_radius = min(
        candidates, key=lambda item: item[0]
    )
    in_plane_scale = min(float(extent[0]), float(extent[1]))
    refined = []
    for dx_fraction in (-0.07, 0.0, 0.07):
        for dy_fraction in (-0.07, 0.0, 0.07):
            for radius_delta in (-0.04, 0.0, 0.04):
                center = coarse_center.clone()
                center[0] += dx_fraction * extent[0]
                center[1] += dy_fraction * extent[1]
                center = torch.maximum(
                    minimum, torch.minimum(center, minimum + extent)
                )
                radius = max(
                    0.08 * in_plane_scale,
                    coarse_radius + radius_delta * in_plane_scale,
                )
                refined.append(
                    (candidate_cost(scene, experiments, center, radius), center, radius)
                )
    best = min([(coarse_cost, coarse_center, coarse_radius), *refined], key=lambda item: item[0])
    return best[0], best[1], best[2], len(candidates) + len(refined)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--max-nfev", type=int, default=40)
    parser.add_argument("--patient-start", type=int, default=0)
    parser.add_argument("--patient-limit", type=int)
    parser.add_argument("--output-tag")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    patients = [row for row in manifest["patients"] if row["split"] == "test"]
    patients = patients[args.patient_start :]
    if args.patient_limit is not None:
        patients = patients[: args.patient_limit]
    if not patients:
        raise ValueError("Selected test-patient range is empty")
    rows = []
    for patient in patients:
        scene, experiments = load_patient(args.dataset, patient)
        check_patient_consistency(scene, experiments)
        localization_cost, center, radius, candidate_count = localize_region(
            scene, experiments
        )
        estimate = fit_patient(
            scene,
            experiments,
            observation_key="surface_motion_observed",
            max_nfev=args.max_nfev,
            inclusion_center=center,
            inclusion_radius=radius,
        )
        true_center = experiments[0]["inclusion_center"].to(torch.float64)
        true_radius = float(experiments[0]["inclusion_radius"])
        geometry_scale = float(
            (scene["nodes"].max(dim=0).values - scene["nodes"].min(dim=0).values)[:2].max()
        )
        row = {
            "patient_id": patient["patient_id"],
            "center_error_normalized": float(
                torch.linalg.vector_norm(center - true_center) / geometry_scale
            ),
            "radius_relative_error": relative_error(radius, true_radius),
            "E_background_relative_error": relative_error(
                estimate["E_background"], patient["E_background"]
            ),
            "inclusion_ratio_relative_error": relative_error(
                estimate["inclusion_ratio"], patient["inclusion_ratio"]
            ),
            "localization_cost": localization_cost,
            "region_candidates": candidate_count,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "protocol": "unknown inclusion xy/radius; noisy 3-D surface motion",
        "method": (
            "scale-invariant coarse 27-region plus local 27-region scan, "
            "then E/ratio pattern search"
        ),
        "test_patient_count": len(rows),
        "center_error_normalized_median": float(
            np.median([row["center_error_normalized"] for row in rows])
        ),
        "radius_relative_error_median": float(
            np.median([row["radius_relative_error"] for row in rows])
        ),
        "E_background_relative_error_median": float(
            np.median([row["E_background_relative_error"] for row in rows])
        ),
        "inclusion_ratio_relative_error_median": float(
            np.median([row["inclusion_ratio_relative_error"] for row in rows])
        ),
        "records": rows,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_tag}" if args.output_tag else ""
    (RESULTS / f"metrics_unknown_region{suffix}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
