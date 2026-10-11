"""Inverse audit when observations come from fiber or viscous finite strain.

The fitter stays isotropic Neo-Hookean. Only the observed multiview tracks are
reprojected from the mismatched forward displacement.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_screened_fem_locked import (  # noqa: E402
    CT_DATASET,
    CT_HYBRID,
    _index_records,
    _normalize_id,
    _official_2028_ratio_index,
    _official_ratio_key,
    _predicted_region,
)
from lung_inverse_rendering.evaluate_sim_lung_v2 import (  # noqa: E402
    fit_patient,
    force_scale_target,
    load_patient,
    relative_error,
)
from physics.constitutive_mismatch import solve_mismatch  # noqa: E402
from physics.fem import make_heterogeneous_E_field, solve_nh_heterogeneous  # noqa: E402
from rendering.gaussian_pbr import project  # noqa: E402

IDS = ROOT / "results" / "ct_reviewer_drivers_v6" / "paired_scene_ids.json"
OUT = ROOT / "results" / "ct_reviewer_drivers_v6" / "constitutive_mismatch"


def _truth_field(scene: dict, experiment: dict) -> torch.Tensor:
    return make_heterogeneous_E_field(
        scene["nodes"],
        experiment["inclusion_center"],
        float(experiment["inclusion_radius"]),
        float(experiment["E_background"]),
        float(experiment["E_inclusion"]),
    )


def _reproject(scene: dict, experiments: list[dict], law: str) -> dict:
    intrinsics = experiments[0]["render_intrinsics"]
    focal = float(intrinsics["focal_px"])
    height = int(intrinsics["height_px"])
    width = int(intrinsics["width_px"])
    surface = scene["surface_node_ids"]
    nodes = scene["nodes"]
    peak_gaps = []
    copied = []
    for experiment in experiments:
        fields = dict(experiment)
        observed = fields["image_uv_deformed_multiview_seq"].to(torch.float64).clone()
        E_nodes = _truth_field(scene, experiment)
        for frame in range(observed.shape[0]):
            force = experiment["forces"][frame].to(torch.float64)
            mismatch = solve_mismatch(
                nodes, scene["elems"], E_nodes, scene["nu_true"], force, scene["fixed"], law=law,
                n_load_steps=4, max_iter=8,
            )
            if frame == 2:
                reference = solve_nh_heterogeneous(
                    nodes, scene["elems"], E_nodes, scene["nu_true"], force, scene["fixed"]
                )
                peak_gaps.append(float((mismatch - reference).norm()) / (float(reference.norm()) + 1e-12))
            points = (nodes + mismatch.view(-1, 3))[surface]
            for view in range(observed.shape[1]):
                uv, _, _ = project(
                    points, experiment["poses_multiview"][frame, view], focal=focal, H=height, W=width,
                )
                observed[frame, view] = uv.to(observed.dtype)
        fields["image_uv_deformed_multiview_seq"] = observed
        copied.append(fields)
    ordered = sorted(peak_gaps)
    return {"experiments": copied, "peak_displacement_relative_gap": ordered[len(ordered) // 2]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--law", choices=("fiber", "viscous"), required=True)
    args = parser.parse_args()
    patient_ids = json.loads(IDS.read_text(encoding="utf-8"))
    manifest = json.loads((CT_DATASET / "manifest.json").read_text(encoding="utf-8"))
    rows = {row["patient_id"]: row for row in manifest["patients"]}
    predictions = _index_records(CT_HYBRID, method="ours_hybrid_initializer")
    official_ratios = _official_2028_ratio_index()
    destination = OUT / args.law
    destination.mkdir(parents=True, exist_ok=True)
    records_path = destination / "records.jsonl"
    done = {}
    if records_path.exists():
        done = {
            json.loads(line)["patient_id"]: json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    with records_path.open("a", encoding="utf-8") as handle:
        for patient_id in patient_ids:
            if patient_id in done:
                print("skip", patient_id, flush=True)
                continue
            scene, experiments = load_patient(CT_DATASET, rows[patient_id])
            reprojected = _reproject(scene, experiments, args.law)
            prior = predictions[_normalize_id(patient_id)]
            official = official_ratios.get(_official_ratio_key(rows[patient_id])) or prior
            center, radius = _predicted_region(scene, prior)
            estimate = fit_patient(
                scene,
                reprojected["experiments"],
                observation_key="image_tracks",
                max_nfev=48,
                inclusion_center=center,
                inclusion_radius=radius,
                track_noise_px=0.05,
                tracking_error_model="iid",
                use_all_track_frames=True,
                temporal_track_regression=True,
                track_smoothing_iterations=2,
                multiview_tracks=True,
                minimum_track_confidence=0.2,
                region_track_weight=8.0,
                initial_E_background=float(prior["E_background_estimated"]),
                initial_ratio=float(official["inclusion_ratio_estimated"]),
                search_half_width_log=(
                    max(0.15, 2.0 * max(0.03, 2.0 * float(prior.get("log_E_std", 0.06)))),
                    0.0,
                ),
                optimize_force_scale=True,
                force_prior_sigma=0.05,
                material_prior_sigma=(
                    max(0.03, 2.0 * float(prior.get("log_E_std", 0.06))),
                    max(0.05, 2.0 * float(official.get("log_ratio_std", 0.23))),
                ),
                material_prior_weight=0.05,
                material_prior_center=(float(prior["E_background_estimated"]), float(official["inclusion_ratio_estimated"])),
                screening_metric="map",
                objective_metric="map",
                screening_minimum_cost_reduction=0.05,
                lock_ratio=True,
                eq_correlation_rollback=0.85,
                candidate_probe_count=2,
            )
            record = {
                "patient_id": patient_id,
                "law": args.law,
                "peak_displacement_relative_gap": reprojected["peak_displacement_relative_gap"],
                "E_background_relative_error": relative_error(estimate["E_background"], float(experiments[0]["E_background"])),
                "inclusion_ratio_relative_error": relative_error(estimate["inclusion_ratio"], float(experiments[0]["inclusion_ratio"])),
                "force_scale_relative_error": relative_error(estimate["force_scale"], force_scale_target(experiments)),
                "refinement_accepted": bool(estimate.get("refinement_accepted")),
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(patient_id, record["E_background_relative_error"], record["peak_displacement_relative_gap"], flush=True)


if __name__ == "__main__":
    main()
