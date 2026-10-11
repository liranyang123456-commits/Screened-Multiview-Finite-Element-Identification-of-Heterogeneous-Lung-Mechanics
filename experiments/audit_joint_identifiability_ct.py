#!/usr/bin/env python3
"""Paired CT joint material/force/region sensitivity audit.

A smooth SDF occupancy relaxation is used only for regional local sensitivities;
it is not the hard-indicator estimator used by the benchmark fitter. Results
are local sensitivities at synthetic parameter truth, not a refit, uncertainty
interval, clinical inference, or global identifiability proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATASET = ROOT / "dataset" / "ion_ct_synthetic_mechanics540"
IDS_PATH = ROOT / "results" / "ct_reviewer_drivers_v6" / "paired_scene_ids.json"
OUT = ROOT / "results" / "ct_reviewer_drivers_v6" / "joint_identifiability_audit_v1.json"
EXPECTED_IDS_SHA256 = "c85c901b500a6e0dcb467804cc9884abce1d81803cd1bb45878a23cdfb983d7e"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median_nn_step(nodes: torch.Tensor) -> float:
    distances = torch.cdist(nodes, nodes)
    distances.fill_diagonal_(float("inf"))
    return float(distances.min(dim=1).values.median())


def load_case(row: dict) -> tuple[dict, list[dict]]:
    experiments = [
        torch.load(DATASET / item["relative_path"], map_location="cpu", weights_only=False)
        for item in row["experiments"]
    ]
    first = experiments[0]
    scene = {
        "nodes": first["nodes"].to(torch.float64),
        "elems": first["elems"],
        "surface_node_ids": first["surface_node_ids"],
        "fixed": first["fixed"],
        "Nn": len(first["nodes"]),
        "D": 3,
        "nu_true": torch.tensor(float(row["nu"]), dtype=torch.float64),
    }
    return scene, experiments


def predicted_flows(scene: dict, experiments: list[dict], theta: np.ndarray, steps: np.ndarray, smoothing_width: float, *, use_true_force: bool = True) -> tuple[np.ndarray, dict]:
    from physics.fem import solve_nh_heterogeneous
    from rendering.gaussian_pbr import project

    nodes = scene["nodes"]
    extent = nodes.amax(dim=0) - nodes.amin(dim=0)
    lmin = float(torch.minimum(extent[0], extent[1]))
    first = experiments[0]
    center0 = first["inclusion_center"].to(torch.float64)
    radius0 = float(first["inclusion_radius"])
    center = center0 + torch.as_tensor(theta[3:6] * steps[3:6], dtype=torch.float64) * extent
    radius = radius0 + float(theta[6] * steps[6] * lmin)
    if radius <= 0:
        raise ValueError("radius perturbation crossed zero")
    E_bg = float(first["E_background"]) * math.exp(float(theta[0] * steps[0]))
    ratio = float(first["inclusion_ratio"]) * math.exp(float(theta[1] * steps[1]))
    q = math.exp(float(theta[2] * steps[2]))
    distance = torch.linalg.vector_norm(nodes - center, dim=1)
    occupancy = torch.sigmoid((radius - distance) / smoothing_width)
    E_nodes = E_bg * (1.0 + (ratio - 1.0) * occupancy)
    surface_ids = scene["surface_node_ids"]
    frame = 2  # prescribed peak-envelope frame; frames 2 and 3 share the envelope.
    output: list[np.ndarray] = []
    visible_counts: dict[str, int] = {}
    for load_i, experiment in enumerate(experiments):
        force_field = experiment["forces"] if use_true_force else experiment["forces_measured"]
        force = force_field[frame].to(torch.float64) * q
        displacement = solve_nh_heterogeneous(
            nodes, scene["elems"], E_nodes, scene["nu_true"], force, scene["fixed"], D=3
        ).view(-1, 3)
        rest_points = nodes[surface_ids]
        deformed_points = (nodes + displacement)[surface_ids]
        confidence = experiment["image_occlusion_confidence_multiview_seq"][frame]
        rest_uv = experiment["image_uv_rest_multiview_seq"][frame]
        observed_uv = experiment["image_uv_deformed_multiview_seq"][frame]
        intrinsics = experiment["render_intrinsics"]
        for view in range(int(experiment["poses_multiview"].shape[1])):
            mask = confidence[view] >= 0.2
            visible_counts[f"load{load_i}_view{view}"] = int(mask.sum())
            if not mask.any():
                continue
            pose = experiment["poses_multiview"][frame, view].to(torch.float64)
            pred_uv, _, _ = project(deformed_points, pose, focal=float(intrinsics["focal_px"]), H=int(intrinsics["height_px"]), W=int(intrinsics["width_px"]))
            pred_rest_uv, _, _ = project(rest_points, pose, focal=float(intrinsics["focal_px"]), H=int(intrinsics["height_px"]), W=int(intrinsics["width_px"]))
            predicted_flow = pred_uv - pred_rest_uv
            observed_flow = observed_uv[view] - rest_uv[view]
            output.append((predicted_flow[mask] - observed_flow[mask]).detach().cpu().numpy().reshape(-1))
    return np.concatenate(output), {"visible_nodes_by_load_view": visible_counts, "soft_occupancy_sum": float(occupancy.sum()), "radius_fraction_lmin": radius / lmin}


def matrix_summary(matrix: np.ndarray) -> dict:
    colnorm = np.linalg.norm(matrix, axis=0)
    safe = np.where(colnorm > 1e-15, colnorm, 1.0)
    unit = matrix / safe
    s_raw = np.linalg.svd(matrix, compute_uv=False)
    s_unit = np.linalg.svd(unit, compute_uv=False)
    corr = unit.T @ unit
    return {
        "parameter_physical_sensitivity_norms": colnorm.tolist(),
        "raw_coordinate_singular_values": s_raw.tolist(),
        "raw_coordinate_condition_number": float(s_raw[0] / s_raw[-1]) if s_raw[-1] > 0 else None,
        "column_normalized_singular_values_collinearity_only": s_unit.tolist(),
        "column_normalized_smallest_largest_ratio_collinearity_only": float(s_unit[-1] / s_unit[0]) if s_unit[0] > 0 else None,
        "column_cosine_similarity": corr.tolist(),
        "JtJ_parameter_physical_coordinates": (matrix.T @ matrix).tolist(),
    }


def summary_from_gram(gram: np.ndarray) -> dict:
    gram = (gram + gram.T) * 0.5
    values = np.maximum(np.linalg.eigvalsh(gram), 0.0)
    singular = np.sqrt(values[::-1])
    return {
        "information_eigenvalues": values.tolist(),
        "singular_values": singular.tolist(),
        "effective_rank_relative_1e-3": int(np.sum(singular > singular[0] * 1e-3)) if singular[0] > 0 else 0,
        "smallest_largest_singular_ratio": float(singular[-1] / singular[0]) if singular[0] > 0 else None,
        "condition_number": float(values[-1] / values[0]) if values[0] > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-limit", type=int, help="debug-only prefix limit; omit for full 18-scene audit")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    ids = json.loads(IDS_PATH.read_text(encoding="utf-8"))
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    rows = {row["patient_id"]: row for row in manifest["patients"]}
    is_full_run = args.patient_limit is None
    ids_hash = sha256(IDS_PATH)
    if is_full_run and ids_hash != EXPECTED_IDS_SHA256:
        raise ValueError(f"locked paired scene ID file changed: {ids_hash}")
    if args.patient_limit is not None:
        ids = ids[:args.patient_limit]
    if (is_full_run and len(ids) != 18) or len(set(ids)) != len(ids) or any(pid not in rows or rows[pid]["split"] != "test" for pid in ids):
        raise ValueError("expected unique paired locked test scenes")
    geom_scenarios: dict[str, set[str]] = {}
    for pid in ids:
        row = rows[pid]
        geom_scenarios.setdefault(row["geometry_id"], set()).add(row["scenario_id"])
        exp_names = [e["name"] for e in row["experiments"]]
        if len(exp_names) != 4 or set(exp_names) != {"press_left", "press_right", "shear_x", "shear_y"}:
            raise ValueError(f"unexpected load modes for {pid}")
    if is_full_run:
        expected = {f"{geom}_scenario_{i:03d}" for geom in geom_scenarios for i in range(3)}
        if set(ids) != expected or len(geom_scenarios) != 6 or any(v != {"scenario_000", "scenario_001", "scenario_002"} for v in geom_scenarios.values()):
            raise ValueError("locked paired IDs must be exactly three scenarios for each of six geometries")

    parameter_names = ["log_E_background", "log_inclusion_ratio", "log_force_scale", "region_center_x", "region_center_y", "region_center_z", "region_radius"]
    per_scene: dict[str, dict] = {}
    jacobians: list[np.ndarray] = []
    geom_jacobians: dict[str, list[np.ndarray]] = {}
    runtime_start = time.time()
    for case_i, pid in enumerate(ids, 1):
        row = rows[pid]
        scene, experiments = load_case(row)
        nodes = scene["nodes"]
        extent = nodes.amax(dim=0) - nodes.amin(dim=0)
        mesh_h = median_nn_step(nodes)
        radius0 = float(experiments[0]["inclusion_radius"])
        radius_h = min(mesh_h, 0.10 * radius0)
        steps = np.array([0.01, 0.01, 0.01, mesh_h/(4*float(extent[0])), mesh_h/(4*float(extent[1])), mesh_h/(4*float(extent[2])), radius_h/(4*float(torch.minimum(extent[0],extent[1])))], dtype=float)
        smooth_width = mesh_h
        zero = np.zeros(7, dtype=float)
        base_residual, details = predicted_flows(scene, experiments, zero, steps, smooth_width)
        jac_columns = []
        for j in range(7):
            plus = zero.copy(); plus[j] = 1.0
            minus = zero.copy(); minus[j] = -1.0
            rplus, _ = predicted_flows(scene, experiments, plus, steps, smooth_width)
            rminus, _ = predicted_flows(scene, experiments, minus, steps, smooth_width)
            if rplus.shape != base_residual.shape or rminus.shape != base_residual.shape:
                raise RuntimeError(f"observation vector changed for {pid}/{parameter_names[j]}")
            jac_columns.append((rplus-rminus)/(2.0*steps[j]))
        J = np.column_stack(jac_columns)
        jacobians.append(J)
        geom = row["geometry_id"]
        geom_jacobians.setdefault(geom, []).append(J)
        per_scene[pid] = {
            "geometry_id": geom,
            "observation_scalar_count": int(J.shape[0]),
            "mesh_spacing": mesh_h,
            "parameter_steps": steps.tolist(),
            "region_smoothing_width": smooth_width,
            "baseline_residual_pixel_rmse": float(np.sqrt(np.mean(base_residual**2))),
            "baseline_note": "truth materials and simulator-truth forces at q=1, but nonzero residual is expected because data were generated with hard occupancy and the local diagnostic uses smooth SDF occupancy",
            "baseline_details": details,
            "sensitivity": matrix_summary(J),
        }
        print(f"[{case_i}/{len(ids)}] {pid}: observations={J.shape[0]} rmse={per_scene[pid]['baseline_residual_pixel_rmse']:.4g}px elapsed={time.time()-runtime_start:.1f}s", flush=True)

    scene_gram = sum((J.T @ J for J in jacobians), np.zeros((7,7)))
    geom_grams = {g: np.mean([J.T @ J for J in matrices], axis=0) for g, matrices in geom_jacobians.items()}
    geometry_equal_gram = sum(geom_grams.values(), np.zeros((7,7))) / len(geom_grams)
    q_prior_precision = np.zeros((7,7)); q_prior_precision[2,2] = 1.0 / (0.05**2)
    scene_prior_gram = scene_gram + q_prior_precision
    geometry_prior_gram = geometry_equal_gram + q_prior_precision

    # Recompute region derivatives at twice the primary physical perturbation on
    # one representative scenario from each geometry; report relative changes.
    step_sensitivity = {}
    reps = [next(pid for pid in ids if rows[pid]["geometry_id"] == g) for g in sorted(geom_jacobians)]
    for pid in reps:
        row = rows[pid]; scene, experiments = load_case(row); nodes = scene["nodes"]
        extent = nodes.amax(0) - nodes.amin(0); mesh_h = median_nn_step(nodes)
        rad = float(experiments[0]["inclusion_radius"]); radius_h = min(mesh_h, 0.10*rad)
        steps2 = np.array([.01,.01,.01,mesh_h/(2*float(extent[0])),mesh_h/(2*float(extent[1])),mesh_h/(2*float(extent[2])),radius_h/(2*float(torch.minimum(extent[0],extent[1])))])
        base,_ = predicted_flows(scene,experiments,np.zeros(7),steps2,mesh_h)
        cols=[]
        for j in range(3,7):
            p=np.zeros(7); m=np.zeros(7); p[j]=1; m[j]=-1
            rp,_=predicted_flows(scene,experiments,p,steps2,mesh_h); rm,_=predicted_flows(scene,experiments,m,steps2,mesh_h)
            cols.append((rp-rm)/(2.0*steps2[j]))
        primary=per_scene[pid]["sensitivity"]["parameter_physical_sensitivity_norms"][3:]
        doubled=np.linalg.norm(np.column_stack(cols),axis=0).tolist()
        step_sensitivity[pid]={"primary_step_region_column_norms":primary,"double_step_region_column_norms":doubled,"double_over_primary_norm_ratio":[(b/a if a>0 else None) for a,b in zip(primary,doubled)]}

    # Geometry-cluster bootstrap on geometry-specific information matrices.
    rng=np.random.default_rng(20261008); boot_ratios=[]; boot_ranks=[]
    geom_ids=sorted(geom_grams)
    for _ in range(1000):
        sampled=rng.choice(geom_ids,size=len(geom_ids),replace=True)
        gram=sum((geom_grams[g] for g in sampled),np.zeros((7,7)))
        summary=summary_from_gram(gram)
        boot_ratios.append(summary["smallest_largest_singular_ratio"] or 0.0)
        boot_ranks.append(summary["effective_rank_relative_1e-3"])

    payload={
        "audit":"Central finite-difference local sensitivity of image-plane flows around synthetic material truth using simulator-truth forces and a smooth SDF occupancy relaxation; not a joint refit/global proof.",
        "cohort":"exact paired 18 locked test scenes, six geometry clusters × three scenarios, four load modes and all available views; prescribed peak-envelope frame 2",
        "parameters":parameter_names,
        "parameterization":"physical-coordinate derivatives for log_E, log_ratio, log_q, region center xyz, and region radius; per-scene finite-difference scales are recorded",
        "region_relaxation":"sigmoid((radius-distance)/h), with h equal to median nearest-neighbor mesh spacing; local surrogate for production hard nodal occupancy",
        "finite_difference_steps":"log E/ratio/q +/-0.01; center +/- one-quarter median mesh spacing; radius +/- min(one-quarter mesh spacing, 2.5% radius)",
        "observation_scaling":"raw pixel flow residuals (no 0.05 px whitening; stored multiview UV tracks are noise-free projections)",
        "baseline_note":"Truth material parameters and simulator-truth forces at q=1, but not a zero-residual fit: smooth occupancy differs from hard-occupancy data generation; per-scene baseline residuals are retained.",
        "identifiability_scope":"Raw JtJ spectra characterize local Gauss-Newton data information in physical parameter coordinates. Column-normalized SVD is only a collinearity diagnostic. The illustrative force-only prior precision 1/0.05^2 is added directly to raw information in log_q coordinates; it is not the per-scene learned material prior used by the fitter.",
        "aggregation":"scene-weighted sum of JtJ across 18 rows; geometry-equal mean of three scenario JtJ matrices across six clusters; bootstrap unit is geometry, not scene",
        "scene_weighted_data_information":summary_from_gram(scene_gram),
        "geometry_equal_data_information":summary_from_gram(geometry_equal_gram),
        "force_prior_augmentation":{"log_force_sigma":0.05,"log_force_prior_precision":400.0,"interpretation":"illustrative force-only prior in raw log-force coordinates; not the per-scene learned material prior","scene_weighted_information_plus_prior":summary_from_gram(scene_prior_gram),"geometry_equal_information_plus_prior":summary_from_gram(geometry_prior_gram)},
        "geometry_cluster_bootstrap_1000":{"seed":20261008,"smallest_largest_singular_ratio_q05_median_q95":np.quantile(boot_ratios,[.05,.5,.95]).tolist(),"effective_rank_q05_median_q95":np.quantile(boot_ranks,[.05,.5,.95]).tolist(),"geometry_clusters":len(geom_ids)},
        "region_step_sensitivity_one_scene_per_geometry":step_sensitivity,
        "region_column_summary_across_scenes":{"median_physical_sensitivity_norms":np.median([x["sensitivity"]["parameter_physical_sensitivity_norms"][3:] for x in per_scene.values()],axis=0).tolist(),"q05_physical_sensitivity_norms":np.quantile([x["sensitivity"]["parameter_physical_sensitivity_norms"][3:] for x in per_scene.values()],0.05,axis=0).tolist(),"q95_physical_sensitivity_norms":np.quantile([x["sensitivity"]["parameter_physical_sensitivity_norms"][3:] for x in per_scene.values()],0.95,axis=0).tolist()},
        "per_scene":per_scene,
        "source_hashes":{"dataset_manifest_sha256":sha256(DATASET/"manifest.json"),"paired_scene_ids_sha256":sha256(IDS_PATH)},
        "runtime_seconds":time.time()-runtime_start,
    }
    OUT.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(OUT),"scene_information":payload["scene_weighted_data_information"],"geometry_equal_information":payload["geometry_equal_data_information"],"force_prior_scene":payload["force_prior_augmentation"]["scene_weighted_information_plus_prior"],"bootstrap_ratio_q05_med_q95":payload["geometry_cluster_bootstrap_1000"]["smallest_largest_singular_ratio_q05_median_q95"],"runtime_seconds":payload["runtime_seconds"]},indent=2))


if __name__=="__main__":
    main()
