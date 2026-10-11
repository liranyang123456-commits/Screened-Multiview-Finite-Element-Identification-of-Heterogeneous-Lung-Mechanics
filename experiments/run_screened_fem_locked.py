"""Screened FEM with official/calibrator priors; ratio and region locked.

CT 120: official hybrid E/center/radius + 2028 bins3 ratio; refine only E and q.
Synthetic 50: calibrator prior; lock ratio when predicted contrast is weak;
rollback E/q when the local E-q Hessian correlation is too high.

Does not overwrite frozen MAP or official_lock.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lung_inverse_rendering.evaluate_sim_lung_v2 import (
    check_patient_consistency,
    fit_patient,
    force_scale_target,
    load_patient,
    relative_error,
)
from lung_inverse_rendering.pinn_inverse import official_bins3_regularized

SYNTHETIC_DATASET = ROOT / "dataset" / "sim_lung_ai_v2_multiview250"
SYNTHETIC_INIT = ROOT / "results" / "lung_mesh_ai_v2" / "node_response_calibrator_strict.json"
CT_DATASET = ROOT / "dataset" / "ion_ct_synthetic_mechanics540"
CT_HYBRID = ROOT / "results" / "ion_ct_expanded120_final" / "benchmark.json"
CT_GNN_2028 = ROOT / "results" / "ion_ct_ratio_optimization" / "calibrate_v2" / "gnn_test.json"

# Protocol keys added after the first frozen arms were written (2026-10-07/10).
# Older run_manifest.json files omit them; their absence means the neutral
# default below, so resuming/re-checking a frozen directory with the current
# script must not be rejected as a protocol mismatch.
LEGACY_PROTOCOL_DEFAULTS = {
    "model_nu": None,
    "posterior_fixation_quantile": None,
    "region_aspect_xy": None,
    "tracking_error_model": "iid",
    "tracking_dropout_rate": 0.0,
    "mesh_jitter_fraction": 0.0,
    "contact_shift_fraction": 0.0,
}


def _normalize_id(value: str) -> str:
    return str(value).replace("/", "_").replace("\\", "_")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _official_ratio_key(row: dict) -> str:
    if row.get("geometry_id") and row.get("scenario_id"):
        return f"{row['geometry_id']}|{row['scenario_id']}"
    patient = str(row.get("patient_id") or "").replace("\\", "/")
    if "/" in patient:
        geometry, scenario = patient.split("/", 1)
        return f"{geometry}|{scenario}"
    if "_scenario_" in patient:
        geometry, scenario = patient.split("_scenario_", 1)
        return f"{geometry}|scenario_{scenario}"
    return patient


def _index_records(path: Path, *, method: str | None = None) -> dict[str, dict]:
    payload = _load_json(path)
    records = payload["records"] if isinstance(payload, dict) and "records" in payload else payload
    index: dict[str, dict] = {}
    for row in records:
        if method is not None and row.get("method") != method:
            continue
        key = _normalize_id(str(row.get("patient_id") or row.get("entry_id")))
        index[key] = row
        if row.get("geometry_id") and row.get("scenario_id"):
            index[_normalize_id(f"{row['geometry_id']}_{row['scenario_id']}")] = row
    return index


def _official_2028_ratio_index() -> dict[str, dict]:
    index: dict[str, dict] = {}
    for row in _load_json(CT_GNN_2028)["records"]:
        payload = {
            "inclusion_ratio_estimated": official_bins3_regularized(row["inclusion_ratio_estimated"]),
            "log_ratio_std": float(row.get("log_ratio_std", 0.23)),
            "heterogeneity_probability": float(row.get("heterogeneity_probability", 0.5)),
            "flow_contrast": float(row.get("flow_contrast", 1.0)),
        }
        index[_official_ratio_key(row)] = payload
        index[_normalize_id(str(row["patient_id"]))] = payload
    return index


def _rebuild_posterior_fixed(nodes: torch.Tensor, quantile: float) -> torch.Tensor:
    """Fix the posterior fraction of nodes by z-quantile (default 0.80 = 20%)."""
    if not 0.5 <= float(quantile) < 1.0:
        raise ValueError("posterior fixation quantile must lie in [0.5, 1)")
    back_cut = torch.quantile(nodes[:, 2].to(torch.float64), float(quantile))
    back = torch.where(nodes[:, 2].to(torch.float64) >= back_cut)[0]
    return torch.cat([3 * back, 3 * back + 1, 3 * back + 2])


def _ellipsoid_occupancy(
    nodes: torch.Tensor,
    center: torch.Tensor,
    radius: float,
    aspect_xy: float,
) -> torch.Tensor:
    """Hard occupancy for an axis-aligned ellipsoid stretched in x relative to y/z."""
    if float(aspect_xy) <= 0:
        raise ValueError("region aspect_xy must be positive")
    delta = nodes.to(torch.float64) - center.to(torch.float64)
    scaled = delta.clone()
    scaled[:, 0] = scaled[:, 0] / float(aspect_xy)
    return (torch.linalg.vector_norm(scaled, dim=1) < float(radius)).to(torch.float64)


def _jitter_nodes(nodes: torch.Tensor, fraction: float, patient_id: str) -> torch.Tensor:
    """Deterministic mesh-node perturbation as a segmentation/mesh-error proxy."""
    if fraction < 0:
        raise ValueError("mesh jitter fraction must be non-negative")
    cloned = nodes.to(torch.float64).clone()
    extent = (cloned.max(dim=0).values - cloned.min(dim=0).values).clamp_min(1e-6)
    seed = abs(hash(patient_id)) % 1_000_000
    generator = torch.Generator().manual_seed(50_000 + seed)
    noise = torch.rand(cloned.shape, generator=generator, dtype=cloned.dtype) - 0.5
    return cloned + (2.0 * float(fraction)) * noise * extent


def _shift_contact_forces(experiments: list, scene: dict, shift_fraction: float) -> list:
    """Move nodal contact loads along +x by a fraction of the mesh width."""
    if shift_fraction == 0:
        return experiments
    nodes = scene["nodes"].to(torch.float64)
    surface = torch.unique(scene["surface_tris"].reshape(-1))
    dx = float(shift_fraction) * float(nodes[:, 0].max() - nodes[:, 0].min())
    shifted = []
    for experiment in experiments:
        copied = dict(experiment)
        for key in ("forces", "forces_measured"):
            if key not in copied:
                continue
            forces = copied[key].to(torch.float64).clone()
            frames = []
            for force in forces:
                nodal = force.view(-1, 3)
                loaded = torch.where(nodal.norm(dim=1) > 0)[0]
                moved = nodal.clone()
                if len(loaded):
                    targets = nodes[loaded, 0] + dx
                    for source, target_x in zip(loaded.tolist(), targets.tolist()):
                        candidates = surface
                        order = torch.argsort((nodes[candidates, 0] - target_x).abs())
                        destination = int(candidates[int(order[0])])
                        if destination != source:
                            moved[destination] = moved[destination] + moved[source]
                            moved[source] = 0
                frames.append(moved.reshape(-1))
            copied[key] = torch.stack(frames)
        shifted.append(copied)
    return shifted


def _predicted_region(scene: dict, prior: dict) -> tuple[torch.Tensor, float]:
    minimum = scene["nodes"].amin(dim=0)
    extent = scene["nodes"].amax(dim=0) - minimum
    center_fraction = torch.tensor(prior["center_fraction_estimated"], dtype=torch.float64)
    center = minimum + center_fraction * extent
    radius = float(prior["radius_fraction_estimated"]) * min(float(extent[0]), float(extent[1]))
    return center, radius


def _low_contrast(prior: dict, official: dict | None, threshold: float) -> bool:
    ratio = float(prior["inclusion_ratio_estimated"])
    if official is not None:
        ratio = float(official["inclusion_ratio_estimated"])
        hetero = float(official.get("heterogeneity_probability", 1.0))
        contrast = float(official.get("flow_contrast", 1.0))
        if hetero < 0.4 or contrast < 1e-6:
            return True
    return ratio <= threshold


def _median_and_bootstrap(values: list[float], replicates: int = 5000) -> dict:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return {"median": math.nan, "bootstrap_95_ci": [math.nan, math.nan]}
    rng = np.random.default_rng(2026)
    samples = [float(np.median(rng.choice(array, size=len(array), replace=True))) for _ in range(replicates)]
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"median": float(np.median(array)), "bootstrap_95_ci": [float(low), float(high)]}


def _summarize(records: list[dict]) -> dict:
    heterogeneous = [row for row in records if row.get("heterogeneous_true")]
    geometry_ratio: dict[str, list[float]] = {}
    for row in records:
        geometry = row.get("geometry_id")
        if geometry:
            geometry_ratio.setdefault(geometry, []).append(row["inclusion_ratio_relative_error"])
    geometry_medians = {key: float(np.median(values)) for key, values in geometry_ratio.items()}
    return {
        "n": len(records),
        "heterogeneous_n": len(heterogeneous),
        "E_background_median_relative_error": _median_and_bootstrap(
            [row["E_background_relative_error"] for row in records]
        ),
        "inclusion_ratio_median_relative_error": _median_and_bootstrap(
            [row["inclusion_ratio_relative_error"] for row in records]
        ),
        "center_error_normalized_median": _median_and_bootstrap(
            [
                row["center_error_normalized"]
                for row in heterogeneous
                if row["center_error_normalized"] is not None
            ]
        ),
        "radius_relative_error_median": _median_and_bootstrap(
            [
                row["radius_relative_error"]
                for row in heterogeneous
                if row["radius_relative_error"] is not None
            ]
        ),
        "force_scale_median_relative_error": _median_and_bootstrap(
            [row["force_scale_relative_error"] for row in records]
        ),
        "refinement_acceptance_rate": float(np.mean([bool(row["refinement_accepted"]) for row in records])),
        "screening_rejected_rate": float(np.mean([bool(row["screening_rejected"]) for row in records])),
        "eq_correlation_rejected_rate": float(
            np.mean([bool(row.get("eq_correlation_rejected")) for row in records])
        ),
        "ratio_locked_rate": float(np.mean([bool(row.get("lock_ratio")) for row in records])),
        "function_evaluations_median": float(np.median([row["function_evaluations"] for row in records])),
        "wall_time_seconds_median": float(np.median([row["wall_time_seconds"] for row in records])),
        "ratio_geometry_medians": geometry_medians,
        "ratio_geometry_pass_0.25": (
            f"{sum(value < 0.25 for value in geometry_medians.values())}/{len(geometry_medians)}"
            if geometry_medians
            else None
        ),
    }


def _endpoint_geometry(scene: dict, experiments: list[dict], prior: dict) -> dict:
    first = experiments[0]
    nodes = scene["nodes"].to(torch.float64)
    minimum = nodes.amin(dim=0)
    extent = nodes.amax(dim=0) - minimum
    lmin = float(min(float(extent[0]), float(extent[1])))
    true_center = ((first["inclusion_center"].to(torch.float64) - minimum) / extent).cpu().tolist()
    true_radius = float(first["inclusion_radius"]) / lmin if lmin > 0 else 0.0
    estimated_center = list(prior["center_fraction_estimated"])
    estimated_radius = float(prior["radius_fraction_estimated"])
    heterogeneous = float(first["inclusion_ratio"]) > 1.05
    center_error = None
    radius_error = None
    if heterogeneous:
        delta = torch.tensor(estimated_center, dtype=torch.float64) - torch.tensor(
            true_center, dtype=torch.float64
        )
        center_error = float(torch.linalg.vector_norm(delta))
        if true_radius > 0:
            radius_error = abs(estimated_radius / true_radius - 1.0)
    return {
        "center_fraction_true": true_center,
        "center_fraction_estimated": estimated_center,
        "radius_fraction_true": true_radius,
        "radius_fraction_estimated": estimated_radius,
        "center_error_normalized": center_error,
        "radius_relative_error": radius_error,
        "heterogeneous_true": heterogeneous,
    }


def run_split(args: argparse.Namespace) -> dict:
    """Auditable per-arm FEM checkpoint; setting identity is frozen at creation."""
    torch.set_default_dtype(torch.float64)
    # Default keeps the historical coupling (objective follows screening) so
    # existing run manifests still match; --objective-metric decouples them.
    objective_metric = args.objective_metric or args.screening_metric
    run_protocol = {
        "cohort": args.cohort,
        "prior_width_scale": args.prior_width_scale,
        "screen_threshold": args.screen_threshold,
        "accept_threshold": args.accept_threshold,
        "probe_count": args.probe_count,
        "model_force_bias": args.model_force_bias,
        "pose_rotation_error_deg": args.pose_rotation_error_deg,
        "model_nu": args.model_nu,
        "posterior_fixation_quantile": args.posterior_fixation_quantile,
        "region_aspect_xy": args.region_aspect_xy,
        "tracking_error_model": args.tracking_error_model,
        "tracking_dropout_rate": args.tracking_dropout_rate,
        "mesh_jitter_fraction": args.mesh_jitter_fraction,
        "contact_shift_fraction": args.contact_shift_fraction,
        "initialization_mode": args.initialization_mode,
        "prior_center_mode": args.prior_center_mode,
        "material_prior_weight": args.material_prior_weight,
        "screening_metric": args.screening_metric,
        "objective_metric": objective_metric,
        "max_nfev": args.max_nfev,
        "contrast_ratio_lock": args.contrast_ratio_lock,
        "eq_correlation_rollback": args.eq_correlation_rollback,
        "patient_start": args.patient_start,
        "patient_limit": args.patient_limit,
    }
    # Only recorded when explicitly requested so default-flag manifests stay
    # byte-identical to the frozen arms (absence == acceptance follows objective).
    if args.acceptance_metric is not None:
        run_protocol["acceptance_metric"] = args.acceptance_metric
    if args.cohort == "synthetic50":
        dataset = SYNTHETIC_DATASET
        predictions = _index_records(SYNTHETIC_INIT)
        official_ratios: dict[str, dict] = {}
        results = ROOT / "results" / "lung_mesh_ai_v2" / "screened_fem_locked50_v1"
        always_lock_ratio = False
    else:
        dataset = CT_DATASET
        predictions = _index_records(CT_HYBRID, method="ours_hybrid_initializer")
        official_ratios = _official_2028_ratio_index()
        results = ROOT / "results" / "ion_ct_official_prior_screened_fem120_v1"
        always_lock_ratio = True
    if args.results:
        results = args.results
    results.mkdir(parents=True, exist_ok=True)
    manifest_path = results / "run_manifest.json"
    manifest = _load_json(dataset / "manifest.json")
    patients = [row for row in manifest["patients"] if row["split"] == "test"]
    patients = patients[args.patient_start :]
    if args.patient_ids:
        requested = set(args.patient_ids)
        available = {str(row["patient_id"]) for row in patients}
        missing = requested - available
        if missing:
            raise ValueError(f"requested patient IDs are outside the selected test split: {sorted(missing)[:3]}")
        patients = [row for row in patients if str(row["patient_id"]) in requested]
    if args.patient_limit is not None:
        patients = patients[: args.patient_limit]
    run_protocol["patient_ids"] = [str(row["patient_id"]) for row in patients]
    run_protocol["frozen_input_hashes"] = {
        "dataset_manifest": hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest(),
        "hybrid_initializer": hashlib.sha256((CT_HYBRID if args.cohort == "ct120" else SYNTHETIC_INIT).read_bytes()).hexdigest(),
        "official_ratio_source": hashlib.sha256(CT_GNN_2028.read_bytes()).hexdigest() if args.cohort == "ct120" else None,
    }
    if manifest_path.exists():
        prior_protocol = {**LEGACY_PROTOCOL_DEFAULTS, **_load_json(manifest_path)}
        if prior_protocol != run_protocol:
            differing = sorted(
                key for key in set(prior_protocol) | set(run_protocol)
                if prior_protocol.get(key) != run_protocol.get(key)
            )
            raise RuntimeError(
                "checkpoint protocol/data mismatch; use a fresh --results directory "
                f"(differing keys: {differing})"
            )
    else:
        manifest_path.write_text(json.dumps(run_protocol, indent=2), encoding="utf-8")
    checkpoint = results / "records.jsonl"
    done: dict[str, dict] = {}
    if args.overwrite and checkpoint.exists():
        checkpoint.unlink()
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[str(row["patient_id"])] = row
    patient_ids = {str(row["patient_id"]) for row in patients}
    if checkpoint.exists():
        existing_ids = {str(json.loads(line)["patient_id"]) for line in checkpoint.read_text(encoding="utf-8").splitlines() if line.strip()}
        unexpected_ids = existing_ids - patient_ids
        if unexpected_ids:
            raise RuntimeError(f"checkpoint includes rows outside this arm's paired patient range: {sorted(unexpected_ids)[:3]}")
    records = []
    with checkpoint.open("a", encoding="utf-8") as writer:
        for row in patients:
            patient_id = row["patient_id"]
            if patient_id in done:
                records.append(done[patient_id])
                print(f"[skip] {patient_id}", flush=True)
                continue
            scene, experiments = load_patient(dataset, row)
            check_patient_consistency(scene, experiments)
            if args.model_nu is not None:
                scene = dict(scene)
                scene["nu_true"] = torch.tensor(float(args.model_nu), dtype=torch.float64)
            if args.posterior_fixation_quantile is not None:
                scene = dict(scene)
                scene["fixed"] = _rebuild_posterior_fixed(
                    scene["nodes"], args.posterior_fixation_quantile
                )
            if args.mesh_jitter_fraction:
                scene = dict(scene)
                scene["nodes"] = _jitter_nodes(scene["nodes"], args.mesh_jitter_fraction, patient_id)
            if args.contact_shift_fraction:
                experiments = _shift_contact_forces(
                    experiments, scene, args.contact_shift_fraction
                )
            prior = predictions.get(_normalize_id(patient_id))
            if prior is None:
                raise KeyError(f"No initializer record for {patient_id}")
            official = official_ratios.get(_official_ratio_key(row)) or official_ratios.get(
                _normalize_id(patient_id)
            )
            initial_E = float(prior["E_background_estimated"])
            initial_ratio = float(prior["inclusion_ratio_estimated"])
            if args.initialization_mode == "fixed":
                initial_E = args.fixed_initial_E
            if official is not None:
                initial_ratio = float(official["inclusion_ratio_estimated"])
            lock_ratio = always_lock_ratio or _low_contrast(
                prior, official, args.contrast_ratio_lock
            )
            center, radius = _predicted_region(scene, prior)
            soft_occupancy = None
            if args.region_aspect_xy is not None and abs(float(args.region_aspect_xy) - 1.0) > 1e-12:
                soft_occupancy = _ellipsoid_occupancy(
                    scene["nodes"], center, radius, args.region_aspect_xy
                )
            prior_center_E = (
                float(prior["E_background_estimated"])
                if args.prior_center_mode == "learned"
                else args.fixed_prior_E
            )
            material_prior_sigma = (
                max(0.03, args.prior_width_scale * float(prior.get("log_E_std", 0.06))),
                max(0.05, args.prior_width_scale * float((official or prior).get("log_ratio_std", 0.23))),
            )
            if args.material_prior_weight == 0 or objective_metric == "residual":
                material_prior_sigma = None
            search_half_width = (
                None
                if material_prior_sigma is None or args.material_prior_weight == 0 or objective_metric == "residual"
                else (
                    max(0.15, 2.0 * material_prior_sigma[0]),
                    0.0 if lock_ratio else max(0.15, 2.0 * material_prior_sigma[1]),
                )
            )
            estimate = fit_patient(
                scene,
                experiments,
                observation_key="image_tracks",
                max_nfev=args.max_nfev,
                force_scale_error=args.model_force_bias,
                pose_rotation_error_deg=args.pose_rotation_error_deg,
                # Always pass the *predicted* centre/radius: fit_patient falls back
                # to the ground-truth region when they are None, which would leak
                # the true region into the region-attention weights of the
                # ellipsoid (--region-aspect-xy) arms. The material occupancy
                # itself still comes from soft_occupancy when it is given.
                inclusion_center=center,
                inclusion_radius=radius,
                soft_occupancy=soft_occupancy,
                track_noise_px=0.05,
                tracking_error_model=args.tracking_error_model,
                tracking_dropout_rate=args.tracking_dropout_rate,
                use_all_track_frames=True,
                temporal_track_regression=True,
                track_smoothing_iterations=2,
                multiview_tracks=True,
                minimum_track_confidence=0.2,
                region_track_weight=8.0,
                initial_E_background=initial_E,
                initial_ratio=initial_ratio,
                search_half_width_log=search_half_width,
                optimize_force_scale=True,
                force_prior_sigma=0.05,
                material_prior_sigma=material_prior_sigma,
                material_prior_weight=args.material_prior_weight,
                material_prior_center=(prior_center_E, initial_ratio),
                screening_metric=args.screening_metric,
                objective_metric=objective_metric,
                acceptance_metric=args.acceptance_metric,
                screening_minimum_cost_reduction=args.screen_threshold,
                lock_ratio=lock_ratio,
                eq_correlation_rollback=args.eq_correlation_rollback,
                candidate_probe_count=args.probe_count,
            )
            geometry = _endpoint_geometry(scene, experiments, prior)
            record = {
                "patient_id": patient_id,
                "method": "official_prior_screened_fem" if args.cohort == "ct120" else "locked_screened_fem",
                "E_background_true": float(experiments[0]["E_background"]),
                "E_background_estimated": estimate["E_background"],
                "E_background_relative_error": relative_error(
                    estimate["E_background"], float(experiments[0]["E_background"])
                ),
                "inclusion_ratio_true": float(experiments[0]["inclusion_ratio"]),
                "inclusion_ratio_estimated": estimate["inclusion_ratio"],
                "inclusion_ratio_relative_error": relative_error(
                    estimate["inclusion_ratio"], float(experiments[0]["inclusion_ratio"])
                ),
                "force_scale_true": force_scale_target(experiments),
                "force_scale_estimated": estimate["force_scale"],
                "force_scale_relative_error": relative_error(
                    estimate["force_scale"], force_scale_target(experiments)
                ),
                **geometry,
                "used_predicted_region": True,
                "model_nu": float(args.model_nu) if args.model_nu is not None else None,
                "posterior_fixation_quantile": (
                    float(args.posterior_fixation_quantile)
                    if args.posterior_fixation_quantile is not None
                    else None
                ),
                "region_aspect_xy": (
                    float(args.region_aspect_xy) if args.region_aspect_xy is not None else None
                ),
                "initialization_mode": args.initialization_mode,
                "prior_center_mode": args.prior_center_mode,
                "material_prior_weight": args.material_prior_weight,
                "objective_metric": objective_metric,
                "screening_metric": args.screening_metric,
                "geometry_id": row.get("geometry_id"),
                "scenario_id": row.get("scenario_id"),
                **{
                    key: estimate[key]
                    for key in (
                        "refinement_accepted",
                        "screening_rejected",
                        "eq_correlation_rejected",
                        "lock_ratio",
                        "E_force_local_posterior_correlation",
                        "E_force_joint_log_covariance",
                        "E_force_joint_log_std",
                        "E_force_correlation_status",
                        "candidate_probe_count",
                        "candidate_directions",
                        "relative_cost_reduction",
                        "function_evaluations",
                        "wall_time_seconds",
                        "termination_reason",
                        "initial_E_background",
                        "initial_ratio",
                        "cost",
                        "initial_cost",
                        # present only when --acceptance-metric was given
                        "acceptance_metric",
                        "acceptance_initial_cost",
                        "acceptance_optimized_cost",
                        "acceptance_relative_cost_reduction",
                    )
                    if key in estimate
                },
            }
            writer.write(json.dumps(record) + "\n")
            writer.flush()
            records.append(record)
            print(
                f"[{len(records)}/{len(patients)}] {patient_id} "
                f"E={record['E_background_relative_error']:.3f} "
                f"ratio={record['inclusion_ratio_relative_error']:.3f} "
                f"lock={int(lock_ratio)} accept={int(record['refinement_accepted'])} "
                f"t={record['wall_time_seconds']:.1f}s",
                flush=True,
            )
    summary = {
        "protocol": {
            "cohort": args.cohort,
            "same_observations": "image_tracks + multiview temporal regression + Huber 1.5",
            "region": "initializer predicted sphere, not optimized",
            "ratio": "locked official 2028+bins3 on CT; contrast-locked on synthetic",
            "optimized": "log E_background and log q only when ratio is locked",
            "screening": f"Newton FEM Huber; first-round and total acceptance threshold {args.screen_threshold:.3f}",
            "prior_width_scale": args.prior_width_scale,
            "screen_threshold": args.screen_threshold,
            "accept_threshold": args.accept_threshold,
            "candidate_probe_count": args.probe_count,
            "candidate_directions": (
                [-1.0, 1.0]
                if args.probe_count == 2
                else [
                    *[
                        float(-value)
                        for value in np.linspace(1.0 / args.probe_count, 1.0, args.probe_count // 2)[::-1]
                    ],
                    *[
                        float(value)
                        for value in np.linspace(1.0 / args.probe_count, 1.0, args.probe_count // 2)
                    ],
                ]
            ),
            "intentional_force_model_bias": args.model_force_bias,
            "intentional_pose_rotation_error_deg": args.pose_rotation_error_deg,
            "initialization_mode": args.initialization_mode,
            "prior_center_mode": args.prior_center_mode,
            "material_prior_weight": args.material_prior_weight,
            "screening_metric": args.screening_metric,
            "objective_metric": objective_metric,
            "material_prior_center": [
                args.fixed_prior_E if args.prior_center_mode == "fixed" else records[0]["initial_E_background"],
                records[0]["initial_ratio"],
            ],
            "eq_correlation_rollback": args.eq_correlation_rollback,
            "steps_budget": args.max_nfev,
            **(
                {"acceptance_metric": args.acceptance_metric}
                if args.acceptance_metric is not None
                else {}
            ),
        },
        "screened_fem_locked": _summarize(records),
        "reference_screened_map_synthetic": {
            "E_background_median_relative_error": 0.041107,
            "inclusion_ratio_median_relative_error": 0.186924,
        },
        "reference_official_hybrid_ct": {
            "E_background_median_relative_error": 0.0855,
            "inclusion_ratio_median_relative_error": 0.1777,
            "center_error_normalized_median": 0.0960,
            "radius_relative_error_median": 0.1146,
        },
        "records": records,
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["screened_fem_locked"], indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("synthetic50", "ct120"), default="ct120")
    parser.add_argument("--patient-start", type=int, default=0)
    parser.add_argument("--patient-limit", type=int)
    parser.add_argument("--patient-ids", nargs="+", help="run an explicit paired subset of patient IDs")
    parser.add_argument("--max-nfev", type=int, default=48)
    parser.add_argument("--prior-width-scale", type=float, default=2.0)
    parser.add_argument("--screen-threshold", type=float, default=0.05)
    parser.add_argument("--probe-count", type=int, default=2)
    parser.add_argument("--accept-threshold", type=float, default=0.05)
    parser.add_argument("--model-force-bias", type=float, default=0.0, help="intentional force-model mismatch, e.g. 0.10")
    parser.add_argument("--pose-rotation-error-deg", type=float, default=0.0)
    parser.add_argument(
        "--model-nu",
        type=float,
        default=None,
        help="override FEM Poisson ratio (assumption sensitivity; data retain generator nu)",
    )
    parser.add_argument(
        "--posterior-fixation-quantile",
        type=float,
        default=None,
        help="rebuild fixed DOFs from z-quantile (0.80 ≈ posterior 20%% fixed)",
    )
    parser.add_argument(
        "--region-aspect-xy",
        type=float,
        default=None,
        help="ellipsoid x/y aspect for assumed inclusion occupancy (1.0 = sphere)",
    )
    parser.add_argument("--tracking-error-model", default="iid", choices=("iid", "dropout", "mixed"))
    parser.add_argument("--tracking-dropout-rate", type=float, default=0.0)
    parser.add_argument("--mesh-jitter-fraction", type=float, default=0.0)
    parser.add_argument("--contact-shift-fraction", type=float, default=0.0)
    parser.add_argument("--initialization-mode", choices=("learned", "fixed"), default="learned")
    parser.add_argument("--prior-center-mode", choices=("learned", "fixed"), default="learned")
    parser.add_argument("--fixed-initial-E", type=float, default=5000.0)
    parser.add_argument("--fixed-prior-E", type=float, default=5000.0)
    parser.add_argument("--material-prior-weight", type=float, default=0.05)
    parser.add_argument("--screening-metric", choices=("map", "residual"), default="map")
    parser.add_argument(
        "--objective-metric",
        choices=("map", "residual"),
        default=None,
        help="optimisation objective; defaults to the screening metric (historical coupled behaviour)",
    )
    parser.add_argument(
        "--acceptance-metric",
        choices=("map", "residual"),
        default=None,
        help=(
            "metric for the final acceptance/rollback test; defaults to the "
            "objective metric (historical behaviour, key omitted from manifests)"
        ),
    )
    parser.add_argument("--eq-correlation-rollback", type=float, default=0.85)
    parser.add_argument("--contrast-ratio-lock", type=float, default=1.20)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.prior_width_scale <= 0:
        parser.error("--prior-width-scale must be positive")
    if args.material_prior_weight < 0:
        parser.error("--material-prior-weight must be non-negative")
    if args.fixed_initial_E <= 0 or args.fixed_prior_E <= 0:
        parser.error("fixed E settings must be positive")
    if not 0.0 <= args.screen_threshold < 1.0:
        parser.error("--screen-threshold must be in [0, 1)")
    if not 0.0 <= args.accept_threshold < 1.0:
        parser.error("--accept-threshold must be in [0, 1)")
    if args.probe_count < 2 or args.probe_count % 2:
        parser.error("--probe-count must be an even integer >= 2")
    run_split(args)


if __name__ == "__main__":
    main()
