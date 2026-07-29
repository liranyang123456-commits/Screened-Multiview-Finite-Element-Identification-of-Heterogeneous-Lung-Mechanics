"""Jointly identify one material model from multiple experiments per patient."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physics.fem import solve_nh_heterogeneous
from rendering.gaussian_pbr import gaussian_centers, project


DATASET = ROOT / "dataset" / "sim_lung_v2"
RESULTS = ROOT / "results" / "sim_lung_v2"


def measured_forces(experiment: dict) -> torch.Tensor:
    """Return force observations, preferring the explicit measured field."""
    if "measured_forces" in experiment:
        return experiment["measured_forces"]
    if "forces_measured" in experiment:
        return experiment["forces_measured"]
    if "forces" in experiment:
        return experiment["forces"]
    raise KeyError("Experiment contains neither 'measured_forces' nor legacy 'forces'")


def observation_forces(
    experiment: dict, *, use_true_forces: bool = False
) -> torch.Tensor:
    """Select measured forces normally and simulator truth only for oracle runs."""
    if use_true_forces:
        if "forces" not in experiment:
            raise KeyError("Oracle force evaluation requires simulator 'forces'")
        return experiment["forces"]
    return measured_forces(experiment)


def material_occupancy(
    nodes: torch.Tensor,
    *,
    center: torch.Tensor | None = None,
    radius: float | None = None,
    node_partition: torch.Tensor | None = None,
    sdf: torch.Tensor | None = None,
    soft_occupancy: torch.Tensor | None = None,
    E_nodes: torch.Tensor | None = None,
    sdf_soft_width: float = 0.05,
) -> tuple[torch.Tensor, str]:
    """Convert supported region representations to a soft nodal occupancy."""
    supplied = sum(
        item is not None
        for item in (node_partition, sdf, soft_occupancy, E_nodes)
    )
    if supplied > 1:
        raise ValueError("Provide only one of node_partition, sdf, soft_occupancy, E_nodes")
    if soft_occupancy is not None:
        occupancy = torch.as_tensor(soft_occupancy, dtype=nodes.dtype)
        source = "soft_occupancy"
    elif node_partition is not None:
        partition = torch.as_tensor(node_partition)
        if partition.ndim != 1:
            raise ValueError("node_partition must be a one-dimensional node label array")
        occupancy = (partition != partition.min()).to(nodes.dtype)
        source = "node_partition"
    elif sdf is not None:
        signed_distance = torch.as_tensor(sdf, dtype=nodes.dtype)
        if sdf_soft_width <= 0:
            raise ValueError("sdf_soft_width must be positive")
        occupancy = torch.sigmoid(-signed_distance / sdf_soft_width)
        source = "sdf"
    elif E_nodes is not None:
        field = torch.as_tensor(E_nodes, dtype=nodes.dtype)
        log_field = field.clamp_min(torch.finfo(nodes.dtype).tiny).log()
        span = log_field.max() - log_field.min()
        occupancy = (
            (log_field - log_field.min()) / span
            if float(span) > 1e-12
            else torch.zeros_like(log_field)
        )
        source = "E_nodes_shape"
    else:
        if center is None or radius is None:
            raise ValueError("A spherical center/radius or a nodal region field is required")
        occupancy = (
            torch.linalg.vector_norm(nodes - center.to(nodes.dtype), dim=1) < radius
        ).to(nodes.dtype)
        source = "sphere"
    if occupancy.shape != (len(nodes),):
        raise ValueError(
            f"Region field has shape {tuple(occupancy.shape)}; expected {(len(nodes),)}"
        )
    if not torch.isfinite(occupancy).all():
        raise ValueError("Region field contains non-finite values")
    return occupancy.clamp(0.0, 1.0), source


def parameterized_E_nodes(
    occupancy: torch.Tensor, E_background: float, ratio: float
) -> torch.Tensor:
    return E_background * (1.0 + (ratio - 1.0) * occupancy)


def smooth_surface_tracks(
    values: torch.Tensor,
    surface_nodes: torch.Tensor,
    surface_tris: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    if iterations <= 0:
        return values
    local_index = {
        int(node): index for index, node in enumerate(surface_nodes.tolist())
    }
    neighbors: list[set[int]] = [set() for _ in range(len(surface_nodes))]
    for triangle in surface_tris.tolist():
        local = [local_index[int(node)] for node in triangle if int(node) in local_index]
        for first in local:
            neighbors[first].update(second for second in local if second != first)
    result = values
    for _ in range(iterations):
        rows = []
        for index, adjacent in enumerate(neighbors):
            if adjacent:
                neighbor_mean = result[list(adjacent)].mean(dim=0)
                rows.append(0.5 * result[index] + 0.5 * neighbor_mean)
            else:
                rows.append(result[index])
        result = torch.stack(rows)
    return result


def pose_with_yaw_error(pose: torch.Tensor, degrees: float) -> torch.Tensor:
    if degrees == 0.0:
        return pose
    angle = math.radians(degrees)
    rotation = torch.tensor(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ],
        dtype=pose.dtype,
    )
    perturbed = pose.clone()
    perturbed[:3, :3] = pose[:3, :3] @ rotation
    return perturbed


def load_patient(dataset: Path, row: dict) -> tuple[dict, list[dict]]:
    experiments = [
        torch.load(dataset / item["relative_path"], map_location="cpu", weights_only=False)
        for item in row["experiments"]
    ]
    first = experiments[0]
    scene = {
        "nodes": first["nodes"].to(torch.float64),
        "elems": first["elems"],
        "surface_tris": first["surface_tris"],
        "surface_node_ids": first["surface_node_ids"],
        "fixed": first["fixed"],
        "Nn": len(first["nodes"]),
        "D": 3,
        "nu_true": torch.tensor(row["nu"], dtype=torch.float64),
    }
    return scene, experiments


def check_patient_consistency(scene: dict, experiments: list[dict]) -> None:
    """Reject a dataset if one patient changes geometry/material across loads."""
    reference = experiments[0]
    for experiment in experiments[1:]:
        if not torch.equal(reference["nodes"], experiment["nodes"]):
            raise ValueError("Geometry changed between experiments for one patient")
        if not torch.equal(reference["fixed"], experiment["fixed"]):
            raise ValueError("Boundary conditions changed between experiments")
        for key in ("E_background", "E_inclusion", "inclusion_ratio"):
            if float(reference[key]) != float(experiment[key]):
                raise ValueError(f"Patient material changed across experiments: {key}")
    if not torch.equal(scene["surface_node_ids"], reference["surface_node_ids"]):
        raise ValueError("Surface observation nodes are inconsistent")


def fit_patient(
    scene: dict,
    experiments: list[dict],
    *,
    observation_key: str,
    max_nfev: int,
    force_scale_error: float = 0.0,
    experiments_limit: int | None = None,
    pose_rotation_error_deg: float = 0.0,
    inclusion_center: torch.Tensor | None = None,
    inclusion_radius: float | None = None,
    track_noise_px: float = 0.25,
    use_all_track_frames: bool = False,
    force_calibration_factor: float = 1.0,
    track_entity: str = "nodes",
    temporal_track_regression: bool = False,
    track_smoothing_iterations: int = 0,
    multiview_tracks: bool = False,
    minimum_track_confidence: float = 0.2,
    region_track_weight: float = 0.0,
    initial_E_background: float = 5000.0,
    initial_ratio: float = 1.8,
    search_half_width_log: tuple[float, float] | None = None,
    optimize_force_scale: bool = False,
    use_true_forces: bool = False,
    force_prior_sigma: float | None = None,
    initial_force_scale: float = 1.0,
    initial_force_scale_std: float | None = None,
    material_prior_sigma: tuple[float, float] | None = None,
    material_prior_weight: float = 1.0,
    minimum_refinement_cost_reduction: float = 0.05,
    screening_minimum_cost_reduction: float = 0.0,
    node_partition: torch.Tensor | None = None,
    sdf: torch.Tensor | None = None,
    soft_occupancy: torch.Tensor | None = None,
    E_nodes: torch.Tensor | None = None,
    sdf_soft_width: float = 0.05,
    convergence_threshold: float = 0.01,
) -> dict:
    """Fit material parameters and, with an informative prior, force scale."""
    started_at = time.perf_counter()
    if optimize_force_scale and (
        force_prior_sigma is None
        or not math.isfinite(force_prior_sigma)
        or force_prior_sigma <= 0
    ):
        raise ValueError(
            "Force scale is not identifiable jointly with stiffness without an "
            "informative Gaussian prior; provide force_prior_sigma > 0"
        )
    if initial_force_scale <= 0:
        raise ValueError("initial_force_scale must be positive")
    if convergence_threshold <= 0:
        raise ValueError("convergence_threshold must be positive")
    if material_prior_weight < 0:
        raise ValueError("material_prior_weight must be non-negative")
    if experiments_limit is not None:
        experiments = experiments[:experiments_limit]
    first = experiments[0]
    has_nodal_region = any(
        item is not None for item in (node_partition, sdf, soft_occupancy, E_nodes)
    )
    center = (
        inclusion_center.to(torch.float64)
        if inclusion_center is not None
        else first["inclusion_center"].to(torch.float64)
        if "inclusion_center" in first
        else None
    )
    radius = (
        float(inclusion_radius)
        if inclusion_radius is not None
        else float(first["inclusion_radius"])
        if "inclusion_radius" in first
        else None
    )
    if not has_nodal_region and (center is None or radius is None):
        raise ValueError("Spherical material region requires inclusion_center/radius")
    occupancy, region_source = material_occupancy(
        scene["nodes"],
        center=center,
        radius=radius,
        node_partition=node_partition,
        sdf=sdf,
        soft_occupancy=soft_occupancy,
        E_nodes=E_nodes,
        sdf_soft_width=sdf_soft_width,
    )
    surface_nodes = scene["surface_node_ids"]
    peak_frame = 2
    image_track_mode = observation_key == "image_tracks"
    if image_track_mode:
        observations, track_masks, track_intrinsics, track_weights = [], [], [], []
        fit_experiments, fit_frames, fit_views = [], [], []
        if region_track_weight > 0:
            if center is None:
                occupancy_sum = occupancy.sum().clamp_min(1e-6)
                attention_center = (
                    scene["nodes"] * occupancy[:, None]
                ).sum(dim=0) / occupancy_sum
                attention_radius = torch.sqrt(
                    (
                        (scene["nodes"] - attention_center)
                        .square()
                        .sum(dim=1)
                        * occupancy
                    ).sum()
                    / occupancy_sum
                ).clamp_min(1e-6)
            else:
                attention_center = center
                attention_radius = max(float(radius or 0.0), 1e-6)
            surface_distance = torch.linalg.vector_norm(
                scene["nodes"][surface_nodes] - attention_center, dim=1
            )
            surface_attention = 1.0 + region_track_weight * torch.exp(
                -0.5 * (surface_distance / (2.0 * attention_radius)) ** 2
            )
        else:
            surface_attention = torch.ones(
                len(surface_nodes), dtype=scene["nodes"].dtype
            )
        frame_indices = (1, 2, 3, 4, 5) if use_all_track_frames else (peak_frame,)
        generator = torch.Generator().manual_seed(
            93_000 + int(first["patient_id"].rsplit("_", 1)[1])
        )
        for experiment in experiments:
            if multiview_tracks:
                rest = experiment["image_uv_rest_multiview_seq"]
                deformed = experiment["image_uv_deformed_multiview_seq"]
                confidence = experiment["image_occlusion_confidence_multiview_seq"]
                intrinsics = experiment["render_intrinsics"]
                for view_index in range(rest.shape[1]):
                    if temporal_track_regression:
                        regression_frames = (1, 2, 3, 4, 5)
                        envelopes = (0.45, 1.0, 1.0, 0.65, 0.25)
                        mask = torch.stack(
                            [
                                confidence[frame_index, view_index]
                                >= minimum_track_confidence
                                for frame_index in regression_frames
                            ]
                        ).all(dim=0)
                        numerator = None
                        denominator = sum(value * value for value in envelopes)
                        for frame_index, envelope in zip(
                            regression_frames, envelopes
                        ):
                            noise = track_noise_px * torch.randn(
                                deformed[frame_index, view_index].shape,
                                generator=generator,
                                dtype=deformed.dtype,
                            )
                            weighted = envelope * (
                                deformed[frame_index, view_index]
                                + noise
                                - rest[frame_index, view_index]
                            )
                            numerator = (
                                weighted if numerator is None else numerator + weighted
                            )
                        regressed = numerator / denominator
                        regressed = smooth_surface_tracks(
                            regressed,
                            surface_nodes,
                            scene["surface_tris"],
                            track_smoothing_iterations,
                        )
                        observations.append(regressed[mask])
                        track_masks.append(mask)
                        track_intrinsics.append(intrinsics)
                        track_weights.append(surface_attention[mask])
                        fit_experiments.append(experiment)
                        fit_frames.append(peak_frame)
                        fit_views.append(view_index)
                    else:
                        for frame_index in frame_indices:
                            mask = (
                                confidence[frame_index, view_index]
                                >= minimum_track_confidence
                            )
                            noise = track_noise_px * torch.randn(
                                deformed[frame_index, view_index].shape,
                                generator=generator,
                                dtype=deformed.dtype,
                            )
                            flow = (
                                deformed[frame_index, view_index]
                                + noise
                                - rest[frame_index, view_index]
                            )
                            flow = smooth_surface_tracks(
                                flow,
                                surface_nodes,
                                scene["surface_tris"],
                                track_smoothing_iterations,
                            )
                            observations.append(flow[mask])
                            track_masks.append(mask)
                            track_intrinsics.append(intrinsics)
                            track_weights.append(surface_attention[mask])
                            fit_experiments.append(experiment)
                            fit_frames.append(frame_index)
                            fit_views.append(view_index)
                continue
            prefix = "image_gaussian_" if track_entity == "gaussians" else "image_"
            rest_key = f"{prefix}uv_rest_seq"
            deformed_key = f"{prefix}uv_deformed_seq"
            visibility_key = f"{prefix}visibility_seq"
            if rest_key not in experiment:
                raise ValueError(
                    "Dataset has no persisted image tracks; run "
                    "augment_sim_lung_v2_tracks.py"
                )
            intrinsics = experiment["render_intrinsics"]
            if temporal_track_regression:
                regression_frames = (1, 2, 3, 4, 5)
                envelopes = (0.45, 1.0, 1.0, 0.65, 0.25)
                masks = [
                    experiment[visibility_key][frame_index]
                    for frame_index in regression_frames
                ]
                mask = torch.stack(masks).all(dim=0)
                numerator = None
                denominator = sum(value * value for value in envelopes)
                for frame_index, envelope in zip(regression_frames, envelopes):
                    reference_uv = experiment[rest_key][frame_index]
                    observed_uv = experiment[deformed_key][frame_index]
                    noise = track_noise_px * torch.randn(
                        observed_uv.shape,
                        generator=generator,
                        dtype=observed_uv.dtype,
                    )
                    weighted = envelope * (observed_uv + noise - reference_uv)
                    numerator = weighted if numerator is None else numerator + weighted
                regressed = numerator / denominator
                if track_entity == "nodes":
                    regressed = smooth_surface_tracks(
                        regressed,
                        surface_nodes,
                        scene["surface_tris"],
                        track_smoothing_iterations,
                    )
                observations.append(regressed[mask])
                track_masks.append(mask)
                track_intrinsics.append(intrinsics)
                track_weights.append(
                    surface_attention[mask]
                    if track_entity == "nodes"
                    else torch.ones(int(mask.sum()), dtype=scene["nodes"].dtype)
                )
                fit_experiments.append(experiment)
                fit_frames.append(peak_frame)
                fit_views.append(None)
            else:
                for frame_index in frame_indices:
                    reference_uv = experiment[rest_key][frame_index]
                    observed_uv = experiment[deformed_key][frame_index]
                    mask = experiment[visibility_key][frame_index]
                    noise = track_noise_px * torch.randn(
                        observed_uv.shape,
                        generator=generator,
                        dtype=observed_uv.dtype,
                    )
                    flow = observed_uv + noise - reference_uv
                    if track_entity == "nodes":
                        flow = smooth_surface_tracks(
                            flow,
                            surface_nodes,
                            scene["surface_tris"],
                            track_smoothing_iterations,
                        )
                    observations.append(flow[mask])
                    track_masks.append(mask)
                    track_intrinsics.append(intrinsics)
                    track_weights.append(
                        surface_attention[mask]
                        if track_entity == "nodes"
                        else torch.ones(int(mask.sum()), dtype=scene["nodes"].dtype)
                    )
                    fit_experiments.append(experiment)
                    fit_frames.append(frame_index)
                    fit_views.append(None)
        scales = [
            max(float(torch.sqrt(observation.square().mean())), max(track_noise_px, 1e-3))
            for observation in observations
        ]
    else:
        fit_experiments = experiments
        fit_frames = [peak_frame] * len(experiments)
        observations = [
            experiment[observation_key][peak_frame].to(torch.float64)
            for experiment in experiments
        ]
        scales = [
            max(float(torch.sqrt(observation.square().mean())), 1e-6)
            for observation in observations
        ]
    evaluations = 0
    diagnostic_evaluations = 0

    def objective(parameters: np.ndarray, *, diagnostic: bool = False) -> float:
        nonlocal evaluations, diagnostic_evaluations
        if diagnostic:
            diagnostic_evaluations += 1
        else:
            evaluations += 1
        E_background = math.exp(float(parameters[0]))
        ratio = math.exp(float(parameters[1]))
        force_scale = math.exp(float(parameters[2])) if optimize_force_scale else 1.0
        current_E_nodes = parameterized_E_nodes(occupancy, E_background, ratio)
        experiment_losses = []
        displacement_cache: dict[tuple[int, int], torch.Tensor] = {}
        for index, (experiment, observation, scale) in enumerate(
            zip(fit_experiments, observations, scales)
        ):
            frame_index = fit_frames[index]
            cache_key = (id(experiment), frame_index)
            if cache_key not in displacement_cache:
                displacement_cache[cache_key] = solve_nh_heterogeneous(
                    scene["nodes"],
                    scene["elems"],
                    current_E_nodes,
                    scene["nu_true"],
                    observation_forces(
                        experiment, use_true_forces=use_true_forces
                    )[frame_index]
                    * (1.0 + force_scale_error)
                    * force_calibration_factor
                    * force_scale,
                    scene["fixed"],
                )
            displacement = displacement_cache[cache_key]
            if image_track_mode:
                intrinsics = track_intrinsics[index]
                predicted_pose = pose_with_yaw_error(
                    experiment["poses_multiview"][frame_index, fit_views[index]]
                    if fit_views[index] is not None
                    else experiment["poses"][frame_index],
                    pose_rotation_error_deg,
                )
                if track_entity == "gaussians":
                    track_gaussians = {
                        "host_tri": experiment["image_gaussian_host_tri"],
                        "bary": experiment["image_gaussian_bary"],
                    }
                    zero = torch.zeros_like(displacement)
                    reference_points = gaussian_centers(
                        track_gaussians, scene, zero
                    )
                    deformed_points = gaussian_centers(
                        track_gaussians, scene, displacement
                    )
                else:
                    reference_points = scene["nodes"][surface_nodes]
                    deformed_points = (
                        scene["nodes"] + displacement.view(-1, 3)
                    )[surface_nodes]
                predicted_reference_uv, _, _ = project(
                    reference_points,
                    predicted_pose,
                    focal=float(intrinsics["focal_px"]),
                    H=int(intrinsics["height_px"]),
                    W=int(intrinsics["width_px"]),
                )
                predicted_uv, _, _ = project(
                    deformed_points,
                    predicted_pose,
                    focal=float(intrinsics["focal_px"]),
                    H=int(intrinsics["height_px"]),
                    W=int(intrinsics["width_px"]),
                )
                predicted_flow = predicted_uv - predicted_reference_uv
                if track_entity == "nodes":
                    predicted_flow = smooth_surface_tracks(
                        predicted_flow,
                        surface_nodes,
                        scene["surface_tris"],
                        track_smoothing_iterations,
                    )
                predicted = predicted_flow[track_masks[index]]
            else:
                predicted = displacement.view(-1, 3)[surface_nodes]
            normalized = (predicted - observation) / scale
            absolute = normalized.abs()
            huber = torch.where(
                absolute <= 1.5,
                0.5 * normalized.square(),
                1.5 * (absolute - 0.75),
            )
            if image_track_mode:
                point_loss = huber.mean(dim=-1)
                weight = track_weights[index].to(point_loss)
                experiment_losses.append(
                    float((point_loss * weight).sum() / weight.sum().clamp_min(1e-8))
                )
            else:
                experiment_losses.append(float(huber.mean()))
        # Each load/view is an independent observation. Summing their robust
        # likelihoods keeps Gaussian parameter priors on the correct scale;
        # averaging here previously made the force prior dominate the data.
        data_cost = float(np.sum(experiment_losses))
        if material_prior_sigma is not None and material_prior_weight > 0:
            prior_sigma = np.asarray(material_prior_sigma, dtype=float)
            if prior_sigma.shape != (2,) or np.any(prior_sigma <= 0):
                raise ValueError("material_prior_sigma must contain two positive values")
            prior_center = np.log([initial_E_background, initial_ratio])
            data_cost += 0.5 * material_prior_weight * float(
                np.square((parameters[:2] - prior_center) / prior_sigma).sum()
            )
        if optimize_force_scale:
            assert force_prior_sigma is not None
            data_cost += 0.5 * (float(parameters[2]) / force_prior_sigma) ** 2
        return data_cost

    global_lower = np.log(
        [1000.0, 1.0, 0.25] if optimize_force_scale else [1000.0, 1.0]
    )
    global_upper = np.log(
        [15000.0, 5.0, 4.0] if optimize_force_scale else [15000.0, 5.0]
    )
    initial_values = [initial_E_background, initial_ratio]
    if optimize_force_scale:
        initial_values.append(initial_force_scale)
    parameters = np.clip(
        np.log(initial_values),
        global_lower,
        global_upper,
    )
    initial_parameters = parameters.copy()
    if search_half_width_log is None:
        lower, upper = global_lower, global_upper
    else:
        half_width = np.asarray(search_half_width_log, dtype=float)
        if optimize_force_scale:
            force_width = max(
                0.05,
                2.0 * (
                    initial_force_scale_std / initial_force_scale
                    if initial_force_scale_std is not None
                    else force_prior_sigma
                ),
            )
            half_width = np.concatenate((half_width, [force_width]))
        lower = np.maximum(global_lower, parameters - half_width)
        upper = np.minimum(global_upper, parameters + half_width)
    material_steps = (
        [
            min(0.15, max(0.025, 0.5 * material_prior_sigma[0])),
            min(0.15, max(0.025, 0.5 * material_prior_sigma[1])),
        ]
        if material_prior_sigma is not None
        else [0.15, 0.15]
    )
    steps = np.asarray(
        [*material_steps, max(0.025, force_prior_sigma or 0.05)]
        if optimize_force_scale
        else material_steps
    )
    best_cost = objective(parameters)
    initial_cost = best_cost
    screening_rejected = False
    search_iteration = 0
    # Deterministic coarse-to-fine coordinate pattern search. It avoids mixed
    # OpenMP runtimes from scipy+torch and is adequate for this 2-D problem.
    while evaluations < max_nfev:
        search_iteration += 1
        candidates = []
        for dimension in range(len(parameters)):
            for direction in (-1.0, 1.0):
                candidate = parameters.copy()
                candidate[dimension] += direction * steps[dimension]
                candidates.append(np.clip(candidate, lower, upper))
        improved = False
        best_candidate = None
        best_candidate_cost = best_cost
        for candidate in candidates:
            if evaluations >= max_nfev:
                break
            cost = objective(candidate)
            if cost < best_candidate_cost:
                best_candidate, best_candidate_cost = candidate, cost
        if best_candidate is not None:
            parameters, best_cost = best_candidate, best_candidate_cost
            improved = True
        if search_iteration == 1 and screening_minimum_cost_reduction > 0:
            screening_reduction = (initial_cost - best_cost) / max(
                abs(initial_cost), 1e-12
            )
            if screening_reduction < screening_minimum_cost_reduction:
                screening_rejected = True
                break
        if not improved:
            steps *= 0.5
        if float(steps.max()) < convergence_threshold:
            break
    optimized_cost = best_cost
    relative_cost_reduction = (initial_cost - optimized_cost) / max(
        abs(initial_cost), 1e-12
    )
    refinement_accepted = bool(
        relative_cost_reduction >= minimum_refinement_cost_reduction
    )
    if not refinement_accepted:
        parameters = initial_parameters
        best_cost = initial_cost
    fitted = np.exp(parameters)
    E_background, ratio = fitted[:2]
    force_scale = float(fitted[2]) if optimize_force_scale else 1.0
    hessian_correlation = None
    hessian_status = "not_requested"
    if optimize_force_scale and not screening_rejected:
        # A local finite-difference Hessian exposes the expected E/force
        # confounding. The Gaussian prior makes this posterior curvature finite.
        h = max(0.005, min(0.02, (force_prior_sigma or 0.05) / 2.0))
        base = objective(parameters, diagnostic=True)
        e_plus, e_minus = parameters.copy(), parameters.copy()
        f_plus, f_minus = parameters.copy(), parameters.copy()
        e_plus[0] += h
        e_minus[0] -= h
        f_plus[2] += h
        f_minus[2] -= h
        h_ee = (
            objective(e_plus, diagnostic=True)
            - 2.0 * base
            + objective(e_minus, diagnostic=True)
        ) / h**2
        h_ff = (
            objective(f_plus, diagnostic=True)
            - 2.0 * base
            + objective(f_minus, diagnostic=True)
        ) / h**2
        pp, pm, mp, mm = (parameters.copy() for _ in range(4))
        pp[[0, 2]] += h
        pm[0] += h
        pm[2] -= h
        mp[0] -= h
        mp[2] += h
        mm[[0, 2]] -= h
        h_ef = (
            objective(pp, diagnostic=True)
            - objective(pm, diagnostic=True)
            - objective(mp, diagnostic=True)
            + objective(mm, diagnostic=True)
        ) / (4.0 * h**2)
        local_hessian = np.asarray([[h_ee, h_ef], [h_ef, h_ff]])
        if np.isfinite(local_hessian).all() and np.linalg.det(local_hessian) > 1e-12:
            covariance = np.linalg.inv(local_hessian)
            denominator = math.sqrt(max(covariance[0, 0] * covariance[1, 1], 0.0))
            if denominator > 0:
                hessian_correlation = float(covariance[0, 1] / denominator)
                hessian_status = "local_laplace"
            else:
                hessian_status = "non_positive_covariance"
        else:
            hessian_status = "singular_or_indefinite"
    elif screening_rejected:
        hessian_status = "screening_rejected"
    converged = bool(
        screening_rejected or float(steps.max()) < convergence_threshold
    )
    termination_reason = (
        "insufficient_initial_physics_evidence"
        if screening_rejected
        else "step_tolerance"
        if converged
        else "evaluation_budget"
    )
    return {
        "E_background": float(E_background),
        "inclusion_ratio": float(ratio),
        "force_scale": force_scale,
        "force_scale_log": math.log(force_scale),
        "force_prior_sigma": force_prior_sigma,
        "force_scale_identifiability": (
            "informative_prior" if optimize_force_scale else "externally_calibrated"
        ),
        "E_force_local_posterior_correlation": hessian_correlation,
        "E_force_correlation_status": hessian_status,
        "region_source": region_source,
        "optimizer_success": converged,
        "converged": converged,
        "optimizer_status": 1 if converged else 0,
        "optimizer_message": "pattern search completed",
        "termination_reason": termination_reason,
        "function_evaluations": evaluations,
        "diagnostic_function_evaluations": diagnostic_evaluations,
        "wall_time_seconds": time.perf_counter() - started_at,
        "cost": best_cost,
        "initial_cost": initial_cost,
        "optimized_cost": optimized_cost,
        "relative_cost_reduction": relative_cost_reduction,
        "refinement_accepted": refinement_accepted,
        "screening_rejected": screening_rejected,
        "final_max_log_step": float(steps.max()),
        "initial_E_background": float(initial_E_background),
        "initial_ratio": float(initial_ratio),
        "initial_force_scale": float(initial_force_scale),
    }


def relative_error(estimate: float, target: float) -> float:
    return abs(estimate - target) / target


def predicted_array(
    prediction: dict[str, Any], name: str, prediction_file: Path
) -> torch.Tensor | None:
    """Load an inline or sidecar nodal prediction."""
    if name in prediction:
        return torch.as_tensor(prediction[name], dtype=torch.float64)
    path_key = f"{name}_path"
    if path_key not in prediction:
        return None
    path = Path(prediction[path_key])
    if not path.is_absolute():
        path = prediction_file.parent / path
    if path.suffix == ".npy":
        return torch.from_numpy(np.load(path)).to(torch.float64)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        payload = payload[name]
    return torch.as_tensor(payload, dtype=torch.float64)


def force_scale_target(experiments: list[dict]) -> float:
    """Best available simulation target for force-scale error reporting."""
    first = experiments[0]
    for key in ("force_scale_true", "true_force_scale"):
        if key in first:
            return float(first[key])
    if "force_scale_true_N" in first and "force_scale_measured_N" in first:
        measured_scale = float(first["force_scale_measured_N"])
        return (
            float(first["force_scale_true_N"]) / measured_scale
            if measured_scale > 0
            else 1.0
        )
    if "forces_measured" in first and "forces" in first:
        measured = first["forces_measured"].to(torch.float64).reshape(-1)
        truth = first["forces"].to(torch.float64).reshape(-1)
        denominator = float(measured.square().sum())
        if denominator > 0:
            return float((measured * truth).sum() / denominator)
    if "true_forces" in first:
        measured = measured_forces(first).to(torch.float64).reshape(-1)
        truth = first["true_forces"].to(torch.float64).reshape(-1)
        denominator = float(measured.square().sum())
        if denominator > 0:
            return float((measured * truth).sum() / denominator)
    return 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument(
        "--observation",
        choices=("oracle", "noisy", "image_tracks"),
        default="noisy",
    )
    parser.add_argument("--max-nfev", type=int, default=40)
    parser.add_argument("--force-scale-error", type=float, default=0.0)
    parser.add_argument("--force-calibration-factor", type=float, default=1.0)
    parser.add_argument("--estimate-force-scale", action="store_true")
    parser.add_argument(
        "--use-true-forces",
        action="store_true",
        help="Oracle-only: use simulator forces instead of measured forces",
    )
    parser.add_argument(
        "--force-prior-sigma",
        type=float,
        choices=(0.02, 0.05, 0.10),
        help="Gaussian prior std for log force scale; also enables MAP force fitting",
    )
    parser.add_argument("--experiments-limit", type=int)
    parser.add_argument("--pose-rotation-error-deg", type=float, default=0.0)
    parser.add_argument("--track-noise-px", type=float, default=0.25)
    parser.add_argument("--all-track-frames", action="store_true")
    parser.add_argument(
        "--track-entity", choices=("nodes", "gaussians"), default="nodes"
    )
    parser.add_argument("--temporal-track-regression", action="store_true")
    parser.add_argument("--track-smoothing-iterations", type=int, default=0)
    parser.add_argument("--multiview-tracks", action="store_true")
    parser.add_argument("--minimum-track-confidence", type=float, default=0.2)
    parser.add_argument("--region-track-weight", type=float, default=0.0)
    parser.add_argument("--patient-start", type=int, default=0)
    parser.add_argument("--patient-limit", type=int)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-tag")
    parser.add_argument("--benchmark-method")
    parser.add_argument(
        "--initial-predictions",
        type=Path,
        help="JSON with records keyed by patient_id from the lung MeshGNN",
    )
    parser.add_argument("--adaptive-search-sigma", type=float, default=2.0)
    parser.add_argument("--use-predicted-region", action="store_true")
    parser.add_argument("--sdf-soft-width", type=float, default=0.05)
    parser.add_argument("--convergence-threshold", type=float, default=0.01)
    parser.add_argument("--material-prior-weight", type=float, default=1.0)
    parser.add_argument(
        "--minimum-refinement-cost-reduction", type=float, default=0.05
    )
    parser.add_argument(
        "--screening-minimum-cost-reduction", type=float, default=0.0
    )
    args = parser.parse_args()
    if args.use_true_forces and args.estimate_force_scale:
        parser.error("--use-true-forces cannot be combined with --estimate-force-scale")
    optimize_force_scale = (not args.use_true_forces) and (
        args.estimate_force_scale or args.force_prior_sigma is not None
    )
    if optimize_force_scale and args.force_prior_sigma is None:
        parser.error("--estimate-force-scale requires --force-prior-sigma")
    torch.set_default_dtype(torch.float64)
    manifest_path = (
        args.dataset if args.dataset.is_file() else args.dataset / "manifest.json"
    )
    dataset_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_patients = [row for row in manifest["patients"] if row["split"] == args.split]
    test_patients = test_patients[args.patient_start :]
    if args.patient_limit is not None:
        test_patients = test_patients[: args.patient_limit]
    if not test_patients:
        raise ValueError("Selected test-patient range is empty")
    observation_key = {
        "oracle": "surface_motion_true",
        "noisy": "surface_motion_observed",
        "image_tracks": "image_tracks",
    }[args.observation]
    initial_predictions = {}
    if args.initial_predictions:
        prediction_payload = json.loads(
            args.initial_predictions.read_text(encoding="utf-8")
        )
        prediction_records = (
            prediction_payload.get("validation_records", [])
            if args.split == "val"
            else prediction_payload["records"]
        )
        initial_predictions = {
            row["patient_id"]: row for row in prediction_records
        }
    records = []
    for patient in test_patients:
        scene, experiments = load_patient(dataset_root, patient)
        check_patient_consistency(scene, experiments)
        prior = initial_predictions.get(patient["patient_id"])
        if prior is None:
            initial_E_background, initial_ratio, search_width = 5000.0, 1.8, None
            predicted_center, predicted_radius = None, None
            initial_force_scale, initial_force_scale_std = 1.0, None
            nodal_region = {}
            material_prior_sigma = None
        else:
            initial_E_background = float(prior["E_background_estimated"])
            initial_ratio = float(prior["inclusion_ratio_estimated"])
            search_width = (
                max(0.15, args.adaptive_search_sigma * float(prior["log_E_std"])),
                max(0.15, args.adaptive_search_sigma * float(prior["log_ratio_std"])),
            )
            material_prior_sigma = (
                max(0.03, float(prior["log_E_std"])),
                max(0.05, float(prior["log_ratio_std"])),
            )
            if args.use_predicted_region:
                minimum = scene["nodes"].amin(dim=0)
                extent = scene["nodes"].amax(dim=0) - minimum
                center_fraction = torch.tensor(
                    prior["center_fraction_estimated"], dtype=torch.float64
                )
                predicted_center = minimum + center_fraction * extent
                predicted_radius = float(prior["radius_fraction_estimated"]) * min(
                    float(extent[0]), float(extent[1])
                )
            else:
                predicted_center, predicted_radius = None, None
            if "force_scale_mean" in prior:
                initial_force_scale = float(prior["force_scale_mean"])
            elif "log_force_scale_mean" in prior:
                initial_force_scale = math.exp(float(prior["log_force_scale_mean"]))
            else:
                initial_force_scale = 1.0
            initial_force_scale_std = (
                float(prior["force_scale_std"])
                if "force_scale_std" in prior
                else initial_force_scale * float(prior["log_force_scale_std"])
                if "log_force_scale_std" in prior
                else None
            )
            nodal_region = {}
            if args.use_predicted_region:
                for field_name in (
                    "node_partition",
                    "sdf",
                    "soft_occupancy",
                    "E_nodes",
                ):
                    field = predicted_array(prior, field_name, args.initial_predictions)
                    if field is not None:
                        nodal_region = {field_name: field}
                        predicted_center, predicted_radius = None, None
                        break
        estimate = fit_patient(
            scene,
            experiments,
            observation_key=observation_key,
            max_nfev=args.max_nfev,
            force_scale_error=args.force_scale_error,
            force_calibration_factor=args.force_calibration_factor,
            experiments_limit=args.experiments_limit,
            pose_rotation_error_deg=args.pose_rotation_error_deg,
            track_noise_px=args.track_noise_px,
            use_all_track_frames=args.all_track_frames,
            track_entity=args.track_entity,
            temporal_track_regression=args.temporal_track_regression,
            track_smoothing_iterations=args.track_smoothing_iterations,
            multiview_tracks=args.multiview_tracks,
            minimum_track_confidence=args.minimum_track_confidence,
            region_track_weight=args.region_track_weight,
            initial_E_background=initial_E_background,
            initial_ratio=initial_ratio,
            search_half_width_log=search_width,
            inclusion_center=predicted_center,
            inclusion_radius=predicted_radius,
            optimize_force_scale=optimize_force_scale,
            use_true_forces=args.use_true_forces,
            force_prior_sigma=args.force_prior_sigma,
            initial_force_scale=initial_force_scale,
            initial_force_scale_std=initial_force_scale_std,
            material_prior_sigma=material_prior_sigma,
            material_prior_weight=args.material_prior_weight,
            minimum_refinement_cost_reduction=args.minimum_refinement_cost_reduction,
            screening_minimum_cost_reduction=args.screening_minimum_cost_reduction,
            sdf_soft_width=args.sdf_soft_width,
            convergence_threshold=args.convergence_threshold,
            **nodal_region,
        )
        true_force_scale = force_scale_target(experiments)
        effective_force_scale = (
            estimate["force_scale"]
            * args.force_calibration_factor
            * (1.0 + args.force_scale_error)
        )
        record = {
            "patient_id": patient["patient_id"],
            "force_observation_source": (
                "simulator_truth_oracle" if args.use_true_forces else "measured"
            ),
            "experiment_count": (
                min(len(experiments), args.experiments_limit)
                if args.experiments_limit is not None
                else len(experiments)
            ),
            "E_background_true": patient["E_background"],
            "E_background_estimated": estimate["E_background"],
            "E_background_relative_error": relative_error(
                estimate["E_background"], patient["E_background"]
            ),
            "inclusion_ratio_true": patient["inclusion_ratio"],
            "inclusion_ratio_estimated": estimate["inclusion_ratio"],
            "inclusion_ratio_relative_error": relative_error(
                estimate["inclusion_ratio"], patient["inclusion_ratio"]
            ),
            "force_scale_true": true_force_scale,
            "force_scale_estimated": effective_force_scale,
            "force_scale_relative_error": relative_error(
                effective_force_scale, true_force_scale
            ),
            "used_learned_initialization": prior is not None,
            "used_predicted_region": prior is not None
            and args.use_predicted_region,
            **{
                key: value
                for key, value in estimate.items()
                if key not in ("E_background", "inclusion_ratio")
            },
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    result = {
        "dataset": manifest.get("version", args.dataset.name),
        "split": args.split,
        "protocol": args.observation,
        "benchmark_method": args.benchmark_method,
        "force_scale_error": args.force_scale_error,
        "force_calibration_factor": args.force_calibration_factor,
        "force_scale_estimated_by_map": optimize_force_scale,
        "force_observation_source": (
            "simulator_truth_oracle" if args.use_true_forces else "measured"
        ),
        "force_prior_sigma": args.force_prior_sigma,
        "pose_rotation_error_deg": args.pose_rotation_error_deg,
        "track_noise_px": args.track_noise_px if args.observation == "image_tracks" else None,
        "track_frames": "loading_to_unloading" if args.all_track_frames else "peak_only",
        "track_entity": args.track_entity if args.observation == "image_tracks" else None,
        "temporal_track_regression": args.temporal_track_regression,
        "track_smoothing_iterations": args.track_smoothing_iterations,
        "multiview_tracks": args.multiview_tracks,
        "minimum_track_confidence": (
            args.minimum_track_confidence if args.multiview_tracks else None
        ),
        "region_track_weight": args.region_track_weight,
        "minimum_refinement_cost_reduction": args.minimum_refinement_cost_reduction,
        "screening_minimum_cost_reduction": args.screening_minimum_cost_reduction,
        "learned_initialization": bool(args.initial_predictions),
        "predicted_region": args.use_predicted_region,
        "known": [
            "patient geometry",
            "contact force and position",
            "boundary-condition DOFs",
            *([] if args.use_predicted_region else ["inclusion region"]),
        ],
        "estimated": [
            "patient E_background",
            "patient inclusion_ratio",
            *(["force scale with Gaussian MAP prior"] if optimize_force_scale else []),
            *(["inclusion region from MeshGNN"] if args.use_predicted_region else []),
        ],
        "test_patient_count": len(records),
        "experiments_per_patient": (
            min(len(test_patients[0]["experiments"]), args.experiments_limit)
            if args.experiments_limit is not None
            else len(test_patients[0]["experiments"])
        ),
        "E_background_median_relative_error": float(
            np.median([row["E_background_relative_error"] for row in records])
        ),
        "inclusion_ratio_median_relative_error": float(
            np.median([row["inclusion_ratio_relative_error"] for row in records])
        ),
        "force_scale_median_relative_error": float(
            np.median([row["force_scale_relative_error"] for row in records])
        ),
        "optimizer_success_rate": float(
            np.mean([bool(row["optimizer_success"]) for row in records])
        ),
        "refinement_acceptance_rate": float(
            np.mean([bool(row["refinement_accepted"]) for row in records])
        ),
        "function_evaluations_median": float(
            np.median([row["function_evaluations"] for row in records])
        ),
        "wall_time_seconds_median": float(
            np.median([row["wall_time_seconds"] for row in records])
        ),
        "records": records,
    }
    args.results.mkdir(parents=True, exist_ok=True)
    suffix = args.observation
    if args.force_scale_error:
        suffix += f"_force_{args.force_scale_error:+.2f}".replace("+", "p").replace("-", "m")
    if args.force_calibration_factor != 1.0:
        suffix += f"_cal_{args.force_calibration_factor:.4f}"
    if optimize_force_scale:
        suffix += f"_force_map_sigma_{args.force_prior_sigma:.2f}"
    if args.experiments_limit is not None:
        suffix += f"_loads_{args.experiments_limit}"
    if args.pose_rotation_error_deg:
        suffix += f"_pose_{args.pose_rotation_error_deg:+.1f}".replace("+", "p").replace("-", "m")
    if args.observation == "image_tracks":
        suffix += f"_px_{args.track_noise_px:.2f}"
        suffix += f"_{args.track_entity}"
        if args.all_track_frames:
            suffix += "_allframes"
        if args.temporal_track_regression:
            suffix += "_temporalreg"
        if args.track_smoothing_iterations:
            suffix += f"_smooth{args.track_smoothing_iterations}"
        if args.multiview_tracks:
            suffix += f"_multiview_conf{args.minimum_track_confidence:.2f}"
        if args.region_track_weight:
            suffix += f"_regionweight{args.region_track_weight:.1f}"
    if args.output_tag:
        suffix += f"_{args.output_tag}"
    if args.initial_predictions:
        suffix += "_gnn_init"
    if args.use_predicted_region:
        suffix += "_predicted_region"
    (args.results / f"metrics_{suffix}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
