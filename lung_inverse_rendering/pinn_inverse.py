"""PINN inverse-mechanics arm on the existing sim_lung_v2 generator.

The displacement field is a load-conditioned MLP. Material uses the same
compact two-level field as screened MAP. The data term is the frozen
multiview Huber track residual; the physics term is the weak Neo-Hookean
residual assembled on the tetrahedral mesh. Region (c, r) is taken from the
same initializer as the FEM arms so the four endpoints stay comparable.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from lung_inverse_rendering.evaluate_sim_lung_v2 import (
    force_scale_target,
    material_occupancy,
    observation_forces,
    parameterized_E_nodes,
    relative_error,
    smooth_surface_tracks,
)
from lung_inverse_rendering.tracking_error import TrackingError
from physics.fem import assemble_internal_force, node_E_to_per_elem_mu_lam
from rendering.gaussian_pbr import project


class DisplacementPINN(nn.Module):
    """Maps normalized rest coordinates and a load index to nodal displacement."""

    def __init__(self, n_loads: int = 4, width: int = 64, depth: int = 4, embed_dim: int = 8):
        super().__init__()
        self.load_embed = nn.Embedding(n_loads, embed_dim)
        layers: list[nn.Module] = [nn.Linear(3 + embed_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 3))
        self.net = nn.Sequential(*layers)
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)

    def forward(self, x_norm: torch.Tensor, load_ids: torch.Tensor) -> torch.Tensor:
        features = torch.cat([x_norm, self.load_embed(load_ids)], dim=-1)
        return self.net(features)


def official_bins3_regularized(ratio: float) -> float:
    """Locked CT piecewise ratio calibrator (validation geometry-pass rule).

    Matches ``experiments.calibrate_ct_ratio_v2.apply_bins`` for
    ``bins3_regularized``: right-closed bins and clip to ``[1, 5]``.
    """
    edges = (0.0, 2.3522442307010984, 3.1483814709163886, 10.0)
    multipliers = (0.625, 1.175, 1.05)
    value = float(ratio)
    for index, high in enumerate(edges[1:]):
        if value <= high or index == len(multipliers) - 1:
            return float(min(5.0, max(1.0, value * multipliers[index])))
    return float(min(5.0, max(1.0, value)))


def _as_device(value: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return value.to(device=device, dtype=dtype)


def _fixed_dof_mask(fixed: torch.Tensor, n_nodes: int, dim: int, device: torch.device) -> torch.Tensor:
    mask = torch.ones(n_nodes * dim, dtype=torch.bool, device=device)
    if fixed.numel() == 0:
        return mask
    if fixed.dtype == torch.bool:
        if fixed.numel() == n_nodes:
            mask = (~fixed.to(device)).repeat_interleave(dim)
            return mask
        mask = ~fixed.to(device=device)
        return mask
    mask[fixed.to(device=device, dtype=torch.long)] = False
    return mask


def build_track_observations(
    scene: dict,
    experiments: list[dict],
    *,
    occupancy: torch.Tensor,
    center: torch.Tensor | None,
    radius: float | None,
    track_noise_px: float,
    minimum_track_confidence: float,
    region_track_weight: float,
    track_smoothing_iterations: int,
    tracking_error_model: str = "iid",
) -> list[dict[str, Any]]:
    """Frozen-protocol amplitude-regressed multiview tracks (CPU tensors)."""
    surface_nodes = scene["surface_node_ids"]
    nodes = scene["nodes"]
    if region_track_weight > 0:
        if center is None:
            occupancy_sum = occupancy.sum().clamp_min(1e-6)
            attention_center = (nodes * occupancy[:, None]).sum(dim=0) / occupancy_sum
            attention_radius = torch.sqrt(
                ((nodes - attention_center).square().sum(dim=1) * occupancy).sum()
                / occupancy_sum
            ).clamp_min(1e-6)
        else:
            attention_center = center
            attention_radius = max(float(radius or 0.0), 1e-6)
        surface_distance = torch.linalg.vector_norm(
            nodes[surface_nodes] - attention_center, dim=1
        )
        surface_attention = 1.0 + region_track_weight * torch.exp(
            -0.5 * (surface_distance / (2.0 * attention_radius)) ** 2
        )
    else:
        surface_attention = torch.ones(len(surface_nodes), dtype=nodes.dtype)

    first = experiments[0]
    patient_token = str(first["patient_id"]).rsplit("_", 1)[-1]
    try:
        patient_seed = int(patient_token)
    except ValueError:
        patient_seed = abs(hash(str(first["patient_id"]))) % 1_000_000
    generator = torch.Generator().manual_seed(93_000 + patient_seed)
    tracker = TrackingError(tracking_error_model, track_noise_px, generator)
    peak_frame = 2
    regression_frames = (1, 2, 3, 4, 5)
    envelopes = (0.45, 1.0, 1.0, 0.65, 0.25)
    packs: list[dict[str, Any]] = []
    for load_index, experiment in enumerate(experiments):
        rest = experiment["image_uv_rest_multiview_seq"]
        deformed = experiment["image_uv_deformed_multiview_seq"]
        confidence = experiment["image_occlusion_confidence_multiview_seq"]
        intrinsics = experiment["render_intrinsics"]
        for view_index in range(rest.shape[1]):
            mask = torch.stack(
                [
                    confidence[frame_index, view_index] >= minimum_track_confidence
                    for frame_index in regression_frames
                ]
            ).all(dim=0)
            numerator = None
            keep_all = None
            denominator = sum(value * value for value in envelopes)
            for frame_index, envelope in zip(regression_frames, envelopes):
                noise, keep = tracker.noise(
                    tuple(deformed[frame_index, view_index].shape),
                    deformed.dtype,
                    view_index,
                    confidence[frame_index, view_index],
                )
                if keep is not None:
                    keep_all = keep if keep_all is None else keep_all & keep
                weighted = envelope * (
                    deformed[frame_index, view_index] + noise - rest[frame_index, view_index]
                )
                numerator = weighted if numerator is None else numerator + weighted
            if keep_all is not None:
                mask = mask & keep_all
            regressed = numerator / denominator
            regressed = smooth_surface_tracks(
                regressed, surface_nodes, scene["surface_tris"], track_smoothing_iterations
            )
            scale = max(float(torch.sqrt(regressed[mask].square().mean())), max(track_noise_px, 1e-3))
            packs.append(
                {
                    "load_index": load_index,
                    "view_index": view_index,
                    "frame_index": peak_frame,
                    "mask": mask,
                    "observation": regressed[mask].to(torch.float64),
                    "weight": surface_attention[mask].to(torch.float64),
                    "scale": scale,
                    "pose": experiment["poses_multiview"][peak_frame, view_index].to(torch.float64),
                    "intrinsics": intrinsics,
                    "force": observation_forces(experiment)[peak_frame].to(torch.float64),
                }
            )
    return packs


def _huber(normalized: torch.Tensor) -> torch.Tensor:
    absolute = normalized.abs()
    return torch.where(absolute <= 1.5, 0.5 * normalized.square(), 1.5 * (absolute - 0.75))


@dataclass
class PINNFitConfig:
    steps: int = 800
    learning_rate: float = 2.0e-3
    physics_weight: float = 1.0
    width: int = 64
    depth: int = 4
    force_prior_sigma: float = 0.05
    material_prior_weight: float = 0.05
    track_noise_px: float = 0.05
    minimum_track_confidence: float = 0.2
    region_track_weight: float = 8.0
    track_smoothing_iterations: int = 2


def fit_patient_pinn(
    scene: dict,
    experiments: list[dict],
    *,
    initial_E_background: float,
    initial_ratio: float,
    inclusion_center: torch.Tensor | None,
    inclusion_radius: float | None,
    material_prior_sigma: tuple[float, float] | None,
    device: torch.device,
    config: PINNFitConfig,
    center_fraction_estimated: list[float] | None = None,
    radius_fraction_estimated: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    dtype = torch.float32
    nodes_cpu = scene["nodes"].to(torch.float64)
    occupancy, region_source = material_occupancy(
        nodes_cpu, center=inclusion_center, radius=inclusion_radius
    )
    packs = build_track_observations(
        scene,
        experiments,
        occupancy=occupancy,
        center=inclusion_center,
        radius=inclusion_radius,
        track_noise_px=config.track_noise_px,
        minimum_track_confidence=config.minimum_track_confidence,
        region_track_weight=config.region_track_weight,
        track_smoothing_iterations=config.track_smoothing_iterations,
    )
    n_nodes = int(scene["Nn"])
    dim = 3
    nodes = _as_device(nodes_cpu, device, dtype)
    elems = scene["elems"].to(device=device)
    occupancy_dev = _as_device(occupancy, device, dtype)
    nu = torch.as_tensor(float(scene["nu_true"]), device=device, dtype=dtype)
    bbox_min = nodes.min(dim=0).values
    bbox_ext = (nodes.max(dim=0).values - bbox_min).clamp_min(1e-6)
    x_norm = 2.0 * (nodes - bbox_min) / bbox_ext - 1.0
    free_mask = _fixed_dof_mask(scene["fixed"], n_nodes, dim, device)
    surface_nodes = scene["surface_node_ids"].to(device=device, dtype=torch.long)
    n_loads = len(experiments)
    net = DisplacementPINN(n_loads=n_loads, width=config.width, depth=config.depth).to(device)
    log_E = nn.Parameter(
        torch.tensor(math.log(initial_E_background), device=device, dtype=dtype)
    )
    log_rho = nn.Parameter(torch.tensor(math.log(initial_ratio), device=device, dtype=dtype))
    log_q = nn.Parameter(torch.zeros((), device=device, dtype=dtype))
    optimizer = torch.optim.Adam(
        [
            {"params": net.parameters(), "lr": config.learning_rate},
            {"params": [log_E, log_rho, log_q], "lr": config.learning_rate * 1.5},
        ]
    )
    log_bounds = (
        (math.log(1000.0), math.log(15000.0)),
        (math.log(1.0), math.log(5.0)),
        (math.log(0.25), math.log(4.0)),
    )
    prior_center = torch.tensor(
        [math.log(initial_E_background), math.log(initial_ratio)],
        device=device,
        dtype=dtype,
    )
    device_packs = []
    for pack in packs:
        device_packs.append(
            {
                **pack,
                "mask": pack["mask"].to(device),
                "observation": _as_device(pack["observation"], device, dtype),
                "weight": _as_device(pack["weight"], device, dtype),
                "pose": _as_device(pack["pose"], device, dtype),
                "force": _as_device(pack["force"], device, dtype),
            }
        )
    load_ids = {
        load_index: torch.full((n_nodes,), load_index, device=device, dtype=torch.long)
        for load_index in range(n_loads)
    }
    history: list[dict[str, float]] = []
    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        E_bg = log_E.exp()
        ratio = log_rho.exp()
        force_scale = log_q.exp()
        E_nodes = parameterized_E_nodes(occupancy_dev, E_bg, ratio)
        mu, lam, _ = node_E_to_per_elem_mu_lam(E_nodes, elems, nu)
        data_terms = []
        phys_terms = []
        displacements: dict[int, torch.Tensor] = {}
        force_by_load = {item["load_index"]: item["force"] for item in device_packs}
        for load_index in range(n_loads):
            raw_u = net(x_norm, load_ids[load_index]).reshape(-1)
            u_flat = raw_u * free_mask.to(raw_u.dtype)
            displacements[load_index] = u_flat
            fint = assemble_internal_force(u_flat, nodes, elems, mu, lam, n_nodes, dim)
            residual = fint - force_scale * force_by_load[load_index]
            force_norm = force_by_load[load_index].square().mean().clamp_min(1e-12)
            phys_terms.append(residual[free_mask].square().mean() / force_norm)
        for item in device_packs:
            u_flat = displacements[item["load_index"]]
            deformed = nodes + u_flat.view(n_nodes, dim)
            pose = item["pose"]
            rest_uv, _, _ = project(
                nodes[surface_nodes],
                pose,
                focal=float(item["intrinsics"]["focal_px"]),
                H=int(item["intrinsics"]["height_px"]),
                W=int(item["intrinsics"]["width_px"]),
            )
            pred_uv, _, _ = project(
                deformed[surface_nodes],
                pose,
                focal=float(item["intrinsics"]["focal_px"]),
                H=int(item["intrinsics"]["height_px"]),
                W=int(item["intrinsics"]["width_px"]),
            )
            predicted = (pred_uv - rest_uv)[item["mask"]]
            point_loss = _huber((predicted - item["observation"]) / item["scale"]).mean(dim=-1)
            data_terms.append(
                (point_loss * item["weight"]).sum() / item["weight"].sum().clamp_min(1e-8)
            )
        data_loss = torch.stack(data_terms).sum()
        phys_loss = torch.stack(phys_terms).mean()
        prior_loss = log_q.square() / (2.0 * config.force_prior_sigma**2)
        if material_prior_sigma is not None and config.material_prior_weight > 0:
            sigma = torch.tensor(material_prior_sigma, device=device, dtype=dtype)
            params = torch.stack([log_E, log_rho])
            prior_loss = prior_loss + 0.5 * config.material_prior_weight * (
                ((params - prior_center) / sigma).square().sum()
            )
        loss = data_loss + config.physics_weight * phys_loss + prior_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            log_E.clamp_(*log_bounds[0])
            log_rho.clamp_(*log_bounds[1])
            log_q.clamp_(*log_bounds[2])
        if step == 0 or step + 1 == config.steps or (step + 1) % 200 == 0:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "data": float(data_loss.detach()),
                    "physics": float(phys_loss.detach()),
                    "prior": float(prior_loss.detach()),
                }
            )

    E_background = float(log_E.exp().detach().cpu())
    ratio = float(log_rho.exp().detach().cpu())
    force_scale = float(log_q.exp().detach().cpu())
    first = experiments[0]
    true_E = float(first["E_background"])
    true_ratio = float(first["inclusion_ratio"])
    nodes_np = nodes_cpu
    minimum = nodes_np.amin(dim=0)
    extent = nodes_np.amax(dim=0) - minimum
    lmin = float(min(float(extent[0]), float(extent[1])))
    if center_fraction_estimated is None and inclusion_center is not None:
        center_fraction_estimated = ((inclusion_center - minimum) / extent).tolist()
    if radius_fraction_estimated is None and inclusion_radius is not None and lmin > 0:
        radius_fraction_estimated = float(inclusion_radius) / lmin
    true_center = first["inclusion_center"].to(torch.float64)
    true_center_frac = ((true_center - minimum) / extent).cpu().tolist()
    true_radius_frac = float(first["inclusion_radius"]) / lmin if lmin > 0 else 0.0
    heterogeneous = true_ratio > 1.05
    center_error = None
    radius_error = None
    if heterogeneous and center_fraction_estimated is not None:
        delta = torch.tensor(center_fraction_estimated, dtype=torch.float64) - torch.tensor(
            true_center_frac, dtype=torch.float64
        )
        center_error = float(torch.linalg.vector_norm(delta))
    if heterogeneous and radius_fraction_estimated is not None and true_radius_frac > 0:
        radius_error = abs(radius_fraction_estimated / true_radius_frac - 1.0)
    return {
        "patient_id": str(first["patient_id"]),
        "method": "pinn_inverse",
        "E_background_true": true_E,
        "E_background_estimated": E_background,
        "E_background_relative_error": relative_error(E_background, true_E),
        "inclusion_ratio_true": true_ratio,
        "inclusion_ratio_estimated": ratio,
        "inclusion_ratio_relative_error": relative_error(ratio, true_ratio),
        "force_scale_true": force_scale_target(experiments),
        "force_scale_estimated": force_scale,
        "force_scale_relative_error": relative_error(force_scale, force_scale_target(experiments)),
        "center_fraction_true": true_center_frac,
        "center_fraction_estimated": center_fraction_estimated,
        "radius_fraction_true": true_radius_frac,
        "radius_fraction_estimated": radius_fraction_estimated,
        "center_error_normalized": center_error,
        "radius_relative_error": radius_error,
        "heterogeneous_true": heterogeneous,
        "region_source": region_source,
        "used_predicted_region": inclusion_center is not None,
        "initial_E_background": float(initial_E_background),
        "initial_ratio": float(initial_ratio),
        "function_evaluations": int(config.steps * n_loads),
        "steps": config.steps,
        "physics_weight": config.physics_weight,
        "wall_time_seconds": time.perf_counter() - started,
        "final_loss": history[-1] if history else {},
        "loss_history": history,
        "optimizer_success": True,
        "device": str(device),
    }
