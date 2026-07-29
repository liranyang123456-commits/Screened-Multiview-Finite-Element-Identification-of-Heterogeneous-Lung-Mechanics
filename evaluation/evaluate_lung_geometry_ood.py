"""Geometry-only OOD validation for de-identified ION CT meshes.

No material labels are assumed.  Consequently this module never computes
Young's-modulus or modulus-ratio errors.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.sim_lung_graph import tetrahedra_to_edge_index  # noqa: E402
from experiments.train_lung_mesh_gnn import build_model  # noqa: E402
from lung_inverse_rendering.ct_geometry import (  # noqa: E402
    build_scene_from_ct_mesh,
    make_synthetic_ct_surrogate,
)
from models.lung_mesh_material_gnn import (  # noqa: E402
    LOG_E_BOUNDS,
    LOG_RATIO_BOUNDS,
    LOGVAR_BOUNDS,
    RADIUS_BOUNDS,
)
from physics import deformation_gradient  # noqa: E402


def surface_mesh_qc(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    valid_shape = (
        vertices.ndim == 2
        and vertices.shape[1:] == (3,)
        and faces.ndim == 2
        and faces.shape[1:] == (3,)
    )
    if not valid_shape:
        return {"valid": False, "reason": "invalid_array_shape"}
    indices_valid = bool(
        len(vertices) > 0
        and len(faces) > 0
        and faces.min(initial=0) >= 0
        and faces.max(initial=-1) < len(vertices)
    )
    finite = bool(np.isfinite(vertices).all())
    if indices_valid and finite:
        triangles = vertices[faces]
        double_area = np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=1,
        )
        degenerate_faces = int(np.count_nonzero(double_area <= 1e-12))
        extent = np.ptp(vertices, axis=0)
    else:
        degenerate_faces = len(faces)
        extent = np.zeros(3)
    return {
        "valid": bool(indices_valid and finite and degenerate_faces == 0),
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "finite_vertices": finite,
        "valid_indices": indices_valid,
        "degenerate_face_count": degenerate_faces,
        "nonzero_extent_axes": int(np.count_nonzero(extent > 0)),
    }


def fem_mesh_qc(scene: dict[str, Any], displacement: torch.Tensor | None = None) -> dict[str, Any]:
    nodes = torch.as_tensor(scene["nodes"], dtype=torch.float64)
    elems = torch.as_tensor(scene["elems"], dtype=torch.long)
    corners = nodes[elems]
    columns = torch.stack(
        [corners[:, index + 1] - corners[:, 0] for index in range(3)], dim=-1
    )
    determinants = torch.linalg.det(columns)
    absolute = determinants.abs()
    scale = torch.linalg.vector_norm(columns, dim=1).prod(dim=1).clamp_min(1e-15)
    normalized = absolute / scale
    result: dict[str, Any] = {
        "construction_success": True,
        "node_count": int(len(nodes)),
        "tetrahedron_count": int(len(elems)),
        "finite_nodes": bool(torch.isfinite(nodes).all()),
        "degenerate_tetrahedron_count": int((absolute <= 1e-12).sum()),
        "mixed_orientation": bool((determinants > 0).any() and (determinants < 0).any()),
        "minimum_normalized_rest_jacobian": float(normalized.min()),
        "rest_jacobian_positive_magnitude": bool((absolute > 1e-12).all()),
    }
    current = nodes if displacement is None else nodes + displacement.to(torch.float64)
    jacobian = torch.det(deformation_gradient(current[elems], corners))
    result["minimum_deformation_jacobian"] = float(jacobian.min())
    result["nonpositive_deformation_jacobian_count"] = int((jacobian <= 0).sum())
    return result


def _geometry_graph(scene: dict[str, Any], input_dim: int) -> dict[str, torch.Tensor]:
    if input_dim < 5 or (input_dim - 5) % 6:
        raise ValueError("Checkpoint input dimension is incompatible with lung graph features")
    nodes = torch.as_tensor(scene["nodes"], dtype=torch.float32)
    minimum = nodes.amin(dim=0)
    extent = (nodes.amax(dim=0) - minimum).clamp_min(torch.finfo(nodes.dtype).eps)
    coordinates = (nodes - minimum) / extent
    fixed_mask = torch.zeros(len(nodes), dtype=torch.float32)
    fixed_nodes = torch.unique(torch.as_tensor(scene["fixed"], dtype=torch.long) // 3)
    fixed_mask[fixed_nodes] = 1.0
    surface_mask = torch.zeros(len(nodes), dtype=torch.float32)
    surface_nodes = torch.unique(torch.as_tensor(scene["surface_tris"], dtype=torch.long))
    surface_mask[surface_nodes] = 1.0
    padding = torch.zeros((len(nodes), input_dim - 5), dtype=torch.float32)
    return {
        "x": torch.cat(
            (coordinates, fixed_mask[:, None], surface_mask[:, None], padding), dim=1
        ),
        "edge_index": tetrahedra_to_edge_index(torch.as_tensor(scene["elems"])),
    }


@torch.no_grad()
def model_stability(scene: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    model = build_model(
        config["model"],
        int(config["input_dim"]),
        int(config["hidden_dim"]),
        int(config["layers"]),
        float(config["dropout"]),
    )
    model.load_state_dict(state["model"])
    model.eval()
    output = model(_geometry_graph(scene, int(config["input_dim"])))
    log_e = float(output["log_E_background_mean"].squeeze())
    log_ratio = float(output["log_ratio_mean"].squeeze())
    radius = float(output["radius_fraction_mean"].squeeze())
    log_e_std = math.exp(0.5 * float(output["log_E_background_logvar"].squeeze()))
    log_ratio_std = math.exp(0.5 * float(output["log_ratio_logvar"].squeeze()))
    calibration = state.get("uncertainty_calibration_scales", {})
    log_e_std *= float(calibration.get("log_E", 1.0))
    log_ratio_std *= float(calibration.get("log_ratio", 1.0))
    finite = all(
        math.isfinite(value)
        for value in (log_e, log_ratio, radius, log_e_std, log_ratio_std)
    )
    return {
        "evaluated": True,
        "finite_outputs": finite,
        "within_declared_output_ranges": bool(
            LOG_E_BOUNDS[0] <= log_e <= LOG_E_BOUNDS[1]
            and LOG_RATIO_BOUNDS[0] <= log_ratio <= LOG_RATIO_BOUNDS[1]
            and RADIUS_BOUNDS[0] <= radius <= RADIUS_BOUNDS[1]
            and math.exp(0.5 * LOGVAR_BOUNDS[0])
            <= log_e_std / float(calibration.get("log_E", 1.0))
            <= math.exp(0.5 * LOGVAR_BOUNDS[1])
        ),
        "predicted_log_E_range_value": log_e,
        "predicted_log_ratio_range_value": log_ratio,
        "predicted_radius_fraction": radius,
        "log_E_predictive_std": log_e_std,
        "log_ratio_predictive_std": log_ratio_std,
        "interpretation": "stability_only_without_ground_truth",
    }


def motion_metrics(
    motion_path: Path, scene: dict[str, Any]
) -> tuple[dict[str, Any], torch.Tensor | None]:
    data = np.load(motion_path)
    observed = np.asarray(data["observed"], dtype=np.float64)
    reconstructed = np.asarray(data["reconstructed"], dtype=np.float64)
    if observed.shape != reconstructed.shape:
        raise ValueError("Observed and reconstructed motion must have identical shapes")
    difference = reconstructed - observed
    residual = {
        "evaluated": True,
        "sample_count": int(observed.size),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "mae": float(np.mean(np.abs(difference))),
    }
    displacement = None
    if reconstructed.shape == tuple(torch.as_tensor(scene["nodes"]).shape):
        displacement = torch.from_numpy(reconstructed)
    return residual, displacement


def evaluate_geometry(
    *,
    mesh_path: Path | None = None,
    geometry_id: str = "synthetic_surrogate",
    checkpoint: Path | None = None,
    motion_path: Path | None = None,
    seed: int = 2026,
) -> dict[str, Any]:
    """Evaluate one mesh or an always-available synthetic surrogate."""
    surface_qc = None
    if mesh_path is None:
        scene = make_synthetic_ct_surrogate(geometry_id, seed=seed, E_true=5_000.0)
        geometry_source = "synthetic_ct_surrogate"
    else:
        with np.load(mesh_path) as mesh:
            surface_qc = surface_mesh_qc(mesh["vertices"], mesh["faces"])
        scene = build_scene_from_ct_mesh(mesh_path, geometry_id=geometry_id, E_true=5_000.0)
        geometry_source = "deidentified_ct_mesh"

    motion = {"evaluated": False, "reason": "motion_not_provided"}
    displacement = None
    if motion_path is not None:
        motion, displacement = motion_metrics(motion_path, scene)
    result = {
        "schema_version": 1,
        "geometry_id": geometry_id,
        "geometry_source": geometry_source,
        "evidence_scope": "geometry_domain_stability_only",
        "surface_mesh_qc": surface_qc,
        "fem_mesh_qc": fem_mesh_qc(scene, displacement),
        "model_output_stability": (
            model_stability(scene, checkpoint)
            if checkpoint is not None
            else {"evaluated": False, "reason": "checkpoint_not_provided"}
        ),
        "motion_reconstruction_residual": motion,
        "ground_truth_material_metrics_reported": False,
        "privacy": {
            "raw_paths_reported": False,
            "uids_reported": False,
            "names_reported": False,
            "dicom_tags_reported": False,
        },
    }
    forbidden = ("relative_error", "ground_truth_E", "ground_truth_ratio")
    serialized = json.dumps(result)
    if any(token in serialized for token in forbidden):
        raise AssertionError("Ground-truth material error field entered OOD report")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--geometry-id", default="synthetic_surrogate")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = evaluate_geometry(
        mesh_path=args.mesh,
        geometry_id=args.geometry_id,
        checkpoint=args.checkpoint,
        motion_path=args.motion,
        seed=args.seed,
    )
    text = json.dumps(result, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
