"""De-identified CT-mesh adapter for local lung/airway-wall FEM patches.

Raw DICOM is intentionally not consumed by this module.  A separate approved
preprocessing workflow must export a de-identified surface mesh; this prevents
patient tags and source paths entering the benchmark manifest.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from physics import make_tet_grid
from simulator.anatomy import _surface_tris_by_centroid


def _scene_from_nodes(
    nodes: torch.Tensor,
    elems: torch.Tensor,
    *,
    geometry_id: str,
    geometry_source: str,
    E_true: float,
    nu_true: float = 0.45,
) -> dict:
    nodes = nodes.to(torch.float64)
    nodes = nodes - nodes.min(dim=0).values
    scale = float(nodes[:, :2].max().clamp_min(1e-6))
    nodes = nodes / scale
    surface_tris = _surface_tris_by_centroid(nodes, elems, "front", 0.25)
    back_cut = torch.quantile(nodes[:, 2], 0.80)
    back = torch.where(nodes[:, 2] >= back_cut)[0]
    fixed = torch.cat([3 * back, 3 * back + 1, 3 * back + 2])
    return {
        "nodes": nodes,
        "elems": elems,
        "Nn": len(nodes),
        "D": 3,
        "fixed": fixed,
        "surface_tris": surface_tris,
        "E_true": torch.tensor(float(E_true), dtype=torch.float64),
        "nu_true": torch.tensor(float(nu_true), dtype=torch.float64),
        "lx": float(nodes[:, 0].max()),
        "ly": float(nodes[:, 1].max()),
        "lz": float(nodes[:, 2].max() - nodes[:, 2].min()),
        "geometry_id": geometry_id,
        "geometry_source": geometry_source,
    }


def make_synthetic_ct_surrogate(
    geometry_id: str, *, seed: int, E_true: float
) -> dict:
    """Create an explicitly synthetic, patient-shaped local lung-wall patch.

    The geometry perturbation represents variation in local curvature and wall
    thickness only.  It is never labelled as a patient CT geometry.
    """
    generator = torch.Generator().manual_seed(seed)
    nodes, elems = make_tet_grid(7, 7, 3, 1.0, 0.82, 0.28)
    x = nodes[:, 0] - 0.5
    y = nodes[:, 1] / 0.82 - 0.5
    curvature = float(0.06 + 0.10 * torch.rand((), generator=generator))
    obliquity = float(-0.08 + 0.16 * torch.rand((), generator=generator))
    front = nodes[:, 2] < 1e-6
    nodes[:, 2] -= front * (curvature * (x.square() + 1.4 * y.square()) + obliquity * x * y)
    nodes[:, 0] += front * 0.05 * torch.sin(2.0 * torch.pi * nodes[:, 1] / 0.82)
    return _scene_from_nodes(
        nodes,
        elems,
        geometry_id=geometry_id,
        geometry_source="synthetic_ct_surrogate",
        E_true=E_true,
    )


def build_scene_from_ct_mesh(mesh_path: Path, *, geometry_id: str, E_true: float) -> dict:
    """Convert one approved de-identified surface mesh into a thin FEM patch."""
    if mesh_path.suffix.lower() == ".npz":
        mesh = np.load(mesh_path)
        vertices, faces = mesh["vertices"], mesh["faces"]
    else:
        try:
            import open3d as o3d
        except ImportError as error:
            raise RuntimeError(
                "CT mesh import requires open3d; install it in the simulation environment."
            ) from error
        loaded = o3d.io.read_triangle_mesh(str(mesh_path))
        vertices, faces = np.asarray(loaded.vertices), np.asarray(loaded.triangles)
    if len(vertices) < 50 or len(faces) < 20:
        raise ValueError("CT mesh must contain at least 50 vertices and 20 triangles")

    # Reuse the established real-surface -> local thin-shell conversion.  It
    # preserves local aspect ratio/orientation while keeping FEM sizes tractable.
    from dataset.real_colon_fem import patch_to_tet_shell

    center = vertices.mean(axis=0)
    distance = np.linalg.norm(vertices - center, axis=1)
    keep = distance <= np.quantile(distance, 0.20)
    index = -np.ones(len(vertices), dtype=np.int64)
    index[keep] = np.arange(keep.sum())
    patch_faces = index[faces]
    patch_faces = patch_faces[(patch_faces >= 0).all(axis=1)]
    patch_vertices = vertices[keep]
    if len(patch_faces) < 20:
        raise ValueError("CT mesh central patch has insufficient connected faces")
    centered = patch_vertices - patch_vertices.mean(axis=0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    normal = vectors[-1]
    nodes, elems = patch_to_tet_shell(patch_vertices, patch_faces, normal, grid_nx=7, grid_ny=7)
    return _scene_from_nodes(
        nodes,
        elems,
        geometry_id=geometry_id,
        geometry_source="approved_deidentified_ct_mesh",
        E_true=E_true,
    )
