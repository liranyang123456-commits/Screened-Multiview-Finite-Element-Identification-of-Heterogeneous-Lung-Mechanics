"""Generate a patient-geometry-split local-lung synthetic inverse benchmark."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lung_inverse_rendering.ct_geometry import (
    build_scene_from_ct_mesh,
    make_synthetic_ct_surrogate,
)
from physics import deformation_gradient
from physics.fem import make_heterogeneous_E_field, solve_nh_heterogeneous
from rendering.gaussian_pbr import render, seed_surface_gaussians, set_albedo
from simulator.scene import contact_force_sequence, make_camera_poses


DEFAULT_OUT = ROOT / "dataset" / "sim_lung_v1"
E_VALUES = (2.0e3, 5.0e3, 9.0e3)
RATIOS = (1.0, 2.0, 4.0)
ALBEDOS = ((0.78, 0.38, 0.35), (0.88, 0.55, 0.48), (0.70, 0.30, 0.32))


def split_for_patient(patient_index: int, patient_count: int) -> str:
    if patient_count < 5:
        raise ValueError("at least five geometries are required for a 60/20/20 split")
    train_count = int(patient_count * 0.60)
    val_count = int(patient_count * 0.20)
    if patient_index < train_count:
        return "train"
    if patient_index < train_count + val_count:
        return "val"
    return "test"


def factor_row(scene_id: int, patient_index: int, patient_count: int) -> dict:
    return {
        "scene_id": scene_id,
        "geometry_id": f"lung_geometry_{patient_index:03d}",
        "patient_split": split_for_patient(patient_index, patient_count),
        "E_background": E_VALUES[scene_id % len(E_VALUES)],
        "lesion_ratio": RATIOS[(scene_id // len(E_VALUES)) % len(RATIOS)],
        "albedo": ALBEDOS[(scene_id // 3) % len(ALBEDOS)],
        "roughness": (0.35, 0.55)[(scene_id // 5) % 2],
        "force_scale": (0.55, 0.8, 1.0)[(scene_id // 7) % 3],
        "pattern": ("press_release", "drag")[(scene_id // 11) % 2],
        "noise_sigma": (0.0, 0.01, 0.03)[(scene_id // 13) % 3],
        "seed": 51_000 + scene_id,
    }


def mesh_for_patient(mesh_root: Path | None, patient_index: int) -> Path | None:
    if mesh_root is None:
        return None
    candidates = sorted(
        path for extension in ("*.obj", "*.ply", "*.npz") for path in mesh_root.glob(extension)
    )
    return candidates[patient_index] if patient_index < len(candidates) else None


def minimum_jacobian(scene: dict, displacement: torch.Tensor) -> float:
    reference = scene["nodes"][scene["elems"]]
    deformed = (scene["nodes"] + displacement.view(scene["Nn"], 3))[scene["elems"]]
    return float(torch.det(deformation_gradient(deformed, reference)).min())


def generate_scene(
    factors: dict,
    out_root: Path,
    *,
    mesh_path: Path | None,
    frames: int,
    resolution: int,
) -> dict:
    torch.manual_seed(factors["seed"])
    np.random.seed(factors["seed"])
    if mesh_path is None:
        scene = make_synthetic_ct_surrogate(
            factors["geometry_id"], seed=factors["seed"] // 31, E_true=factors["E_background"]
        )
    else:
        scene = build_scene_from_ct_mesh(
            mesh_path, geometry_id=factors["geometry_id"], E_true=factors["E_background"]
        )

    center = scene["nodes"].mean(dim=0)
    radius = 0.18 * min(scene["lx"], scene["ly"])
    E_inclusion = factors["E_background"] * factors["lesion_ratio"]
    E_nodes = make_heterogeneous_E_field(
        scene["nodes"], center, radius, factors["E_background"], E_inclusion
    )
    front_z = float(scene["nodes"][torch.unique(scene["surface_tris"].reshape(-1)), 2].max()) + 1e-4
    forces, contact_log = contact_force_sequence(
        scene,
        T=frames,
        max_force=22.0 * factors["force_scale"],
        contact_center=(float(center[0]), float(center[1]), float(center[2])),
        contact_radius=radius,
        pattern=factors["pattern"],
        front_z=front_z,
    )
    extent = max(scene["lx"], scene["ly"])
    poses = make_camera_poses(
        T=frames,
        radius=1.55 * extent,
        height=float(center[2]) + extent,
        look_at=tuple(float(item) for item in center),
    )
    gaussians = seed_surface_gaussians(scene, gaussians_per_tri=3)
    set_albedo(gaussians, factors["albedo"])
    gaussians["roughness"] = torch.full_like(gaussians["roughness"], factors["roughness"])

    displacements, images, min_j = [], [], math.inf
    generator = torch.Generator().manual_seed(factors["seed"] + 1)
    for force, pose in zip(forces, poses):
        displacement = solve_nh_heterogeneous(
            scene["nodes"], scene["elems"], E_nodes, scene["nu_true"], force, scene["fixed"]
        )
        min_j = min(min_j, minimum_jacobian(scene, displacement))
        if min_j <= 0.05:
            raise RuntimeError(f"unstable FEM scene {factors['scene_id']}: minJ={min_j:.3f}")
        clean = render(gaussians, scene, displacement, pose, H=resolution, W=resolution, light_intensity=2.0)
        noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
        images.append((clean + factors["noise_sigma"] * noise).clamp(0, 1))
        displacements.append(displacement)

    scene_dir = out_root / f"scene_{factors['scene_id']:04d}"
    (scene_dir / "images").mkdir(parents=True, exist_ok=True)
    image_tensor = torch.stack(images)
    image_uint8 = (image_tensor * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    for frame, image in enumerate(image_uint8):
        Image.fromarray(image).save(scene_dir / "images" / f"frame_{frame:02d}.png")
    torch.save(
        {
            "schema_version": "sim_lung_v1",
            "geometry_id": factors["geometry_id"],
            "geometry_source": scene["geometry_source"],
            "E_background": factors["E_background"],
            "E_inclusion": E_inclusion,
            "lesion_ratio": factors["lesion_ratio"],
            "candidate_fibre_direction": [1.0, 0.0, 0.0],
            "fibre_model_status": "metadata_only_not_used_by_isotropic_v1_fem",
            "nodes": scene["nodes"],
            "elems": scene["elems"],
            "surface_tris": scene["surface_tris"],
            "fixed": scene["fixed"],
            "forces": torch.stack(forces),
            "poses": poses,
            "u_seq": torch.stack(displacements),
            "images_uint8": image_uint8,
            "contact_log": contact_log,
        },
        scene_dir / "gt.pt",
    )
    return {
        **factors,
        "geometry_source": scene["geometry_source"],
        "E_inclusion": E_inclusion,
        "minimum_jacobian": min_j,
        "frames": frames,
        "resolution": resolution,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--patients", type=int, default=10)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--ct-mesh-root", type=Path)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene_id in range(args.count):
        patient_index = scene_id % args.patients
        factors = factor_row(scene_id, patient_index, args.patients)
        row = generate_scene(
            factors,
            args.out,
            mesh_path=mesh_for_patient(args.ct_mesh_root, patient_index),
            frames=args.frames,
            resolution=args.resolution,
        )
        rows.append(row)
        print(
            f"scene={scene_id:04d} geometry={row['geometry_id']} split={row['patient_split']} "
            f"source={row['geometry_source']} minJ={row['minimum_jacobian']:.3f}",
            flush=True,
        )
    manifest = {
        "version": "sim_lung_v1",
        "patient_level_split": True,
        "raw_patient_identifiers_persisted": False,
        "count": len(rows),
        "split_counts": {
            split: sum(row["patient_split"] == split for row in rows)
            for split in ("train", "val", "test")
        },
        "mechanics_scope": "heterogeneous isotropic Neo-Hookean; fibre metadata only",
        "scenes": rows,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
