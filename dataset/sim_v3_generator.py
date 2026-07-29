"""Publication-scale sim_v3 with controlled complexity and fixed splits."""
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

from dataset.real_colon_fem import build_real_colon_scene  # noqa: E402
from physics import deformation_gradient  # noqa: E402
from physics.fem import (  # noqa: E402
    make_heterogeneous_E_field,
    solve_nh,
    solve_nh_heterogeneous,
)
from rendering.gaussian_pbr import render, seed_surface_gaussians, set_albedo  # noqa: E402
from simulator.anatomy import ANATOMY_REGISTRY, build_anatomy_scene  # noqa: E402
from simulator.scene import contact_force_sequence, make_camera_poses  # noqa: E402


PROCEDURAL = list(ANATOMY_REGISTRY)
SEGMENTS = ["ascend", "cecum", "desc", "full", "sigmoid", "trans"]
GEOMETRIES = PROCEDURAL + [f"c3vd_{segment}" for segment in SEGMENTS]
E_VALUES = [3e3, 8e3, 1.2e4]
ALBEDOS = [(0.82, 0.40, 0.35), (0.90, 0.70, 0.55), (0.75, 0.30, 0.45)]
ROUGHNESSES = [0.35, 0.55]
PATTERNS = ["press_release", "drag"]
NOISE_SIGMAS = [0.0, 0.02, 0.05]
FORCE_SCALES = [0.7, 1.0, 1.3]


def split_for_id(scene_id: int) -> str:
    remainder = scene_id % 6
    if remainder == 0:
        return "test"
    if remainder == 1:
        return "val"
    return "train"


def factors_for_id(scene_id: int) -> dict:
    geometry = GEOMETRIES[scene_id % len(GEOMETRIES)]
    stiffness_mode = "inclusion" if (scene_id // len(GEOMETRIES)) % 2 else "homogeneous"
    return {
        "geometry": geometry,
        "stiffness_mode": stiffness_mode,
        "E_bg": E_VALUES[(scene_id // 2) % len(E_VALUES)],
        "E_ratio": 4.0 if stiffness_mode == "inclusion" else 1.0,
        "albedo": ALBEDOS[(scene_id * 5) % len(ALBEDOS)],
        "roughness": ROUGHNESSES[(scene_id // 3) % len(ROUGHNESSES)],
        "pattern": PATTERNS[(scene_id // 5) % len(PATTERNS)],
        "noise_sigma": NOISE_SIGMAS[(scene_id // 7) % len(NOISE_SIGMAS)],
        "force_scale": FORCE_SCALES[(scene_id // 11) % len(FORCE_SCALES)],
        "pose_jitter": bool((scene_id // 13) % 2),
        "optical_coupling": bool((scene_id // 17) % 2),
        "split": split_for_id(scene_id),
        "seed": 2026 + scene_id,
    }


def _build_scene(factors: dict) -> tuple[dict, float]:
    geometry = factors["geometry"]
    if geometry.startswith("c3vd_"):
        segment = geometry.removeprefix("c3vd_")
        path = ROOT / "dataset" / "C3VD" / f"{segment}_model.obj"
        scene = build_real_colon_scene(
            str(path), segment, E_true=factors["E_bg"], seed=factors["seed"]
        )
        base_force = 15.0
    else:
        scene = build_anatomy_scene(
            geometry, E_true=factors["E_bg"], nu_true=0.45
        )
        base_force = 60.0
    return scene, base_force


def _contact_and_poses(scene: dict, factors: dict, T: int):
    nodes = scene["nodes"]
    center = nodes.mean(dim=0)
    surface_nodes = torch.unique(scene["surface_tris"].reshape(-1))
    surface_center = nodes[surface_nodes].mean(dim=0)
    front_z = float(nodes[surface_nodes, 2].max()) + 1e-4
    contact_radius = 0.18 * min(float(scene["lx"]), float(scene["ly"]))
    forces, contact_log = contact_force_sequence(
        scene,
        T=T,
        max_force=(
            (15.0 if factors["geometry"].startswith("c3vd_") else 60.0)
            * factors["force_scale"]
        ),
        contact_center=(
            float(surface_center[0]),
            float(surface_center[1]),
            float(surface_center[2]),
        ),
        contact_radius=contact_radius,
        pattern=factors["pattern"],
        front_z=front_z,
    )
    extent = max(float(scene["lx"]), float(scene["ly"]))
    radius = 1.6 * extent
    height = float(center[2]) + 1.2 * extent
    look_at = center.clone()
    if factors["pose_jitter"]:
        rng = np.random.default_rng(factors["seed"])
        radius *= float(1.0 + rng.uniform(-0.10, 0.10))
        height *= float(1.0 + rng.uniform(-0.05, 0.05))
        look_at[:2] += torch.as_tensor(
            rng.uniform(-0.02, 0.02, size=2) * extent,
            dtype=look_at.dtype,
        )
    poses = make_camera_poses(
        T=T,
        radius=radius,
        height=height,
        look_at=tuple(float(value) for value in look_at),
    )
    return forces, contact_log, poses


def _minimum_jacobian(scene: dict, displacement: torch.Tensor) -> float:
    nodes, elems = scene["nodes"], scene["elems"]
    reference = nodes[elems]
    deformed = (nodes + displacement.view(scene["Nn"], 3))[elems]
    return float(torch.det(deformation_gradient(deformed, reference)).min())


def generate_scene(
    scene_id: int,
    out_root: Path,
    T: int = 12,
    H: int = 128,
    W: int = 128,
) -> dict:
    factors = factors_for_id(scene_id)
    torch.manual_seed(factors["seed"])
    np.random.seed(factors["seed"])
    scene, _ = _build_scene(factors)
    gaussians = seed_surface_gaussians(scene, gaussians_per_tri=3)
    set_albedo(gaussians, factors["albedo"])
    gaussians["roughness"] = torch.full_like(
        gaussians["roughness"], factors["roughness"]
    )
    if factors["optical_coupling"]:
        gaussians["coupling_k_strain"] = torch.tensor(0.35)
        gaussians["coupling_k_perf"] = torch.tensor(0.20)
    forces, contact_log, poses = _contact_and_poses(scene, factors, T)

    E_inc = factors["E_bg"] * factors["E_ratio"]
    center = scene["nodes"].mean(dim=0)
    inclusion_radius = 0.20 * min(float(scene["lx"]), float(scene["ly"]))
    E_nodes = make_heterogeneous_E_field(
        scene["nodes"],
        center,
        inclusion_radius,
        factors["E_bg"],
        E_inc,
    )
    original_forces = forces
    attenuation = 1.0
    for attempt in range(4):
        forces = [force * attenuation for force in original_forces]
        clean_images, observed_images, displacements, min_j = [], [], [], math.inf
        rng = torch.Generator().manual_seed(factors["seed"] + 100_000)
        for force, pose in zip(forces, poses):
            with torch.no_grad():
                if factors["stiffness_mode"] == "inclusion":
                    displacement = solve_nh_heterogeneous(
                        scene["nodes"],
                        scene["elems"],
                        E_nodes,
                        scene["nu_true"],
                        force,
                        scene["fixed"],
                        D=scene["D"],
                    )
                else:
                    displacement = solve_nh(
                        scene["nodes"],
                        scene["elems"],
                        scene["E_true"],
                        scene["nu_true"],
                        force,
                        scene["fixed"],
                        D=scene["D"],
                    )
                clean = render(
                    gaussians,
                    scene,
                    displacement,
                    pose,
                    H=H,
                    W=W,
                    light_intensity=2.0,
                ).clamp(0, 1)
                noise = torch.randn(clean.shape, generator=rng, dtype=clean.dtype)
                observed = (clean + factors["noise_sigma"] * noise).clamp(0, 1)
            clean_images.append(clean)
            observed_images.append(observed)
            displacements.append(displacement)
            min_j = min(min_j, _minimum_jacobian(scene, displacement))
        if math.isfinite(min_j) and min_j > 0.05:
            break
        attenuation *= 0.5
    if not math.isfinite(min_j) or min_j <= 0.05:
        raise RuntimeError(f"scene {scene_id} invalid minimum Jacobian {min_j}")
    contact_log = [
        {**entry, "force_mag": entry["force_mag"] * attenuation}
        for entry in contact_log
    ]

    clean_tensor = torch.stack(clean_images)
    observed_tensor = torch.stack(observed_images)
    displacement_tensor = torch.stack(displacements)
    scene_dir = out_root / f"scene_{scene_id:04d}"
    (scene_dir / "images").mkdir(parents=True, exist_ok=True)
    (scene_dir / "images_clean").mkdir(parents=True, exist_ok=True)
    for directory, tensor in (
        (scene_dir / "images", observed_tensor),
        (scene_dir / "images_clean", clean_tensor),
    ):
        uint8 = (tensor * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for frame, image in enumerate(uint8):
            Image.fromarray(image).save(directory / f"frame_{frame:02d}.png")

    gt = {
        **factors,
        "effective_force_scale": factors["force_scale"] * attenuation,
        "E": float(factors["E_bg"]),
        "E_bg": float(factors["E_bg"]),
        "E_inc": float(E_inc),
        "inclusion_center": center,
        "inclusion_radius": float(inclusion_radius),
        "nu": 0.45,
        "albedo": list(factors["albedo"]),
        "T": T,
        "H": H,
        "W": W,
        "u_seq": displacement_tensor,
        "forces": torch.stack(forces),
        "poses": poses,
        "contact_log": contact_log,
        "clean_images_uint8": (clean_tensor * 255).to(torch.uint8),
        "scene_spec": {
            "anatomy": factors["geometry"],
            "nodes": scene["nodes"],
            "elems": scene["elems"],
            "fixed": scene["fixed"],
            "surface_tris": scene["surface_tris"],
            "Nn": scene["Nn"],
            "D": scene["D"],
        },
    }
    torch.save(gt, scene_dir / "gt.pt")
    return {
        "id": scene_id,
        **factors,
        "effective_force_scale": factors["force_scale"] * attenuation,
        "E_inc": float(E_inc),
        "inclusion_radius": float(inclusion_radius),
        "max_displacement": float(displacement_tensor.abs().max()),
        "minimum_jacobian": min_j,
    }


def generate_dataset(
    out_root: Path,
    count: int = 144,
    start: int = 0,
    limit: int | None = None,
    overwrite: bool = False,
) -> list[dict]:
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = manifest.get("scenes", [])
    else:
        rows = []
    done = {row["id"] for row in rows}
    stop = count if limit is None else min(count, start + limit)
    for scene_id in range(start, stop):
        if overwrite and scene_id in done:
            rows = [row for row in rows if row["id"] != scene_id]
            done.remove(scene_id)
        if scene_id in done:
            continue
        row = generate_scene(scene_id, out_root)
        rows.append(row)
        rows.sort(key=lambda item: item["id"])
        manifest_path.write_text(
            json.dumps(
                {
                    "version": "sim_v3",
                    "n_scenes": len(rows),
                    "target_scenes": count,
                    "split_counts": {
                        split: sum(row["split"] == split for row in rows)
                        for split in ("train", "val", "test")
                    },
                    "scenes": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"scene {scene_id:04d} {row['geometry']} {row['stiffness_mode']} "
            f"split={row['split']} max|u|={row['max_displacement']:.3f} "
            f"minJ={row['minimum_jacobian']:.3f}",
            flush=True,
        )
    return rows


def main() -> None:
    torch.set_default_dtype(torch.float64)
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "dataset" / "sim_v3")
    parser.add_argument("--count", type=int, default=144)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate_dataset(
        args.out,
        count=args.count,
        start=args.start,
        limit=args.limit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
