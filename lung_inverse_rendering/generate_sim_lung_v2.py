"""Generate patient-consistent lung mechanics under multiple known excitations.

Each virtual patient has one fixed geometry, material field and boundary
condition. Multiple pressure/shear experiments probe that same patient. This
is the minimum valid setup for patient-specific system identification.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lung_inverse_rendering.ct_geometry import make_synthetic_ct_surrogate
from physics import deformation_gradient
from physics.fem import make_heterogeneous_E_field, solve_nh_heterogeneous
from rendering.gaussian_pbr import (
    gaussian_centers,
    project,
    project_with_foreground_confidence,
    render,
    seed_surface_gaussians,
    set_albedo,
)
from simulator.scene import make_camera_poses, make_multiview_camera_poses


DEFAULT_OUT = ROOT / "dataset" / "sim_lung_v2"
DEFAULT_MULTIVIEW_OUT = ROOT / "dataset" / "sim_lung_v2_multiview"
MULTIVIEW_SCHEMA_VERSION = "sim_lung_v2_multiview_v1"
NUM_MULTIVIEW_CAMERAS = 3
EXPERIMENTS = (
    {"name": "press_left", "center_xy": (0.36, 0.45), "direction": (0.0, 0.0, 1.0)},
    {"name": "press_right", "center_xy": (0.64, 0.55), "direction": (0.0, 0.0, 1.0)},
    {"name": "shear_x", "center_xy": (0.50, 0.50), "direction": (0.55, 0.0, 0.835)},
    {"name": "shear_y", "center_xy": (0.50, 0.50), "direction": (0.0, 0.55, 0.835)},
)
LOAD_ENVELOPE = (0.0, 0.45, 1.0, 1.0, 0.65, 0.25, 0.0)


def measured_force_fields(
    forces: torch.Tensor,
    *,
    true_force_scale_N: float,
    seed: int,
    prior_fraction: float = 0.05,
) -> dict:
    """Create deterministic measured-force perturbations with an independent seed."""
    if prior_fraction < 0:
        raise ValueError("Force prior fraction must be non-negative")
    generator = torch.Generator().manual_seed(seed)
    relative_error = prior_fraction * torch.randn(
        forces.shape[0], generator=generator, dtype=forces.dtype
    )
    measured = forces * (1.0 + relative_error[:, None])
    measured_resultants = torch.linalg.vector_norm(
        measured.view(measured.shape[0], -1, 3).sum(dim=1), dim=1
    )
    return {
        "forces_measured": measured,
        "force_measurement_relative_error": relative_error,
        "force_measurement_seed": seed,
        "force_measurement_prior_fraction": prior_fraction,
        "force_scale_true_N": float(true_force_scale_N),
        "force_scale_measured_N": float(measured_resultants.max()),
    }


def multiview_projection_fields(
    rest_points: torch.Tensor,
    deformed_points: torch.Tensor,
    poses_multiview: torch.Tensor,
    *,
    focal: float,
    resolution: int,
    prefix: str = "image",
    depth_softness: float = 0.02,
) -> dict:
    """Build ``(T,V,N,...)`` projection/depth/occlusion protocol fields."""
    if poses_multiview.ndim != 4 or poses_multiview.shape[1] != NUM_MULTIVIEW_CAMERAS:
        raise ValueError("poses_multiview must have shape (T,3,4,4)")
    if deformed_points.shape[0] != poses_multiview.shape[0]:
        raise ValueError("Deformation and camera sequences must be synchronized")

    rows = {
        "uv_rest": [],
        "uv_deformed": [],
        "depth_rest": [],
        "depth_deformed": [],
        "in_frame_rest": [],
        "in_frame_deformed": [],
        "zbuffer_depth_rest": [],
        "zbuffer_depth_deformed": [],
        "foreground_confidence_rest": [],
        "foreground_confidence_deformed": [],
    }
    for frame in range(poses_multiview.shape[0]):
        view_rows = {key: [] for key in rows}
        for view in range(poses_multiview.shape[1]):
            pose = poses_multiview[frame, view]
            rest_values = project_with_foreground_confidence(
                rest_points,
                pose,
                focal=focal,
                H=resolution,
                W=resolution,
                depth_softness=depth_softness,
            )
            deformed_values = project_with_foreground_confidence(
                deformed_points[frame],
                pose,
                focal=focal,
                H=resolution,
                W=resolution,
                depth_softness=depth_softness,
            )
            for name, value in zip(
                (
                    "uv_rest",
                    "depth_rest",
                    "in_frame_rest",
                    "zbuffer_depth_rest",
                    "foreground_confidence_rest",
                ),
                rest_values,
            ):
                view_rows[name].append(value)
            for name, value in zip(
                (
                    "uv_deformed",
                    "depth_deformed",
                    "in_frame_deformed",
                    "zbuffer_depth_deformed",
                    "foreground_confidence_deformed",
                ),
                deformed_values,
            ):
                view_rows[name].append(value)
        for key in rows:
            rows[key].append(torch.stack(view_rows[key]))
    fields = {
        f"{prefix}_{name}_multiview_seq": torch.stack(values)
        for name, values in rows.items()
    }
    fields[f"{prefix}_occlusion_confidence_multiview_seq"] = torch.minimum(
        fields[f"{prefix}_foreground_confidence_rest_multiview_seq"],
        fields[f"{prefix}_foreground_confidence_deformed_multiview_seq"],
    )
    fields[f"{prefix}_visibility_multiview_seq"] = (
        fields[f"{prefix}_in_frame_rest_multiview_seq"]
        & fields[f"{prefix}_in_frame_deformed_multiview_seq"]
        & (fields[f"{prefix}_occlusion_confidence_multiview_seq"] > 0.0)
    )
    return fields


def patient_split(
    patient_index: int,
    patient_count: int,
    train_count: int | None = None,
    val_count: int | None = None,
    interleaved: bool = False,
) -> str:
    if patient_count < 5:
        raise ValueError("At least five patients are required")
    if interleaved:
        fold = patient_index % 5
        return "train" if fold < 3 else "val" if fold == 3 else "test"
    train_end = int(0.60 * patient_count) if train_count is None else train_count
    validation_count = int(0.20 * patient_count) if val_count is None else val_count
    val_end = train_end + validation_count
    if train_end < 1 or validation_count < 1 or val_end >= patient_count:
        raise ValueError("Split must retain train, validation, and test patients")
    return "train" if patient_index < train_end else "val" if patient_index < val_end else "test"


def patient_spec(
    patient_index: int,
    patient_count: int,
    train_count: int | None = None,
    val_count: int | None = None,
    randomize_materials: bool = False,
    interleaved_split: bool = False,
) -> dict:
    """Return immutable material/geometry properties for one virtual patient."""
    E_values = (2600.0, 3400.0, 4300.0, 5200.0, 6500.0)
    ratios = (1.0, 1.5, 2.0, 2.8, 3.6)
    if randomize_materials:
        generator = torch.Generator().manual_seed(173_000 + patient_index)
        E_background = float(
            torch.exp(
                torch.empty((), dtype=torch.float64).uniform_(
                    np.log(2200.0), np.log(8000.0), generator=generator
                )
            )
        )
        homogeneous = bool(torch.rand((), generator=generator) < 0.20)
        ratio = (
            1.0
            if homogeneous
            else float(
                torch.empty((), dtype=torch.float64).uniform_(
                    1.3, 4.0, generator=generator
                )
            )
        )
        center_fraction = (
            float(torch.empty((), dtype=torch.float64).uniform_(0.35, 0.65, generator=generator)),
            float(torch.empty((), dtype=torch.float64).uniform_(0.35, 0.65, generator=generator)),
            float(torch.empty((), dtype=torch.float64).uniform_(0.42, 0.58, generator=generator)),
        )
        radius_fraction = float(
            torch.empty((), dtype=torch.float64).uniform_(0.12, 0.23, generator=generator)
        )
    else:
        E_background = E_values[patient_index % len(E_values)]
        ratio = ratios[(patient_index * 3) % len(ratios)]
        center_fraction = (
            0.42 + 0.04 * (patient_index % 3),
            0.46 + 0.03 * ((patient_index // 2) % 3),
            0.50,
        )
        radius_fraction = 0.16 + 0.015 * (patient_index % 3)
    return {
        "patient_id": f"lung_patient_{patient_index:03d}",
        "split": patient_split(
            patient_index,
            patient_count,
            train_count=train_count,
            val_count=val_count,
            interleaved=interleaved_split,
        ),
        "geometry_seed": 81_000 + patient_index,
        "material_seed": 173_000 + patient_index if randomize_materials else None,
        "E_background": E_background,
        "inclusion_ratio": ratio,
        "inclusion_center_fraction": center_fraction,
        "inclusion_radius_fraction": radius_fraction,
        "nu": 0.45,
        "boundary_condition": "posterior_20_percent_fully_fixed",
    }


def build_patient(
    spec: dict, ct_mesh_scene: dict | None = None
) -> tuple[dict, torch.Tensor, torch.Tensor, float]:
    """Build mechanics fields on a synthetic surrogate or supplied CT mesh scene.

    ``ct_mesh_scene`` is expected to have already passed through the
    de-identified CT mesh adapter.  Keeping path handling outside this function
    prevents source paths from leaking into patient records or manifests.
    """
    scene = (
        make_synthetic_ct_surrogate(
            spec["patient_id"], seed=spec["geometry_seed"], E_true=spec["E_background"]
        )
        if ct_mesh_scene is None
        else ct_mesh_scene
    )
    required = ("nodes", "elems", "surface_tris", "fixed", "Nn", "lx", "ly", "nu_true")
    missing = [key for key in required if key not in scene]
    if missing:
        raise ValueError(f"CT mesh scene is missing required fields: {missing}")
    fraction = torch.tensor(
        spec["inclusion_center_fraction"], dtype=scene["nodes"].dtype
    )
    extent = scene["nodes"].max(dim=0).values - scene["nodes"].min(dim=0).values
    center = scene["nodes"].min(dim=0).values + fraction * extent
    radius = spec["inclusion_radius_fraction"] * min(scene["lx"], scene["ly"])
    E_nodes = make_heterogeneous_E_field(
        scene["nodes"],
        center,
        radius,
        spec["E_background"],
        spec["E_background"] * spec["inclusion_ratio"],
    )
    return scene, E_nodes, center, radius


def force_sequence(
    scene: dict, experiment: dict, *, max_force: float
) -> tuple[torch.Tensor, list[dict]]:
    """Apply measured loads only to nodes on the imaged FEM surface."""
    nodes = scene["nodes"]
    surface_nodes = torch.unique(scene["surface_tris"].reshape(-1))
    minimum = nodes.min(dim=0).values
    extent = nodes.max(dim=0).values - minimum
    center = minimum[:2] + torch.tensor(
        experiment["center_xy"], dtype=nodes.dtype
    ) * extent[:2]
    radius = 0.16 * min(scene["lx"], scene["ly"])
    distance = torch.linalg.vector_norm(nodes[surface_nodes, :2] - center, dim=1)
    selected = surface_nodes[distance < radius]
    if len(selected) < 3:
        selected = surface_nodes[torch.argsort(distance)[:6]]
    selected_distance = torch.linalg.vector_norm(nodes[selected, :2] - center, dim=1)
    weights = torch.exp(-((selected_distance / (0.55 * radius)) ** 2))
    weights = weights / weights.sum()
    direction = torch.tensor(experiment["direction"], dtype=nodes.dtype)
    direction = direction / direction.norm()
    frames, log = [], []
    for frame, envelope in enumerate(LOAD_ENVELOPE):
        force = torch.zeros(3 * scene["Nn"], dtype=nodes.dtype)
        nodal = max_force * envelope * weights[:, None] * direction[None, :]
        force.view(-1, 3)[selected] = nodal
        frames.append(force)
        log.append(
            {
                "frame": frame,
                "commanded_force_N": max_force * envelope,
                "resultant_force_N": float(torch.linalg.vector_norm(nodal.sum(dim=0))),
                "contact_center_xy": [float(center[0]), float(center[1])],
                "contact_node_count": int(len(selected)),
                "direction": [float(value) for value in direction],
            }
        )
    return torch.stack(frames), log


def minimum_jacobian(scene: dict, displacement: torch.Tensor) -> float:
    reference = scene["nodes"][scene["elems"]]
    deformed = (scene["nodes"] + displacement.view(scene["Nn"], 3))[scene["elems"]]
    return float(torch.det(deformation_gradient(deformed, reference)).min())


def generate_experiment(
    out_root: Path,
    patient: dict,
    scene: dict,
    E_nodes: torch.Tensor,
    inclusion_center: torch.Tensor,
    inclusion_radius: float,
    experiment: dict,
    *,
    experiment_index: int,
    resolution: int,
    motion_noise_std: float,
    schema_version: str = "sim_lung_v2",
    save_images: bool = True,
    multiview: bool = False,
    force_prior_fraction: float = 0.05,
) -> dict:
    max_force = 8.0 + 1.5 * experiment_index + 0.4 * (
        int(patient["patient_id"].rsplit("_", 1)[1]) % 3
    )
    forces, force_log = force_sequence(scene, experiment, max_force=max_force)
    center = scene["nodes"].mean(dim=0)
    extent = max(scene["lx"], scene["ly"])
    camera_kwargs = {
        "T": len(LOAD_ENVELOPE),
        "radius": 1.55 * extent,
        "height": float(center[2]) + extent,
        "look_at": tuple(float(value) for value in center),
    }
    if multiview:
        poses_multiview = make_multiview_camera_poses(
            **camera_kwargs, num_views=NUM_MULTIVIEW_CAMERAS
        )
        poses = poses_multiview[:, 1]
    else:
        poses = make_camera_poses(**camera_kwargs)
        poses_multiview = None
    gaussians = seed_surface_gaussians(scene, gaussians_per_tri=3)
    set_albedo(gaussians, (0.82, 0.42, 0.39))
    gaussians["roughness"] = torch.full_like(gaussians["roughness"], 0.45)
    displacements, images, images_multiview, min_j = [], [], [], float("inf")
    for force, pose in zip(forces, poses):
        displacement = solve_nh_heterogeneous(
            scene["nodes"],
            scene["elems"],
            E_nodes,
            scene["nu_true"],
            force,
            scene["fixed"],
        )
        min_j = min(min_j, minimum_jacobian(scene, displacement))
        if min_j <= 0.05:
            raise RuntimeError(
                f"Unstable {patient['patient_id']} {experiment['name']}: minJ={min_j:.3f}"
            )
        if save_images:
            legacy_image = render(
                gaussians,
                scene,
                displacement,
                pose,
                H=resolution,
                W=resolution,
                light_intensity=2.0,
            ).clamp(0, 1)
            images.append(legacy_image)
            if multiview:
                frame = len(displacements)
                images_multiview.append(
                    torch.stack(
                        [
                            render(
                                gaussians,
                                scene,
                                displacement,
                                poses_multiview[frame, view],
                                H=resolution,
                                W=resolution,
                                light_intensity=2.0,
                            ).clamp(0, 1)
                            for view in range(NUM_MULTIVIEW_CAMERAS)
                        ]
                    )
                )
        displacements.append(displacement)
    displacement_tensor = torch.stack(displacements)
    surface_nodes = torch.unique(scene["surface_tris"].reshape(-1))
    surface_motion = displacement_tensor.view(len(LOAD_ENVELOPE), -1, 3)[:, surface_nodes]
    rest_uv_rows, deformed_uv_rows, visibility_rows = [], [], []
    gaussian_rest_rows, gaussian_deformed_rows, gaussian_visibility_rows = [], [], []
    zero_displacement = torch.zeros(3 * scene["Nn"], dtype=scene["nodes"].dtype)
    for displacement, pose in zip(displacement_tensor, poses):
        rest_uv, _, rest_valid = project(
            scene["nodes"][surface_nodes],
            pose,
            focal=200,
            H=resolution,
            W=resolution,
        )
        deformed_uv, _, deformed_valid = project(
            (scene["nodes"] + displacement.view(-1, 3))[surface_nodes],
            pose,
            focal=200,
            H=resolution,
            W=resolution,
        )
        in_frame = (
            (rest_uv[:, 0] >= 0)
            & (rest_uv[:, 0] < resolution)
            & (rest_uv[:, 1] >= 0)
            & (rest_uv[:, 1] < resolution)
            & (deformed_uv[:, 0] >= 0)
            & (deformed_uv[:, 0] < resolution)
            & (deformed_uv[:, 1] >= 0)
            & (deformed_uv[:, 1] < resolution)
        )
        rest_uv_rows.append(rest_uv)
        deformed_uv_rows.append(deformed_uv)
        visibility_rows.append(rest_valid & deformed_valid & in_frame)
        gaussian_rest = gaussian_centers(gaussians, scene, zero_displacement)
        gaussian_deformed = gaussian_centers(gaussians, scene, displacement)
        gaussian_rest_uv, _, gaussian_rest_valid = project(
            gaussian_rest, pose, focal=200, H=resolution, W=resolution
        )
        gaussian_deformed_uv, _, gaussian_deformed_valid = project(
            gaussian_deformed, pose, focal=200, H=resolution, W=resolution
        )
        gaussian_in_frame = (
            (gaussian_rest_uv[:, 0] >= 0)
            & (gaussian_rest_uv[:, 0] < resolution)
            & (gaussian_rest_uv[:, 1] >= 0)
            & (gaussian_rest_uv[:, 1] < resolution)
            & (gaussian_deformed_uv[:, 0] >= 0)
            & (gaussian_deformed_uv[:, 0] < resolution)
            & (gaussian_deformed_uv[:, 1] >= 0)
            & (gaussian_deformed_uv[:, 1] < resolution)
        )
        gaussian_rest_rows.append(gaussian_rest_uv)
        gaussian_deformed_rows.append(gaussian_deformed_uv)
        gaussian_visibility_rows.append(
            gaussian_rest_valid & gaussian_deformed_valid & gaussian_in_frame
        )
    motion_noise_seed = patient["geometry_seed"] + 1000 + experiment_index
    generator = torch.Generator().manual_seed(motion_noise_seed)
    observed_motion = surface_motion + motion_noise_std * torch.randn(
        surface_motion.shape, generator=generator, dtype=surface_motion.dtype
    )
    force_measurement_seed = (
        patient["geometry_seed"] + 70_000 + 101 * experiment_index
    )
    force_fields = measured_force_fields(
        forces,
        true_force_scale_N=max_force,
        seed=force_measurement_seed,
        prior_fraction=force_prior_fraction,
    )
    multiview_fields = {}
    if multiview:
        deformed_surface_points = (
            scene["nodes"][None]
            + displacement_tensor.view(len(LOAD_ENVELOPE), scene["Nn"], 3)
        )[:, surface_nodes]
        multiview_fields.update(
            multiview_projection_fields(
                scene["nodes"][surface_nodes],
                deformed_surface_points,
                poses_multiview,
                focal=200.0,
                resolution=resolution,
            )
        )
        gaussian_rest = gaussian_centers(gaussians, scene, zero_displacement)
        gaussian_deformed = torch.stack(
            [
                gaussian_centers(gaussians, scene, displacement)
                for displacement in displacement_tensor
            ]
        )
        multiview_fields.update(
            multiview_projection_fields(
                gaussian_rest,
                gaussian_deformed,
                poses_multiview,
                focal=200.0,
                resolution=resolution,
                prefix="image_gaussian",
            )
        )
    experiment_dir = out_root / patient["patient_id"] / experiment["name"]
    experiment_dir.mkdir(parents=True, exist_ok=True)
    if save_images:
        (experiment_dir / "images").mkdir(parents=True, exist_ok=True)
        image_uint8 = (
            (torch.stack(images) * 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        for frame, image in enumerate(image_uint8):
            Image.fromarray(image).save(
                experiment_dir / "images" / f"frame_{frame:02d}.png"
            )
        if multiview:
            multiview_uint8 = (
                (torch.stack(images_multiview) * 255)
                .to(torch.uint8)
                .permute(0, 1, 3, 4, 2)
                .cpu()
                .numpy()
            )
            for view in range(NUM_MULTIVIEW_CAMERAS):
                view_dir = experiment_dir / "images_multiview" / f"view_{view:02d}"
                view_dir.mkdir(parents=True, exist_ok=True)
                for frame, image in enumerate(multiview_uint8[:, view]):
                    Image.fromarray(image).save(view_dir / f"frame_{frame:02d}.png")
    intrinsics_matrix = torch.tensor(
        [
            [200.0, 0.0, resolution / 2.0],
            [0.0, 200.0, resolution / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=scene["nodes"].dtype,
    )
    torch.save(
        {
            "schema_version": schema_version,
            "patient_id": patient["patient_id"],
            "split": patient["split"],
            "experiment": experiment["name"],
            "nodes": scene["nodes"],
            "elems": scene["elems"],
            "surface_tris": scene["surface_tris"],
            "surface_node_ids": surface_nodes,
            "fixed": scene["fixed"],
            "forces": forces,
            "poses": poses,
            **force_fields,
            "u_seq": displacement_tensor,
            "surface_motion_true": surface_motion,
            "surface_motion_observed": observed_motion,
            "image_uv_rest_seq": torch.stack(rest_uv_rows),
            "image_uv_deformed_seq": torch.stack(deformed_uv_rows),
            "image_visibility_seq": torch.stack(visibility_rows),
            "image_gaussian_uv_rest_seq": torch.stack(gaussian_rest_rows),
            "image_gaussian_uv_deformed_seq": torch.stack(gaussian_deformed_rows),
            "image_gaussian_visibility_seq": torch.stack(gaussian_visibility_rows),
            "image_gaussian_host_tri": gaussians["host_tri"],
            "image_gaussian_bary": gaussians["bary"],
            "render_intrinsics": {
                "focal_px": 200.0,
                "height_px": resolution,
                "width_px": resolution,
            },
            **(
                {
                    "num_views": NUM_MULTIVIEW_CAMERAS,
                    "poses_multiview": poses_multiview,
                    "camera_frame_index_multiview": torch.arange(
                        len(LOAD_ENVELOPE), dtype=torch.long
                    )[:, None].repeat(1, NUM_MULTIVIEW_CAMERAS),
                    "camera_ring_azimuth_offsets_rad": torch.tensor(
                        [-0.28, 0.0, 0.28], dtype=scene["nodes"].dtype
                    ),
                    "camera_ring_height_offsets_radius": torch.tensor(
                        [-0.12, 0.0, 0.12], dtype=scene["nodes"].dtype
                    ),
                    "intrinsics_multiview": intrinsics_matrix.unsqueeze(0).repeat(
                        NUM_MULTIVIEW_CAMERAS, 1, 1
                    ),
                    **multiview_fields,
                }
                if multiview
                else {}
            ),
            "motion_noise_std": motion_noise_std,
            "motion_noise_seed": motion_noise_seed,
            "geometry_seed": patient["geometry_seed"],
            **{
                key: patient[key]
                for key in (
                    "geometry_id",
                    "geometry_source",
                    "material_source",
                    "mechanics_source",
                )
                if key in patient
            },
            "E_background": patient["E_background"],
            "E_inclusion": patient["E_background"] * patient["inclusion_ratio"],
            "inclusion_ratio": patient["inclusion_ratio"],
            "inclusion_center": inclusion_center,
            "inclusion_radius": inclusion_radius,
            "force_log": force_log,
            "boundary_condition": patient["boundary_condition"],
        },
        experiment_dir / "gt.pt",
    )
    return {
        "name": experiment["name"],
        "relative_path": str(
            Path(patient["patient_id"]) / experiment["name"] / "gt.pt"
        ),
        "max_force_N": max_force,
        "contact_center_fraction": experiment["center_xy"],
        "direction": experiment["direction"],
        "minimum_jacobian": min_j,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--patients", type=int, default=10)
    parser.add_argument("--train-patients", type=int)
    parser.add_argument("--val-patients", type=int)
    parser.add_argument("--resolution", type=int, default=48)
    parser.add_argument("--motion-noise-std", type=float, default=2.5e-4)
    parser.add_argument("--version")
    parser.add_argument("--multiview", action="store_true")
    parser.add_argument(
        "--force-prior-fraction",
        type=float,
        default=0.05,
        help="1-sigma relative measured-force perturbation",
    )
    parser.add_argument("--patient-start", type=int, default=0)
    parser.add_argument("--patient-end", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--randomize-materials", action="store_true")
    parser.add_argument("--interleaved-split", action="store_true")
    parser.add_argument("--stage-snapshots", nargs="*", type=int, default=[])
    args = parser.parse_args()
    if args.out is None:
        args.out = DEFAULT_MULTIVIEW_OUT if args.multiview else DEFAULT_OUT
    if args.version is None:
        args.version = MULTIVIEW_SCHEMA_VERSION if args.multiview else "sim_lung_v2"
    if args.multiview and args.version == "sim_lung_v2":
        raise ValueError("Multiview output requires a distinct schema version")
    torch.set_default_dtype(torch.float64)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    existing_rows: dict[str, dict] = {}
    if manifest_path.exists():
        existing_version = json.loads(
            manifest_path.read_text(encoding="utf-8")
        ).get("version")
        if existing_version != args.version:
            raise ValueError(
                f"Refusing to overwrite schema {existing_version!r} with {args.version!r}"
            )
    if args.resume and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("version") != args.version:
            raise ValueError("Existing manifest version does not match --version")
        expected_config = {
            "resolution": args.resolution,
            "motion_noise_std": args.motion_noise_std,
            "images_saved": not args.no_images,
            "train_patients": args.train_patients,
            "val_patients": args.val_patients,
            "randomize_materials": args.randomize_materials,
            "interleaved_split": args.interleaved_split,
            "multiview": args.multiview,
            "num_views": NUM_MULTIVIEW_CAMERAS if args.multiview else 1,
            "force_prior_fraction": args.force_prior_fraction,
        }
        if existing.get("generation_config") != expected_config:
            raise ValueError("Existing manifest generation_config does not match CLI")
        existing_rows = {row["patient_id"]: row for row in existing["patients"]}
    patient_end = args.patients if args.patient_end is None else args.patient_end
    if not 0 <= args.patient_start < patient_end <= args.patients:
        raise ValueError("Require 0 <= patient-start < patient-end <= patients")

    def write_manifest(rows: dict[str, dict]) -> None:
        patients = [rows[key] for key in sorted(rows)]
        manifest = {
            "version": args.version,
            "patient_count": args.patients,
            "generated_patient_count": len(patients),
            "experiment_count": sum(len(row["experiments"]) for row in patients),
            "patient_level_split": True,
            "generation_config": {
                "resolution": args.resolution,
                "motion_noise_std": args.motion_noise_std,
                "images_saved": not args.no_images,
                "train_patients": args.train_patients,
                "val_patients": args.val_patients,
                "randomize_materials": args.randomize_materials,
                "interleaved_split": args.interleaved_split,
                "multiview": args.multiview,
                "num_views": NUM_MULTIVIEW_CAMERAS if args.multiview else 1,
                "force_prior_fraction": args.force_prior_fraction,
            },
            "known_inputs": [
                "CT-conditioned geometry",
                "per-frame nodal force and contact location",
                "camera pose",
                "measured force with calibrated uncertainty",
                "boundary-condition DOFs",
            ],
            "observations": [
                "rendered image sequence" if not args.no_images else "images omitted",
                "noisy surface motion sequence",
                "image-plane surface-node tracks and visibility",
                (
                    "synchronized multiview depth and foreground confidence"
                    if args.multiview
                    else "legacy single-view protocol"
                ),
            ],
            "patients": patients,
        }
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(manifest_path)
        if len(patients) in set(args.stage_snapshots):
            (args.out / f"manifest_{len(patients)}.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )

    for patient_index in range(args.patient_start, patient_end):
        spec = patient_spec(
            patient_index,
            args.patients,
            train_count=args.train_patients,
            val_count=args.val_patients,
            randomize_materials=args.randomize_materials,
            interleaved_split=args.interleaved_split,
        )
        if args.resume and spec["patient_id"] in existing_rows:
            row = existing_rows[spec["patient_id"]]
            paths_exist = all(
                (args.out / experiment["relative_path"]).exists()
                for experiment in row["experiments"]
            )
            if paths_exist:
                print(f"{spec['patient_id']} already complete; skipping", flush=True)
                continue
        scene, E_nodes, inclusion_center, inclusion_radius = build_patient(spec)
        experiments = [
            generate_experiment(
                args.out,
                spec,
                scene,
                E_nodes,
                inclusion_center,
                inclusion_radius,
                experiment,
                experiment_index=index,
                resolution=args.resolution,
                motion_noise_std=args.motion_noise_std,
                schema_version=args.version,
                save_images=not args.no_images,
                multiview=args.multiview,
                force_prior_fraction=args.force_prior_fraction,
            )
            for index, experiment in enumerate(EXPERIMENTS)
        ]
        existing_rows[spec["patient_id"]] = {**spec, "experiments": experiments}
        write_manifest(existing_rows)
        print(
            f"{spec['patient_id']} split={spec['split']} E={spec['E_background']:.0f} "
            f"ratio={spec['inclusion_ratio']:.2f} experiments={len(experiments)}",
            flush=True,
        )
    write_manifest(existing_rows)


if __name__ == "__main__":
    main()
