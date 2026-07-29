"""Add synchronized multiview tracks to an existing patient-consistent dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lung_inverse_rendering.generate_sim_lung_v2 import (  # noqa: E402
    MULTIVIEW_SCHEMA_VERSION,
    NUM_MULTIVIEW_CAMERAS,
    measured_force_fields,
    multiview_projection_fields,
)
from simulator.scene import make_multiview_camera_poses  # noqa: E402


def augment_ground_truth(
    ground_truth: dict,
    *,
    resolution: int,
    force_prior_fraction: float,
) -> dict:
    nodes = ground_truth["nodes"].to(torch.float64)
    displacements = ground_truth["u_seq"].to(torch.float64)
    surface_nodes = ground_truth["surface_node_ids"].to(torch.long)
    center = nodes.mean(dim=0)
    extent_xyz = nodes.amax(dim=0) - nodes.amin(dim=0)
    extent = max(float(extent_xyz[0]), float(extent_xyz[1]))
    poses = make_multiview_camera_poses(
        T=len(displacements),
        radius=1.55 * extent,
        height=float(center[2]) + extent,
        look_at=tuple(float(value) for value in center),
        num_views=NUM_MULTIVIEW_CAMERAS,
    )
    deformed = (
        nodes[None] + displacements.view(len(displacements), len(nodes), 3)
    )[:, surface_nodes]
    fields = multiview_projection_fields(
        nodes[surface_nodes],
        deformed,
        poses,
        focal=200.0,
        resolution=resolution,
    )
    host_tri = ground_truth["image_gaussian_host_tri"].to(torch.long)
    bary = ground_truth["image_gaussian_bary"].to(torch.float64)
    triangle_nodes = ground_truth["surface_tris"].to(torch.long)[host_tri]
    gaussian_rest = (nodes[triangle_nodes] * bary[:, :, None]).sum(dim=1)
    deformed_nodes = nodes[None] + displacements.view(
        len(displacements), len(nodes), 3
    )
    gaussian_deformed = (
        deformed_nodes[:, triangle_nodes] * bary[None, :, :, None]
    ).sum(dim=2)
    fields.update(
        multiview_projection_fields(
            gaussian_rest,
            gaussian_deformed,
            poses,
            focal=200.0,
            resolution=resolution,
            prefix="image_gaussian",
        )
    )
    patient_index = int(ground_truth["patient_id"].rsplit("_", 1)[1])
    experiment_name = str(ground_truth.get("experiment", "unknown"))
    experiment_seed = sum(ord(value) for value in experiment_name)
    true_forces = ground_truth.get("true_forces", ground_truth["forces"]).to(
        torch.float64
    )
    true_resultants = torch.linalg.vector_norm(
        true_forces.view(len(true_forces), -1, 3).sum(dim=1), dim=1
    )
    force_fields = measured_force_fields(
        true_forces,
        true_force_scale_N=float(true_resultants.max()),
        seed=281_000 + 1009 * patient_index + experiment_seed,
        prior_fraction=force_prior_fraction,
    )
    intrinsics = torch.tensor(
        [[200.0, 0.0, resolution / 2], [0.0, 200.0, resolution / 2], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    return {
        **ground_truth,
        "schema_version": MULTIVIEW_SCHEMA_VERSION,
        "true_forces": true_forces,
        **force_fields,
        "num_views": NUM_MULTIVIEW_CAMERAS,
        "poses": poses[:, 1],
        "poses_multiview": poses,
        "camera_frame_index_multiview": torch.arange(
            len(displacements), dtype=torch.long
        )[:, None].repeat(1, NUM_MULTIVIEW_CAMERAS),
        "intrinsics_multiview": intrinsics.unsqueeze(0).repeat(
            NUM_MULTIVIEW_CAMERAS, 1, 1
        ),
        "render_intrinsics": {
            "focal_px": 200.0,
            "height_px": resolution,
            "width_px": resolution,
        },
        **fields,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resolution", type=int, default=40)
    parser.add_argument("--force-prior-fraction", type=float, default=0.05)
    parser.add_argument("--patient-start", type=int, default=0)
    parser.add_argument("--patient-limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest_path = (
        args.dataset if args.dataset.is_file() else args.dataset / "manifest.json"
    )
    root = manifest_path.parent
    output_root = args.out or root
    if args.out is not None:
        output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    patients = manifest["patients"][args.patient_start :]
    if args.patient_limit is not None:
        patients = patients[: args.patient_limit]
    completed = 0
    for patient in patients:
        for experiment in patient["experiments"]:
            source_path = root / experiment["relative_path"]
            output_path = output_root / experiment["relative_path"]
            if (
                args.resume
                and output_path.exists()
            ):
                existing = torch.load(output_path, map_location="cpu", weights_only=False)
                if (
                    "image_uv_rest_multiview_seq" in existing
                    and "image_occlusion_confidence_multiview_seq" in existing
                    and "image_gaussian_uv_rest_multiview_seq" in existing
                    and "camera_frame_index_multiview" in existing
                ):
                    continue
            ground_truth = torch.load(
                source_path, map_location="cpu", weights_only=False
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                augment_ground_truth(
                    ground_truth,
                    resolution=args.resolution,
                    force_prior_fraction=args.force_prior_fraction,
                ),
                output_path,
            )
            completed += 1
        print(patient["patient_id"], flush=True)
    if args.out is not None or not args.dataset.is_file():
        manifest["version"] = MULTIVIEW_SCHEMA_VERSION
        manifest["generation_config"] = {
            **manifest.get("generation_config", {}),
            "multiview": True,
            "num_views": NUM_MULTIVIEW_CAMERAS,
            "force_prior_fraction": args.force_prior_fraction,
            "multiview_resolution": args.resolution,
            "augmented_from_existing_mechanics": True,
        }
        manifest["observations"] = [
            *manifest.get("observations", []),
            "synchronized three-view seven-frame depth and occlusion confidence",
        ]
        (output_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    print(json.dumps({"augmented_experiments": completed}, indent=2))


if __name__ == "__main__":
    main()
