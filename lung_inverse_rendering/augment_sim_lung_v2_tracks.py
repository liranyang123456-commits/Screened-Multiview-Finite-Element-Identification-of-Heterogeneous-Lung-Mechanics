"""Persist auditable image-plane node tracks in an existing sim_lung_v2 set."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rendering.gaussian_pbr import (
    gaussian_centers,
    project,
    seed_surface_gaussians,
)


DATASET = ROOT / "dataset" / "sim_lung_v2"


def track_fields(gt: dict, resolution: int, focal: float = 200.0) -> dict:
    nodes = gt["nodes"].to(torch.float64)
    surface_nodes = gt["surface_node_ids"]
    rest_rows, deformed_rows, visibility_rows = [], [], []
    scene = {
        "nodes": nodes,
        "surface_tris": gt["surface_tris"],
        "Nn": len(nodes),
        "D": 3,
    }
    gaussians = seed_surface_gaussians(scene, gaussians_per_tri=3)
    gaussian_rest_rows, gaussian_deformed_rows, gaussian_visibility_rows = [], [], []
    zero = torch.zeros(3 * len(nodes), dtype=nodes.dtype)
    for displacement, pose in zip(gt["u_seq"], gt["poses"]):
        rest_uv, _, rest_valid = project(
            nodes[surface_nodes], pose, focal=focal, H=resolution, W=resolution
        )
        deformed_uv, _, deformed_valid = project(
            (nodes + displacement.view(-1, 3))[surface_nodes],
            pose,
            focal=focal,
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
        rest_rows.append(rest_uv)
        deformed_rows.append(deformed_uv)
        visibility_rows.append(rest_valid & deformed_valid & in_frame)
        gaussian_rest = gaussian_centers(gaussians, scene, zero)
        gaussian_deformed = gaussian_centers(gaussians, scene, displacement)
        gaussian_rest_uv, _, gaussian_rest_valid = project(
            gaussian_rest, pose, focal=focal, H=resolution, W=resolution
        )
        gaussian_deformed_uv, _, gaussian_deformed_valid = project(
            gaussian_deformed, pose, focal=focal, H=resolution, W=resolution
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
    return {
        "image_uv_rest_seq": torch.stack(rest_rows),
        "image_uv_deformed_seq": torch.stack(deformed_rows),
        "image_visibility_seq": torch.stack(visibility_rows),
        "image_gaussian_uv_rest_seq": torch.stack(gaussian_rest_rows),
        "image_gaussian_uv_deformed_seq": torch.stack(gaussian_deformed_rows),
        "image_gaussian_visibility_seq": torch.stack(gaussian_visibility_rows),
        "image_gaussian_host_tri": gaussians["host_tri"],
        "image_gaussian_bary": gaussians["bary"],
        "render_intrinsics": {
            "focal_px": focal,
            "height_px": resolution,
            "width_px": resolution,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--resolution", type=int, default=40)
    args = parser.parse_args()
    manifest_path = args.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    count = 0
    for patient in manifest["patients"]:
        for experiment in patient["experiments"]:
            path = args.dataset / experiment["relative_path"]
            gt = torch.load(path, map_location="cpu", weights_only=False)
            gt.update(track_fields(gt, args.resolution))
            torch.save(gt, path)
            count += 1
    observations = manifest.setdefault("observations", [])
    label = "image-plane surface-node tracks and visibility"
    if label not in observations:
        observations.append(label)
    manifest["render_intrinsics"] = {
        "focal_px": 200.0,
        "height_px": args.resolution,
        "width_px": args.resolution,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Augmented {count} experiments")


if __name__ == "__main__":
    main()
