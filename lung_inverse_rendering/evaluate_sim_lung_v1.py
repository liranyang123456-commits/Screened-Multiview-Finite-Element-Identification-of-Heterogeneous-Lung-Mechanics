"""Evaluate controlled heterogeneous FEM inverse recovery on held-out geometries.

This baseline assumes known forces, poses, and optical properties.  It tests
the inverse mechanics core; it does not claim that these quantities can be
inferred from a single clinical video.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inverse.heterogeneous_recover import recover_inclusion
from rendering.gaussian_pbr import seed_surface_gaussians, set_albedo


DATASET = ROOT / "dataset" / "sim_lung_v1"
RESULTS = ROOT / "results" / "sim_lung_v1"


def reconstruct_scene(gt: dict) -> dict:
    nodes = gt["nodes"].to(torch.float64)
    return {
        "nodes": nodes,
        "elems": gt["elems"],
        "surface_tris": gt["surface_tris"],
        "fixed": gt["fixed"],
        "Nn": len(nodes),
        "D": 3,
        "nu_true": torch.tensor(0.45, dtype=torch.float64),
        "E_true": torch.tensor(float(gt["E_background"]), dtype=torch.float64),
        "lx": float(nodes[:, 0].max()),
        "ly": float(nodes[:, 1].max()),
        "lz": float(nodes[:, 2].max() - nodes[:, 2].min()),
    }


def relative_error(estimate: float, target: float) -> float:
    return abs(estimate - target) / target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=64)
    args = parser.parse_args()
    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    test_rows = [row for row in manifest["scenes"] if row["patient_split"] == "test"]
    if not test_rows:
        raise RuntimeError("No held-out patient/geometry test scenes found")
    RESULTS.mkdir(parents=True, exist_ok=True)
    records = []
    for row in test_rows:
        scene_dir = args.dataset / f"scene_{row['scene_id']:04d}"
        gt = torch.load(scene_dir / "gt.pt", map_location="cpu", weights_only=False)
        scene = reconstruct_scene(gt)
        gaussians = seed_surface_gaussians(scene, gaussians_per_tri=3)
        set_albedo(gaussians, row["albedo"])
        gaussians["roughness"] = torch.full_like(gaussians["roughness"], row["roughness"])
        images = torch.as_tensor(gt["images_uint8"], dtype=torch.float64).permute(0, 3, 1, 2) / 255.0
        center = scene["nodes"].mean(dim=0)
        radius = 0.18 * min(scene["lx"], scene["ly"])
        result = recover_inclusion(
            scene,
            gaussians,
            list(gt["forces"]),
            gt["poses"],
            images,
            E_bg_init=5e3,
            E_inc_init=8e3,
            inc_xy_init=(float(center[0]), float(center[1])),
            inc_r_init=radius,
            iters=args.iters,
            H=args.resolution,
            W=args.resolution,
            gt_params={"E_bg": gt["E_background"], "E_inc": gt["E_inclusion"], "r": radius},
        )
        record = {
            "scene_id": row["scene_id"],
            "geometry_id": row["geometry_id"],
            "geometry_source": row["geometry_source"],
            "E_background_error": relative_error(result["E_bg"], gt["E_background"]),
            "E_inclusion_error": relative_error(result["E_inc"], gt["E_inclusion"]),
            "inclusion_ratio_error": relative_error(
                result["E_inc"] / result["E_bg"], gt["E_inclusion"] / gt["E_background"]
            ),
            "radius_error": relative_error(result["inc_radius"], radius),
            "final_image_mse": result["history"]["loss"][-1],
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    summary = {
        "protocol": (
            "held-out patient geometry; known forces, poses, and optical properties; "
            "heterogeneous isotropic FEM; fibre directions are not recovered"
        ),
        "test_scene_count": len(records),
        "background_E_median_relative_error": float(np.median([row["E_background_error"] for row in records])),
        "inclusion_ratio_median_relative_error": float(
            np.median([row["inclusion_ratio_error"] for row in records])
        ),
        "radius_median_relative_error": float(np.median([row["radius_error"] for row in records])),
        "records": records,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
