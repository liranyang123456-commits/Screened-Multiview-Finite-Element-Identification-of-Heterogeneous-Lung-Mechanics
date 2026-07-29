"""Simulation benchmark dataset generator.

Generates a standardized set of synthetic endoscopy scenes by sweeping:
  - Young's modulus E
  - albedo (3 channels)
  - roughness
  - contact pattern (press_release / drag)
For each scene: render a multi-frame image sequence + save FULL ground truth
(E, albedo, roughness, per-frame FEM displacements, contact forces, camera
poses, surface gaussians). This is the benchmark for inverse-rendering metrics.

Saved layout:
  dataset/sim_v1/
    manifest.json        (list of scenes + their GT params)
    scene_<id>/
      gt.pt              (E, nu, albedo, roughness, u_seq, forces, poses, scene_spec)
      images/frame_NN.png (the rendered GT images, uint8, for visual inspection)
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
import numpy as np
from simulator.scene import (build_tissue_scene, contact_force_sequence,
                             make_camera_poses)
from rendering.gaussian_pbr import seed_surface_gaussians, set_albedo, render


# parameter grid
E_GRID = [3e3, 5e3, 8e3, 1.2e4]                 # soft -> stiff tissue
ALBEDO_GRID = [(0.82, 0.40, 0.35),              # reddish (liver-ish)
               (0.90, 0.70, 0.55),              # pale (fat-ish)
               (0.75, 0.30, 0.45)]              # purple-ish (lesion-ish)
ROUGH_GRID = [0.35, 0.55]
PATTERN_GRID = ["press_release", "drag"]


def generate_scene(scene_id, E, albedo, rough, pattern, out_root,
                   T=6, H=64, W=64, nx=6, ny=6, nz=3, max_force=2e2, seed=0):
    torch.manual_seed(seed)
    scene = build_tissue_scene(nx=nx, ny=ny, nz=nz, E_true=E, nu_true=0.45)
    gaussians = seed_surface_gaussians(scene, gaussians_per_tri=3)
    set_albedo(gaussians, albedo)
    gaussians["roughness"] = torch.full_like(gaussians["roughness"], rough)
    forces, contact_log = contact_force_sequence(scene, T=T, max_force=max_force,
                                                 pattern=pattern)
    poses = make_camera_poses(T=T)

    imgs = []
    u_seq = []
    for t in range(T):
        # GT displacement (no grad needed for dataset)
        with torch.no_grad():
            from physics.fem import solve_nh
            u = solve_nh(scene["nodes"], scene["elems"], scene["E_true"],
                         scene["nu_true"], forces[t], scene["fixed"], D=scene["D"])
            img = render(gaussians, scene, u, poses[t], H=H, W=W, light_intensity=2.0)
        u_seq.append(u.detach())
        imgs.append(img.detach())
    I_seq = torch.stack(imgs, dim=0)             # (T,3,H,W)
    u_seq = torch.stack(u_seq, dim=0)

    # save
    sdir = os.path.join(out_root, f"scene_{scene_id:03d}")
    os.makedirs(sdir, exist_ok=True)
    os.makedirs(os.path.join(sdir, "images"), exist_ok=True)
    gt = {
        "E": float(E), "nu": 0.45,
        "albedo": list(albedo), "roughness": float(rough),
        "pattern": pattern, "T": T, "H": H, "W": W,
        "u_seq": u_seq, "forces": torch.stack(forces),
        "poses": poses, "contact_log": contact_log,
        "scene_spec": {"nx": nx, "ny": ny, "nz": nz,
                       "nodes": scene["nodes"], "elems": scene["elems"],
                       "fixed": scene["fixed"], "surface_tris": scene["surface_tris"],
                       "Nn": scene["Nn"], "D": scene["D"]},
    }
    torch.save(gt, os.path.join(sdir, "gt.pt"))
    # save uint8 images for inspection
    I_uint8 = (I_seq.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    for t in range(T):
        from PIL import Image
        Image.fromarray(I_uint8[t]).save(os.path.join(sdir, "images", f"frame_{t:02d}.png"))
    return {"id": scene_id, "E": float(E), "albedo": list(albedo),
            "roughness": float(rough), "pattern": pattern,
            "max_disp": float(u_seq.abs().max()),
            "img_mean": float(I_seq.mean())}


def generate_dataset(out_root="dataset/sim_v1", limit=None):
    os.makedirs(out_root, exist_ok=True)
    scenes = []
    sid = 0
    for E in E_GRID:
        for albedo in ALBEDO_GRID:
            for rough in ROUGH_GRID:
                for pat in PATTERN_GRID:
                    if limit is not None and sid >= limit:
                        break
                    meta = generate_scene(sid, E, albedo, rough, pat, out_root)
                    scenes.append(meta)
                    print(f"  scene {sid:03d}: E={E:.0e} alb={albedo} rough={rough} "
                          f"pat={pat} max|u|={meta['max_disp']:.3f}", flush=True)
                    sid += 1
    with open(os.path.join(out_root, "manifest.json"), "w") as f:
        json.dump({"n_scenes": len(scenes), "scenes": scenes}, f, indent=2)
    print(f"\ngenerated {len(scenes)} scenes -> {out_root}/manifest.json")
    return scenes


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap #scenes (debug)")
    ap.add_argument("--out", default="dataset/sim_v1")
    a = ap.parse_args()
    generate_dataset(out_root=a.out, limit=a.limit)
