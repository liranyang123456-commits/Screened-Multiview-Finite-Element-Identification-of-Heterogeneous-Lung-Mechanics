"""Dataset v2 generator: 128x128, 12 frames, multi-anatomy.

Sweeps anatomy x stiffness x albedo x roughness x pattern to build a
publication-scale benchmark (target JBHI / Medical Physics). Each scene has
fully-known GT and 12 high-res frames.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
import numpy as np
from simulator.anatomy import build_anatomy_scene, ANATOMY_REGISTRY
from simulator.scene import contact_force_sequence, make_camera_poses
from rendering.gaussian_pbr import seed_surface_gaussians, set_albedo, render
from physics.fem import solve_nh


ANATOMIES = list(ANATOMY_REGISTRY.keys())
E_GRID = [3e3, 5e3, 8e3, 1.2e4]
ALBEDO_GRID = [(0.82, 0.40, 0.35), (0.90, 0.70, 0.55), (0.75, 0.30, 0.45)]
ROUGH_GRID = [0.35, 0.55]
PATTERN_GRID = ["press_release", "drag"]


def generate_v2_scene(scene_id, anatomy, E, albedo, rough, pattern, out_root,
                      T=12, H=128, W=128, max_force=1.5e2, seed=0):
    torch.manual_seed(seed)
    scene = build_anatomy_scene(anatomy, E_true=E, nu_true=0.45)
    gaussians = seed_surface_gaussians(scene, gaussians_per_tri=3)
    set_albedo(gaussians, albedo)
    gaussians["roughness"] = torch.full_like(gaussians["roughness"], rough)
    forces, contact_log = contact_force_sequence(scene, T=T, max_force=max_force,
                                                 pattern=pattern)
    poses = make_camera_poses(T=T)
    imgs, u_seq = [], []
    for t in range(T):
        with torch.no_grad():
            u = solve_nh(scene["nodes"], scene["elems"], scene["E_true"],
                         scene["nu_true"], forces[t], scene["fixed"], D=scene["D"])
            img = render(gaussians, scene, u, poses[t], H=H, W=W, light_intensity=2.0)
        u_seq.append(u.detach()); imgs.append(img.detach())
    I_seq = torch.stack(imgs, 0); u_seq = torch.stack(u_seq, 0)

    sdir = os.path.join(out_root, f"scene_{scene_id:04d}")
    os.makedirs(os.path.join(sdir, "images"), exist_ok=True)
    gt = {
        "anatomy": anatomy, "E": float(E), "nu": 0.45,
        "albedo": list(albedo), "roughness": float(rough), "pattern": pattern,
        "T": T, "H": H, "W": W,
        "u_seq": u_seq, "forces": torch.stack(forces), "poses": poses,
        "contact_log": contact_log,
        "scene_spec": {"nx": None, "anatomy": anatomy,
                       "nodes": scene["nodes"], "elems": scene["elems"],
                       "fixed": scene["fixed"], "surface_tris": scene["surface_tris"],
                       "Nn": scene["Nn"], "D": scene["D"]},
    }
    torch.save(gt, os.path.join(sdir, "gt.pt"))
    from PIL import Image
    Iu8 = (I_seq.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    for t in range(T):
        Image.fromarray(Iu8[t]).save(os.path.join(sdir, "images", f"frame_{t:02d}.png"))
    return {"id": scene_id, "anatomy": anatomy, "E": float(E), "albedo": list(albedo),
            "roughness": float(rough), "pattern": pattern,
            "max_disp": float(u_seq.abs().max())}


def generate_v2(out_root="dataset/sim_v2", limit=None, anatomies=None,
                e_per_anatomy=2, albedo_per_anatomy=1, patterns=("press_release",)):
    """Anatomy-first generation for diversity under a limit.

    For each anatomy, takes e_per_anatomy E values, albedo_per_anatomy albedoes,
    the given patterns, and both roughnesses — so the limit yields a balanced
    spread across anatomies rather than exhausting flat_block first.
    """
    os.makedirs(out_root, exist_ok=True)
    anats = anatomies or ANATOMIES
    scenes, sid = [], 0
    Es = E_GRID[:e_per_anatomy] if e_per_anatomy else E_GRID
    albs = ALBEDO_GRID[:albedo_per_anatomy] if albedo_per_anatomy else ALBEDO_GRID
    for anat in anats:
        for E in Es:
            for alb in albs:
                for rough in ROUGH_GRID:
                    for pat in patterns:
                        if limit is not None and sid >= limit:
                            break
                        m = generate_v2_scene(sid, anat, E, alb, rough, pat, out_root)
                        scenes.append(m)
                        print(f"  {sid:04d} {anat:13s} E={E:.0e} a={alb} r={rough} p={pat} "
                              f"max|u|={m['max_disp']:.3f}", flush=True)
                        sid += 1
                    if limit is not None and sid >= limit: break
                if limit is not None and sid >= limit: break
            if limit is not None and sid >= limit: break
        if limit is not None and sid >= limit: break
    json.dump({"n_scenes": len(scenes), "version": "v2", "scenes": scenes},
              open(os.path.join(out_root, "manifest.json"), "w"), indent=2)
    print(f"\ngenerated {len(scenes)} v2 scenes -> {out_root}/manifest.json", flush=True)
    return scenes


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="dataset/sim_v2")
    ap.add_argument("--anatomies", nargs="*", default=None)
    a = ap.parse_args()
    generate_v2(out_root=a.out, limit=a.limit, anatomies=a.anatomies)
