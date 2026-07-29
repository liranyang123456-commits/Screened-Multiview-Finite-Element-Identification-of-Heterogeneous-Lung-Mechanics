"""Real-colon-based simulation dataset generator.

Uses C3VD reference colon meshes (real geometry) as FEM domains. For each colon
segment x material combo, builds a real-geometry FEM patch, simulates
tool-contact deformation, and renders a multi-frame endoscopy sequence with
fully-known GT (E, albedo, roughness, displacements, forces, poses).

This grounds the benchmark in REAL endoscopic geometry (the key selling point
for JBHI/Medical Physics: "validated on real colon shapes, not just procedural
blocks"), while keeping mechanics fully known via simulation.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import glob
import torch
import numpy as np
from dataset.real_colon_fem import build_real_colon_scene
from simulator.scene import contact_force_sequence, make_camera_poses
from rendering.gaussian_pbr import seed_surface_gaussians, set_albedo, render
from physics.fem import solve_nh


def _front_contact_info(scene):
    """Compute front_z threshold and contact center for a real-patch scene."""
    nodes = scene["nodes"]
    front_z = float(torch.quantile(nodes[:, 2], 0.20))
    cx = float(nodes[:, 0].mean()); cy = float(nodes[:, 1].mean())
    return front_z, (cx, cy, 0.0)


def _scene_adaptive_poses(scene, T=8, radius=None, height=None):
    """Camera poses orbiting the scene, adapted to its actual centroid/scale."""
    nodes = scene["nodes"]
    center = nodes.mean(0)
    if radius is None:
        radius = float(1.6 * (nodes[:, 0].max() - nodes[:, 0].min()))
    if height is None:
        height = float(0.9 * radius)
    return make_camera_poses(T=T, radius=radius, height=height,
                             look_at=(float(center[0]), float(center[1]), float(center[2])))


def generate_real_colon_scene(scene_id, obj_path, segment, E, albedo, rough,
                              pattern, out_root, T=8, H=128, W=128,
                              max_force=3e1, seed=0):
    """Generate one real-colon-geometry scene with full GT."""
    torch.manual_seed(seed)
    scene = build_real_colon_scene(obj_path, segment, E_true=E, seed=seed)
    gaussians = seed_surface_gaussians(scene, gaussians_per_tri=3)
    set_albedo(gaussians, albedo)
    gaussians["roughness"] = torch.full_like(gaussians["roughness"], rough)
    front_z, contact_center = _front_contact_info(scene)
    forces, contact_log = contact_force_sequence(
        scene, T=T, max_force=max_force, contact_center=contact_center,
        pattern=pattern, front_z=front_z, contact_radius=0.25)
    poses = _scene_adaptive_poses(scene, T=T)
    imgs, u_seq = [], []
    for t in range(T):
        with torch.no_grad():
            u = solve_nh(scene["nodes"], scene["elems"], scene["E_true"],
                         scene["nu_true"], forces[t], scene["fixed"], D=scene["D"])
            img = render(gaussians, scene, u, poses[t], H=H, W=W, light_intensity=2.5)
        u_seq.append(u.detach()); imgs.append(img.detach())
    I_seq = torch.stack(imgs, 0); u_seq = torch.stack(u_seq, 0)
    sdir = os.path.join(out_root, f"scene_{scene_id:04d}")
    os.makedirs(os.path.join(sdir, "images"), exist_ok=True)
    gt = {
        "anatomy": f"real_colon_{segment}", "segment": segment, "E": float(E), "nu": 0.45,
        "albedo": list(albedo), "roughness": float(rough), "pattern": pattern,
        "T": T, "H": H, "W": W, "u_seq": u_seq, "forces": torch.stack(forces),
        "poses": poses, "contact_log": contact_log,
        "scene_spec": {"anatomy": f"real_colon_{segment}",
                       "nodes": scene["nodes"], "elems": scene["elems"],
                       "fixed": scene["fixed"], "surface_tris": scene["surface_tris"],
                       "Nn": scene["Nn"], "D": scene["D"]},
    }
    torch.save(gt, os.path.join(sdir, "gt.pt"))
    from PIL import Image
    Iu8 = (I_seq.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    for t in range(T):
        Image.fromarray(Iu8[t]).save(os.path.join(sdir, "images", f"frame_{t:02d}.png"))
    return {"id": scene_id, "anatomy": f"real_colon_{segment}", "segment": segment,
            "E": float(E), "albedo": list(albedo), "roughness": float(rough),
            "pattern": pattern, "max_disp": float(u_seq.abs().max())}


def generate_real_colon_dataset(out_root="dataset/real_colon", segments=None,
                                E_values=(3e3, 5e3, 8e3, 1.2e4),
                                albedoes=((0.82, 0.40, 0.35),),
                                roughnesses=(0.45,), limit=None, max_force=3e1):
    os.makedirs(out_root, exist_ok=True)
    c3vd = "dataset/C3VD"
    segs = segments or sorted(os.path.basename(p).replace("_model.obj", "")
                              for p in glob.glob(os.path.join(c3vd, "*_model.obj")))
    scenes, sid = [], 0
    for seg in segs:
        obj = os.path.join(c3vd, f"{seg}_model.obj")
        if not os.path.exists(obj):
            continue
        for E in E_values:
            for alb in albedoes:
                for rough in roughnesses:
                    if limit is not None and sid >= limit:
                        break
                    m = generate_real_colon_scene(sid, obj, seg, E, alb, rough,
                                                  "press_release", out_root,
                                                  max_force=max_force, seed=sid)
                    scenes.append(m)
                    print(f"  {sid:04d} {seg:10s} E={E:.0e} max|u|={m['max_disp']:.3f}", flush=True)
                    sid += 1
                if limit is not None and sid >= limit: break
            if limit is not None and sid >= limit: break
        if limit is not None and sid >= limit: break
    json.dump({"n_scenes": len(scenes), "version": "real_colon", "scenes": scenes},
              open(os.path.join(out_root, "manifest.json"), "w"), indent=2)
    print(f"\ngenerated {len(scenes)} real-colon scenes -> {out_root}/manifest.json", flush=True)
    return scenes


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="dataset/real_colon")
    ap.add_argument("--max_force", type=float, default=3e1)
    a = ap.parse_args()
    generate_real_colon_dataset(out_root=a.out, limit=a.limit, max_force=a.max_force)
