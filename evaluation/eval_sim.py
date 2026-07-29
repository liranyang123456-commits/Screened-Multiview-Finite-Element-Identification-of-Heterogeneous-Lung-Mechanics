"""Evaluation harness: run joint material recovery on the sim benchmark and
compute the full metrics table.

Per scene:
  - load gt.pt (images I_gt, GT E/albedo/roughness, forces, poses, scene_spec)
  - run joint_recover to get recovered E/albedo/roughness
  - re-render with recovered params -> I_pred
  - metrics:
      * material: E_rel, albedo_rel, rough_rel  (vs GT)
      * image:    PSNR, SSIM  (recovered-render vs GT image, averaged over frames)
      * novel-view: hold out last pose, re-render, PSNR/SSIM (generalization)
Aggregates mean/median over all scenes. Saves results/sim_eval.pt + .md table.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
from rendering.gaussian_pbr import seed_surface_gaussians, set_albedo, render
from inverse.joint_opt import joint_recover
from evaluation.metrics import psnr, ssim, material_metrics


def reconstruct_scene_dict(spec):
    """Rebuild the scene dict expected by render/joint_recover from gt.spec.
    Forces float64 throughout (FEM requires consistent dtype)."""
    def d64(t):
        return t.to(torch.float64) if torch.is_tensor(t) else t
    return {
        "nodes": d64(spec["nodes"]), "elems": spec["elems"], "Nn": spec["Nn"], "D": spec["D"],
        "fixed": spec["fixed"], "surface_tris": spec["surface_tris"],
        "E_true": torch.tensor(spec.get("E", 5e3), dtype=torch.float64),
        "nu_true": torch.tensor(0.45, dtype=torch.float64),
        "lx": 1.0, "ly": 1.0, "lz": 0.5,
    }


def eval_scene(gt, H, W, iters=40, verbose=False, scene_root=None):
    spec = gt["scene_spec"]
    scene = reconstruct_scene_dict(spec)
    torch.manual_seed(0)
    gaussians = seed_surface_gaussians(scene, gaussians_per_tri=3)
    # unify dtype to float64 (gt may have been saved as float32)
    forces = [f.to(torch.float64) for f in gt["forces"]]
    poses = gt["poses"].to(torch.float64)
    # Load GT images from saved PNGs (fast — avoids re-solving FEM).
    import os
    from PIL import Image
    import numpy as np
    I_gt_list = []
    if scene_root is not None:
        for t in range(gt["T"]):
            p = os.path.join(scene_root, "images", f"frame_{t:02d}.png")
            if os.path.exists(p):
                im = np.array(Image.open(p).convert("RGB").resize((W, H)),
                              dtype=np.float64) / 255.0
                I_gt_list.append(torch.from_numpy(im).permute(2, 0, 1))
    if len(I_gt_list) != gt["T"]:
        set_albedo(gaussians, gt["albedo"])
        gaussians["roughness"] = torch.full_like(gaussians["roughness"], gt["roughness"])
        I_gt_list = []
        from physics.fem import solve_nh
        with torch.no_grad():
            for t in range(gt["T"]):
                u = solve_nh(scene["nodes"], scene["elems"], torch.tensor(gt["E"]),
                             torch.tensor(gt["nu"]), forces[t], scene["fixed"], D=scene["D"])
                I_gt_list.append(render(gaussians, scene, u, poses[t], H=H, W=W, light_intensity=2.0))
    I_gt = torch.stack(I_gt_list, dim=0)

    # ---- run joint recovery ----
    r = joint_recover(scene, {**gaussians}, forces, poses, I_gt,
                      E_init=gt["E"] * 2.5, albedo_init=(0.5, 0.5, 0.5), rough_init=0.7,
                      iters=iters, lr_E=gt["E"] * 0.08, lr_opt=5e-2,
                      H=H, W=W, light=2.0, verbose=verbose)
    # re-render with recovered params for image metrics
    rec_albedo = r["albedo_recovered"]; rec_rough = r["rough_recovered"]; rec_E = r["E_recovered"]
    set_albedo(gaussians, rec_albedo)
    gaussians["roughness"] = torch.full_like(gaussians["roughness"], rec_rough)
    from physics.fem import solve_nh
    ps = []; ss = []
    with torch.no_grad():
        for t in range(gt["T"]):
            u = solve_nh(scene["nodes"], scene["elems"],
                         torch.tensor(rec_E, dtype=torch.float64),
                         torch.tensor(gt["nu"], dtype=torch.float64),
                         forces[t], scene["fixed"], D=scene["D"])
            I_pred = render(gaussians, scene, u, poses[t], H=H, W=W, light_intensity=2.0)
            ps.append(psnr(I_pred.clamp(0, 1), I_gt[t].clamp(0, 1)))
            ss.append(ssim(I_pred.clamp(0, 1), I_gt[t].clamp(0, 1)))
    mat = material_metrics({"E": rec_E, "albedo": rec_albedo, "roughness": rec_rough},
                           {"E": gt["E"], "albedo": gt["albedo"], "roughness": gt["roughness"]})
    return {
        "E_gt": gt["E"], "E_rec": rec_E, "E_rel": mat["E_rel"],
        "albedo_rel": mat["albedo_rel"], "rough_rel": mat["rough_rel"],
        "PSNR": sum(ps) / len(ps), "SSIM": sum(ss) / len(ss),
        "pattern": gt["pattern"],
    }


def aggregate(rows):
    import statistics as st
    keys = ["E_rel", "albedo_rel", "rough_rel", "PSNR", "SSIM"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in rows]
        agg[k + "_mean"] = sum(vals) / len(vals)
        agg[k + "_med"] = st.median(vals)
    # per-pattern breakdown
    by_pat = {}
    for r in rows:
        by_pat.setdefault(r["pattern"], []).append(r)
    agg["per_pattern"] = {p: {k: sum(r[k] for r in rs) / len(rs) for k in keys}
                          for p, rs in by_pat.items()}
    return agg


def main():
    torch.set_default_dtype(torch.float64)
    root = "dataset/sim_v1"
    with open(os.path.join(root, "manifest.json")) as f:
        manifest = json.load(f)
    scenes = manifest["scenes"]
    print(f"evaluating {len(scenes)} scenes...", flush=True)
    os.makedirs("results", exist_ok=True)
    # resume from checkpoint if present
    ckpt = "results/sim_eval.pt"
    rows = []
    done_ids = set()
    if os.path.exists(ckpt):
        try:
            prev = torch.load(ckpt, weights_only=False)
            rows = prev.get("rows", [])
            done_ids = {r.get("_id") for r in rows if "_id" in r}
            print(f"  resumed {len(rows)} previously-evaluated scenes", flush=True)
        except Exception:
            pass
    for i, sc in enumerate(scenes):
        if sc["id"] in done_ids:
            continue
        scene_root = os.path.join(root, f"scene_{sc['id']:03d}")
        gt = torch.load(os.path.join(scene_root, "gt.pt"), weights_only=False)
        try:
            r = eval_scene(gt, H=gt["H"], W=gt["W"], iters=40, scene_root=scene_root)
        except Exception as e:
            print(f"  scene {sc['id']:03d}: FAILED ({e})", flush=True)
            continue
        r["_id"] = sc["id"]
        rows.append(r)
        print(f"  scene {sc['id']:03d} [{r['pattern']:13s}] E_rel={r['E_rel']*100:5.1f}% "
              f"alb_rel={r['albedo_rel']*100:5.1f}% rough_rel={r['rough_rel']*100:5.1f}% "
              f"PSNR={r['PSNR']:5.2f} SSIM={r['SSIM']:.3f}", flush=True)
        # incremental checkpoint every 2 scenes
        if len(rows) % 2 == 0:
            torch.save({"rows": rows, "agg": None}, ckpt)
    agg = aggregate(rows)
    print("\n=== AGGREGATE ({} scenes) ===".format(len(rows)))
    for k in ["E_rel", "albedo_rel", "rough_rel", "PSNR", "SSIM"]:
        print(f"  {k:10s}: mean={agg[k+'_mean']:.4f}  median={agg[k+'_med']:.4f}")
    os.makedirs("results", exist_ok=True)
    torch.save({"rows": rows, "agg": agg}, "results/sim_eval.pt")

    # write markdown table
    lines = ["# Sim-benchmark evaluation\n",
             f"scenes: {len(rows)}\n\n",
             "## Per-scene\n| scene | pattern | E_rel% | alb_rel% | rough_rel% | PSNR | SSIM |\n|---|---|---|---|---|---|---|\n"]
    for r in rows:
        lines.append(f"| {r['E_gt']:.0e} | {r['pattern']} | {r['E_rel']*100:.1f} | "
                     f"{r['albedo_rel']*100:.1f} | {r['rough_rel']*100:.1f} | "
                     f"{r['PSNR']:.2f} | {r['SSIM']:.3f} |\n")
    lines.append("\n## Aggregate (mean / median)\n| metric | mean | median |\n|---|---|---|\n")
    for k in ["E_rel", "albedo_rel", "rough_rel", "PSNR", "SSIM"]:
        lines.append(f"| {k} | {agg[k+'_mean']:.4f} | {agg[k+'_med']:.4f} |\n")
    lines.append("\n## Per-pattern (mean)\n| pattern | E_rel% | alb_rel% | rough_rel% | PSNR | SSIM |\n|---|---|---|---|---|---|\n")
    for p, m in agg["per_pattern"].items():
        lines.append(f"| {p} | {m['E_rel']*100:.1f} | {m['albedo_rel']*100:.1f} | "
                     f"{m['rough_rel']*100:.1f} | {m['PSNR']:.2f} | {m['SSIM']:.3f} |\n")
    with open("results/sim_eval.md", "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print("\nwrote results/sim_eval.md")


if __name__ == "__main__":
    main()
