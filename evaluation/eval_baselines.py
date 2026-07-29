"""Baseline comparison across scenes — the TMI/MIA comparison table.

Runs on a representative subset (for tractable runtime): one scene per E value.
For each scene runs: (1) unified (ours), (2) optical-only baseline,
(3) 4DGS-then-FEM decoupled baseline. Produces a comparison table.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from evaluation.eval_sim import reconstruct_scene_dict, eval_scene
from evaluation.baselines import baseline_optical_only, baseline_4dgs_then_fem


def main():
    torch.set_default_dtype(torch.float64)
    root = "dataset/sim_v1"
    # pick one scene per E value (the reddish-albedo press_release ones, most stable)
    scenes_by_E = {}
    for sid in range(48):
        p = os.path.join(root, f"scene_{sid:03d}", "gt.pt")
        if os.path.exists(p):
            gt = torch.load(p, weights_only=False)
            if gt["pattern"] == "press_release" and gt["albedo"] == [0.82, 0.4, 0.35] \
               and gt["E"] not in scenes_by_E:
                scenes_by_E[gt["E"]] = sid
    sids = [scenes_by_E[E] for E in sorted(scenes_by_E)]
    print(f"comparing on {len(sids)} scenes (one per E): {sids}", flush=True)

    rows = []
    for sid in sids:
        sr = os.path.join(root, f"scene_{sid:03d}")
        gt = torch.load(os.path.join(sr, "gt.pt"), weights_only=False)
        spec = gt["scene_spec"]
        scene = reconstruct_scene_dict(spec)
        forces = [f.to(torch.float64) for f in gt["forces"]]
        poses = gt["poses"].to(torch.float64)
        from rendering.gaussian_pbr import seed_surface_gaussians, set_albedo
        from inverse.joint_opt import make_gt_sequence
        torch.manual_seed(0)
        g = seed_surface_gaussians(scene, gaussians_per_tri=3)
        set_albedo(g, gt["albedo"]); g["roughness"] = torch.full_like(g["roughness"], gt["roughness"])
        I_gt, _ = make_gt_sequence(scene, g, forces, poses, H=gt["H"], W=gt["W"])
        H, W = gt["H"], gt["W"]; E_gt = gt["E"]

        # ours
        try:
            ours = eval_scene(gt, H=H, W=W, iters=50, scene_root=sr)
        except Exception as e:
            ours = {"E_rel": float("nan"), "albedo_rel": float("nan"), "PSNR": float("nan")}
            print(f"  scene {sid} ours FAILED {e}", flush=True)

        # optical-only
        try:
            r1 = baseline_optical_only(scene, {**g}, forces, poses, I_gt, iters=40, H=H, W=W)
        except Exception as e:
            r1 = {"E_rec": None, "PSNR": float("nan")}; print(f"  scene {sid} optical FAILED {e}", flush=True)

        # 4dgs-then-fem
        try:
            r2 = baseline_4dgs_then_fem(scene, {**g}, forces, poses, I_gt, iters=40, H=H, W=W, E_init=E_gt*2.5)
            e2_err = abs(r2["E_rec"] - E_gt) / E_gt * 100 if r2["E_rec"] else float("nan")
        except Exception as e:
            r2 = {"E_rec": None, "PSNR": float("nan")}; e2_err = float("nan")
            print(f"  scene {sid} 4dgs FAILED {e}", flush=True)

        row = {
            "E_gt": E_gt, "sid": sid,
            "ours_E_err": ours["E_rel"] * 100, "ours_alb_err": ours["albedo_rel"] * 100,
            "ours_PSNR": ours["PSNR"],
            "optical_PSNR": r1.get("PSNR", float("nan")),
            "decoupled_E_err": e2_err,
        }
        rows.append(row)
        print(f"  scene {sid:02d} E_gt={E_gt:.0e}: ours E_err={row['ours_E_err']:.0f}% PSNR={row['ours_PSNR']:.1f} | "
              f"optical PSNR={row['optical_PSNR']:.1f} (no E) | decoupled E_err={row['decoupled_E_err']:.0f}%", flush=True)
        torch.save({"rows": rows}, "results/baseline_cmp.pt")  # checkpoint

    # summary
    print("\n=== SUMMARY (mean) ===", flush=True)
    for k in ["ours_E_err", "ours_alb_err", "ours_PSNR", "optical_PSNR", "decoupled_E_err"]:
        v = [r[k] for r in rows if r[k] == r[k]]  # filter NaN
        if v:
            print(f"  {k}: {sum(v)/len(v):.2f}", flush=True)

    # markdown table
    L = ["# Baseline comparison\n\n",
         "| E_gt | Ours E err% | Ours alb err% | Ours PSNR | Optical-only PSNR (no E) | Decoupled E err% |\n",
         "|---|---|---|---|---|---|\n"]
    for r in rows:
        L.append(f"| {r['E_gt']:.0e} | {r['ours_E_err']:.0f} | {r['ours_alb_err']:.0f} | "
                 f"{r['ours_PSNR']:.1f} | {r['optical_PSNR']:.1f} | {r['decoupled_E_err']:.0f} |\n")
    with open("results/baseline_cmp.md", "w", encoding="utf-8") as f:
        f.write("".join(L))
    print("\nwrote results/baseline_cmp.md", flush=True)


if __name__ == "__main__":
    main()
