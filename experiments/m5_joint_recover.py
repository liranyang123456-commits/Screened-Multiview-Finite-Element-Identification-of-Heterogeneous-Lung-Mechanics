"""M5 end-to-end experiment: joint recovery of E + albedo from image sequence.

This is the unified-framework proof-of-concept. Generates a GT endoscopy
sequence with known E/albedo, then jointly recovers them from images alone.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from simulator.scene import (build_tissue_scene, contact_force_sequence,
                             make_camera_poses)
from rendering.gaussian_pbr import seed_surface_gaussians, set_albedo
from inverse.joint_opt import make_gt_sequence, joint_recover


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    # ---- build GT scene ----
    scene = build_tissue_scene(nx=6, ny=6, nz=3, E_true=5e3, nu_true=0.45)
    gaussians = seed_surface_gaussians(scene, gaussians_per_tri=3)
    ALBEDO_GT = (0.82, 0.40, 0.35)        # reddish tissue
    ROUGH_GT = 0.45
    set_albedo(gaussians, ALBEDO_GT)
    gaussians["roughness"] = torch.full_like(gaussians["roughness"], ROUGH_GT)

    T = 6
    forces, _ = contact_force_sequence(scene, T=T, max_force=2e2)
    poses = make_camera_poses(T=T)

    print(f"scene: {scene['Nn']} nodes, {gaussians['albedo'].shape[0]} gaussians, {T} frames")
    print(f"GT: E={scene['E_true'].item():.0e}, albedo={ALBEDO_GT}, rough={ROUGH_GT}")

    H = W = 64
    print("\ngenerating GT image sequence...")
    I_gt, u_gt = make_gt_sequence(scene, gaussians, forces, poses, H=H, W=W, light=2.0)
    print(f"  GT images: {tuple(I_gt.shape)}, range [{I_gt.min():.3f},{I_gt.max():.3f}]")
    print(f"  GT displacements: max|u|={u_gt.abs().max():.4f}")

    # ---- joint recovery (4x off on E, wrong albedo/rough) ----
    print("\n=== JOINT RECOVERY (E + albedo + roughness) ===")
    result = joint_recover(scene, gaussians, forces, poses, I_gt,
                           E_init=2e4, albedo_init=(0.5, 0.5, 0.5), rough_init=0.7,
                           iters=100, lr_E=5e2, lr_opt=5e-2, H=H, W=W, light=2.0)

    print("\n=== FINAL RESULTS ===")
    print(f"  E:      GT={result['E_gt']:.3e}  recovered={result['E_recovered']:.3e}  "
          f"err={abs(result['E_recovered']-result['E_gt'])/result['E_gt']*100:.2f}%")
    print(f"  albedo: GT={[round(a,3) for a in result['albedo_gt']]}  "
          f"recovered={[round(a,3) for a in result['albedo_recovered']]}")
    print(f"  rough:  GT={result['rough_gt']:.3f}  recovered={result['rough_recovered']:.3f}  "
          f"err={abs(result['rough_recovered']-result['rough_gt'])/result['rough_gt']*100:.2f}%")

    # save results
    os.makedirs("results", exist_ok=True)
    torch.save({"result": result, "I_gt": I_gt, "u_gt": u_gt,
                "scene_params": {"E_true": scene["E_true"], "nu_true": scene["nu_true"]}},
               "results/m5_joint_recovery.pt")
    print("\n  saved to results/m5_joint_recovery.pt")


if __name__ == "__main__":
    main()
