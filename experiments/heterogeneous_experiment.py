"""Heterogeneous-material recovery experiment (Upgrade 1).

Generates a scene with a stiff inclusion (tumor model), renders GT, then
recovers the inclusion parameters (E_bg, E_inclusion, center, radius) from the
image sequence alone. Demonstrates stiffness-MAP recovery, not just scalar E.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from simulator.anatomy import build_anatomy_scene
from simulator.scene import contact_force_sequence, make_camera_poses
from rendering.gaussian_pbr import seed_surface_gaussians, set_albedo
from physics.fem import make_heterogeneous_E_field, solve_nh_heterogeneous
from inverse.heterogeneous_recover import heterogeneous_forward, recover_inclusion


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    scene = build_anatomy_scene("polyp", E_true=5e3)        # polyp-shaped base
    g = seed_surface_gaussians(scene, gaussians_per_tri=3)
    set_albedo(g, (0.82, 0.40, 0.35))
    g["roughness"] = torch.full_like(g["roughness"], 0.45)

    # GT heterogeneous material: stiff inclusion (tumor) inside the polyp
    GT = {"E_bg": 5e3, "E_inc": 2e4, "center": [0.5, 0.5], "r": 0.20}
    nodes = scene["nodes"]
    E_field_gt = make_heterogeneous_E_field(
        nodes, [GT["center"][0], GT["center"][1], float(nodes[:, 2].mean())],
        GT["r"], GT["E_bg"], GT["E_inc"])
    print(f"GT: E_bg={GT['E_bg']:.0e} E_inc={GT['E_inc']:.0e} "
          f"center={GT['center']} r={GT['r']}  ({(E_field_gt>1e4).sum()} stiff nodes)")

    forces, _ = contact_force_sequence(scene, T=6, max_force=1.5e2)
    poses = make_camera_poses(T=6)

    # render GT sequence with the heterogeneous field
    I_gt = heterogeneous_forward(scene, g, forces, poses, GT["E_bg"], GT["E_inc"],
                                 GT["center"], GT["r"], H=64, W=64)
    print(f"GT images: {tuple(I_gt.shape)}, range [{I_gt.min():.3f},{I_gt.max():.3f}]")

    # recover from wrong init
    print("\n=== HETEROGENEOUS INCLUSION RECOVERY ===")
    result = recover_inclusion(
        scene, g, forces, poses, I_gt,
        E_bg_init=6e3, E_inc_init=1.2e4, inc_xy_init=(0.42, 0.55), inc_r_init=0.15,
        iters=35, lr=0.05, H=64, W=64,
        gt_params={"E_bg": GT["E_bg"], "E_inc": GT["E_inc"], "r": GT["r"]})

    print("\n=== FINAL ===")
    print(f"  E_bg:      GT={GT['E_bg']:.0e}  rec={result['E_bg']:.0e}  "
          f"err={abs(result['E_bg']-GT['E_bg'])/GT['E_bg']*100:.0f}%")
    print(f"  E_inc:     GT={GT['E_inc']:.0e}  rec={result['E_inc']:.0e}  "
          f"err={abs(result['E_inc']-GT['E_inc'])/GT['E_inc']*100:.0f}%")
    print(f"  center:    GT={GT['center']}  rec={[round(v,3) for v in result['inc_center']]}")
    print(f"  radius:    GT={GT['r']:.3f}  rec={result['inc_radius']:.3f}  "
          f"err={abs(result['inc_radius']-GT['r'])/GT['r']*100:.0f}%")

    os.makedirs("results", exist_ok=True)
    torch.save({"gt": GT, "result": result}, "results/heterogeneous_recovery.pt")
    print("\nsaved results/heterogeneous_recovery.pt")


if __name__ == "__main__":
    main()
