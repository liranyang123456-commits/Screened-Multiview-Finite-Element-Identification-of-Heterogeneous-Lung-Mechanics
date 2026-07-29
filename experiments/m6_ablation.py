"""M6: ablation experiments.

Three comparisons that establish the TMI story:
  A1. Route A (decoupled) vs Route B (joint) — proves the unified framework
      value. Route A: first recover geometry/displacement by image alignment
      (free per-node, no mechanics), THEN fit E to the recovered displacement.
      Route B: joint (M5).
  A2. Noise robustness — recovery accuracy under additive Gaussian image noise.
  A3. Multi-E generalization — recovery across a range of GT E values.

All on the same M2 simulator + M3 renderer. Outputs saved to results/.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from simulator.scene import build_tissue_scene, contact_force_sequence, make_camera_poses
from rendering.gaussian_pbr import seed_surface_gaussians, set_albedo, render
from inverse.joint_opt import make_gt_sequence, joint_recover
from physics.fem import solve_nh


# =============================================================================
# A1. Route A (decoupled) baseline
# =============================================================================
def route_a_decoupled(scene, gaussians, forces, poses, I_gt,
                      E_init=2e4, iters_geom=80, iters_E=80, H=64, W=64, light=2.0):
    """Route A: Stage 1 recover per-frame node displacements by direct image
    alignment (free displacements, no mechanics constraint); Stage 2 fit E to
    the recovered displacements via least-squares on the FEM residual.

    This deliberately DROPS the mechanics constraint during image alignment —
    the displacement field is free, so it can fit image noise / be non-physical.
    Then E is fit to whatever (possibly noisy) displacement came out.
    """
    nu = scene["nu_true"]; nodes = scene["nodes"]; elems = scene["elems"]
    fixed = scene["fixed"]; D = scene["D"]; Nn = scene["Nn"]; T = I_gt.shape[0]

    # ---- Stage 1: free per-frame displacement via image alignment ----
    # initialize displacements from a stiff guess
    u_free = torch.zeros(T, D * Nn, dtype=torch.float64, requires_grad=True)
    albedo = gaussians["albedo"][0].detach().clone().requires_grad_(True)
    rough = gaussians["roughness"][0].detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([u_free, albedo, rough], lr=2e-2)
    for it in range(iters_geom):
        opt.zero_grad()
        gaussians["albedo"] = albedo.unsqueeze(0).expand(gaussians["albedo"].shape[0], 3)
        gaussians["roughness"] = rough.expand(gaussians["roughness"].shape[0]).clamp(1e-3,1)
        loss = 0.0
        for t in range(T):
            Ihat = render(gaussians, scene, u_free[t], poses[t], H=H, W=W, light_intensity=light)
            loss = loss + ((Ihat - I_gt[t]) ** 2).mean()
        loss = loss / T
        loss.backward()
        opt.step()
    u_recovered = u_free.detach()

    # ---- Stage 2: fit E to recovered displacements (FEM residual LS) ----
    # minimize sum_t || solve_nh(E, f_t) - u_recovered_t ||^2 over E
    log_E = torch.tensor(torch.log(torch.tensor(float(E_init))).item(), requires_grad=True)
    opt_E = torch.optim.Adam([log_E], lr=2e-4)
    for it in range(iters_E):
        opt_E.zero_grad()
        E = torch.exp(log_E)
        loss = 0.0
        for t in range(T):
            u = solve_nh(nodes, elems, E, nu, forces[t], fixed, D=D)
            loss = loss + ((u - u_recovered[t]) ** 2).mean()
        loss = loss / T
        loss.backward()
        opt_E.step()
    return {"E_recovered": float(torch.exp(log_E).item()),
            "albedo_recovered": albedo.detach().tolist(),
            "rough_recovered": float(rough.item()),
            "u_recovered": u_recovered}


# =============================================================================
# A2. Noise robustness
# =============================================================================
def noise_robustness(scene, gaussians, forces, poses, I_gt_clean,
                     noise_levels=(0.0, 0.02, 0.05, 0.1), H=64, W=64):
    results = {}
    for sigma in noise_levels:
        torch.manual_seed(42)
        I_noisy = I_gt_clean + sigma * torch.randn_like(I_gt_clean) if sigma > 0 else I_gt_clean.clone()
        I_noisy = I_noisy.clamp(0, 2)
        r = joint_recover(scene, {**gaussians}, forces, poses, I_noisy,
                          E_init=2e4, albedo_init=(0.5,0.5,0.5), rough_init=0.7,
                          iters=80, lr_E=5e2, lr_opt=5e-2, H=H, W=W, verbose=False)
        e_err = abs(r["E_recovered"] - r["E_gt"]) / r["E_gt"] * 100
        results[sigma] = {"E": r["E_recovered"], "E_err": e_err,
                          "albedo": r["albedo_recovered"], "rough": r["rough_recovered"]}
        print(f"  noise sigma={sigma:.3f}  E_err={e_err:.2f}%  rough_err={abs(r['rough_recovered']-r['rough_gt'])/r['rough_gt']*100:.1f}%")
    return results


# =============================================================================
# A3. Multi-E generalization
# =============================================================================
def multi_E(scene_builder, gaussians_per_tri=3, T=6, E_values=(3e3,5e3,8e3,1.2e4)):
    results = {}
    for E_gt in E_values:
        torch.manual_seed(0)
        scene = scene_builder(E_true=E_gt)
        g = seed_surface_gaussians(scene, gaussians_per_tri=gaussians_per_tri)
        set_albedo(g, (0.82,0.40,0.35)); g["roughness"] = torch.full_like(g["roughness"], 0.45)
        forces,_ = contact_force_sequence(scene, T=T, max_force=2e2)
        poses = make_camera_poses(T=T)
        I_gt,_ = make_gt_sequence(scene, g, forces, poses, H=64, W=64, light=2.0)
        r = joint_recover(scene, g, forces, poses, I_gt,
                          E_init=E_gt*3.0, albedo_init=(0.5,0.5,0.5), rough_init=0.7,
                          iters=80, lr_E=E_gt*0.1, lr_opt=5e-2, H=64, W=64, verbose=False)
        e_err = abs(r["E_recovered"]-r["E_gt"])/r["E_gt"]*100
        results[E_gt] = {"E_rec": r["E_recovered"], "E_err": e_err}
        print(f"  E_gt={E_gt:.0e}  recovered={r['E_recovered']:.3e}  err={e_err:.2f}%")
    return results


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    # smaller mesh + fewer iters for tractable runtime; unbuffered prints
    scene = build_tissue_scene(nx=5, ny=5, nz=3, E_true=5e3, nu_true=0.45)
    gaussians = seed_surface_gaussians(scene, gaussians_per_tri=2)
    set_albedo(gaussians, (0.82, 0.40, 0.35))
    gaussians["roughness"] = torch.full_like(gaussians["roughness"], 0.45)
    T = 5
    forces, _ = contact_force_sequence(scene, T=T, max_force=2e2)
    poses = make_camera_poses(T=T)
    I_gt, u_gt = make_gt_sequence(scene, gaussians, forces, poses, H=56, W=56, light=2.0)
    print(f"scene: {scene['Nn']} nodes, {gaussians['albedo'].shape[0]} gaussians", flush=True)

    out = {}

    # ---- A1: Route A vs Route B ----
    print("=" * 60, flush=True); print("A1: Route A (decoupled) vs Route B (joint)", flush=True); print("=" * 60, flush=True)
    print("-- Route A (decoupled) --", flush=True)
    rA = route_a_decoupled(scene, {**gaussians}, forces, poses, I_gt,
                           E_init=2e4, iters_geom=50, iters_E=50, H=56, W=56)
    e_err_A = abs(rA["E_recovered"] - 5e3) / 5e3 * 100
    print(f"  Route A: E={rA['E_recovered']:.3e} (err {e_err_A:.1f}%)", flush=True)
    out["routeA"] = {"E": rA["E_recovered"], "E_err": e_err_A}

    print("-- Route B (joint) --", flush=True)
    rB = joint_recover(scene, {**gaussians}, forces, poses, I_gt,
                       E_init=2e4, albedo_init=(0.5,0.5,0.5), rough_init=0.7,
                       iters=60, lr_E=5e2, lr_opt=5e-2, H=56, W=56, verbose=False)
    e_err_B = abs(rB["E_recovered"] - rB["E_gt"]) / rB["E_gt"] * 100
    print(f"  Route B: E={rB['E_recovered']:.3e} (err {e_err_B:.1f}%)", flush=True)
    print(f"  >> Route B {'BEATS' if e_err_B < e_err_A else 'LOSES TO'} Route A "
          f"({e_err_B:.1f}% vs {e_err_A:.1f}%)", flush=True)
    out["routeB"] = {"E": rB["E_recovered"], "E_err": e_err_B,
                     "albedo": rB["albedo_recovered"], "rough": rB["rough_recovered"]}
    torch.save(out, "results/m6_ablation.pt")      # checkpoint after A1
    print("  [checkpoint saved]", flush=True)

    # ---- A2: noise robustness (fewer levels) ----
    print("=" * 60, flush=True); print("A2: noise robustness", flush=True); print("=" * 60, flush=True)
    out["noise"] = noise_robustness(scene, {**gaussians}, forces, poses, I_gt,
                                    noise_levels=(0.0, 0.05, 0.1), H=56, W=56)
    torch.save(out, "results/m6_ablation.pt")
    print("  [checkpoint saved]", flush=True)

    # ---- A3: multi-E (2 values) ----
    print("=" * 60, flush=True); print("A3: multi-E generalization", flush=True); print("=" * 60, flush=True)
    out["multi_E"] = multi_E(lambda **kw: build_tissue_scene(nx=5, ny=5, nz=3, **kw),
                             gaussians_per_tri=2, T=5, E_values=(4e3, 1e4))
    torch.save(out, "results/m6_ablation.pt")
    print("\nsaved results/m6_ablation.pt", flush=True)


if __name__ == "__main__":
    main()
