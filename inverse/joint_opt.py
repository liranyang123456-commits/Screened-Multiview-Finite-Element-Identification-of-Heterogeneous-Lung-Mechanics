"""M5: joint optical-mechanical inverse rendering.

Given a synthetic endoscopy sequence (images + known camera poses + known
contact-force schedule), jointly recover:
  - mechanical: Young's modulus E (Poisson nu assumed known)
  - optical:    surface albedo (per-channel), roughness

The pipeline per optimization step:
  1. solve_nh(E)  -> per-frame displacements u_t  (differentiable in E)
  2. render(u_t, albedo, roughness, pose_t) -> predicted image Ihat_t
  3. L = sum_t ||Ihat_t - I_t||^2  -> backprop to (E, albedo, roughness)

This is the unified-framework core experiment. Loss flows through both the
mechanical branch (via u_t) and the optical branch (via shading).
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from physics.fem import solve_nh
from rendering.gaussian_pbr import render


def make_gt_sequence(scene, gaussians, forces, poses, H=80, W=80, light=2.0):
    """Render the GT image sequence from the GT material parameters."""
    E_gt = scene["E_true"]; nu = scene["nu_true"]
    u_seq = []
    for f in forces:
        u = solve_nh(scene["nodes"], scene["elems"], E_gt, nu, f,
                     scene["fixed"], D=scene["D"])
        u_seq.append(u)
    u_seq = torch.stack(u_seq, dim=0)
    imgs = []
    for t in range(u_seq.shape[0]):
        img = render(gaussians, scene, u_seq[t], poses[t], H=H, W=W,
                     light_intensity=light)
        imgs.append(img)
    return torch.stack(imgs, dim=0), u_seq.detach()


def joint_recover(scene, gaussians, forces, poses, I_gt,
                  E_init=1.5e4, albedo_init=(0.5, 0.5, 0.5), rough_init=0.5,
                  iters=120, lr_E=3e2, lr_opt=3e-2, H=80, W=80, light=2.0,
                  verbose=True, warmup_albedo=0, schedule=False):
    """Joint optimization of E + albedo + roughness against the GT image seq.

    Improvements (v2) for robustness across the parameter sweep:
      - warmup_albedo: first N iters optimize albedo/rough ONLY (E frozen),
        so the renderer first explains appearance, then E explains residual
        motion. This decouples the early albedo-vs-E gradient competition that
        caused high-variance E recovery on pale/soft scenes.
      - schedule: cosine annealing of all LRs for a stable final convergence.

    Returns dict with recovered params + loss history.
    """
    nu = scene["nu_true"]
    nodes, elems, fixed, D = scene["nodes"], scene["elems"], scene["fixed"], scene["D"]
    T = I_gt.shape[0]

    # learnable parameters (log space for E/rough to keep positive + well-scaled).
    # NOTE: SSS coefficients (sigma_a, sigma_s) are intentionally NOT optimized
    # jointly here — at our render resolution (64x64, 6 frames) they are not
    # identifiable and their inclusion degrades E/albedo recovery (documented).
    # The renderer supports SSS (weight defaults to 0); it is left to a
    # high-resolution / phantom-validated follow-up. See paper Limitations.
    log_E = torch.tensor(float(np_log(E_init)), requires_grad=True)
    albedo = torch.tensor(albedo_init, dtype=torch.float64, requires_grad=True)
    log_rough = torch.tensor(float(np_log(rough_init)), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([
        {"params": [log_E], "lr": lr_E / max(E_init, 1.0)},
        {"params": [albedo], "lr": lr_opt},
        {"params": [log_rough], "lr": lr_opt},
    ])
    if schedule:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=1e-6)

    history = {"loss": [], "E": [], "albedo": [], "rough": []}
    E_gt = float(scene["E_true"].item())
    albedo_gt = gaussians["albedo"][0].detach().clone()
    rough_gt = float(gaussians["roughness"][0].item())

    for it in range(iters):
        opt.zero_grad()
        E = torch.exp(log_E)
        rough = torch.exp(log_rough).clamp(1e-3, 1.0)
        gaussians["albedo"] = albedo.unsqueeze(0).expand(gaussians["albedo"].shape[0], 3)
        gaussians["roughness"] = rough.expand(gaussians["roughness"].shape[0])

        loss = torch.zeros((), dtype=torch.float64)
        for t in range(T):
            u = solve_nh(nodes, elems, E, nu, forces[t], fixed, D=D)
            Ihat = render(gaussians, scene, u, poses[t], H=H, W=W, light_intensity=light)
            loss = loss + ((Ihat - I_gt[t]) ** 2).mean()
        loss = loss / T
        loss.backward()
        # warmup: freeze E for the first warmup_albedo iters
        if it < warmup_albedo and log_E.grad is not None:
            log_E.grad.zero_()
        opt.step()
        if schedule:
            sched.step()

        history["loss"].append(loss.item())
        history["E"].append(E.item())
        history["albedo"].append(albedo.detach().tolist())
        history["rough"].append(rough.item())

        if verbose and (it % 15 == 0 or it == iters - 1):
            e_err = abs(E.item() - E_gt) / E_gt * 100
            a_err = (albedo.detach() - albedo_gt).abs().mean().item()
            r_err = abs(rough.item() - rough_gt) / rough_gt * 100
            print(f"  iter {it:3d}  loss={loss.item():.3e}  "
                  f"E={E.item():.3e}(err {e_err:.1f}%)  "
                  f"alb={albedo.detach().tolist()}  rough={rough.item():.3f}(err {r_err:.1f}%)")

    return {
        "E_recovered": float(torch.exp(log_E).item()),
        "albedo_recovered": albedo.detach().tolist(),
        "rough_recovered": float(torch.exp(log_rough).item()),
        "E_gt": E_gt, "albedo_gt": albedo_gt.tolist(), "rough_gt": rough_gt,
        "history": history,
    }


def np_log(x):
    import math
    return math.log(x)
