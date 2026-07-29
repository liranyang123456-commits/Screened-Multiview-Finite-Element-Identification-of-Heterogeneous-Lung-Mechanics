"""Baselines for TMI/MIA-grade comparison.

Baseline 1: Optical-only (PR-ENDO-style). Recovers albedo/roughness/SSS with a
FREE per-frame deformation (no mechanics, no E). Tests whether the mechanical
branch adds value beyond a pure relightable-reconstruction approach.

Baseline 2: 4DGS-then-FEM (fair decoupled). Stage 1 fits a low-dimensional
per-frame rigid+scale transform to the images (a fair proxy for 4DGS-style
deformation fields, far better-conditioned than the free-node ablation).
Stage 2 fits E to the recovered transform's magnitude via FEM-residual LS.

Both reuse the same SSS+PBR renderer. Returns material/image metrics on the
same scenes as the unified method.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rendering.gaussian_pbr import render, set_albedo
from physics.fem import solve_nh
from evaluation.metrics import psnr, ssim, material_metrics


def _free_field_stage1(scene, gaussians, forces, poses, I_gt, iters, H, W, light, E_init):
    """Shared Stage-1: free per-node deformation + albedo/rough, best-loss
    tracking. Returns (best u_free, best albedo, best rough)."""
    nu = scene["nu_true"]; nodes = scene["nodes"]; Nn = scene["Nn"]; D = scene["D"]
    T = I_gt.shape[0]
    from physics.fem import solve_nh
    with torch.no_grad():
        u_init = []
        for t in range(T):
            u = solve_nh(nodes, scene["elems"], torch.tensor(float(E_init)),
                         nu, forces[t], scene["fixed"], D=D)
            u_init.append(u.detach())
        u_init = torch.stack(u_init, dim=0)
    u_free = u_init.clone().requires_grad_(True)
    albedo = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64, requires_grad=True)
    rough = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([
        {"params": [u_free], "lr": 1e-3},     # small: field is sensitive
        {"params": [albedo], "lr": 3e-2},
        {"params": [rough], "lr": 3e-2},
    ])
    best_loss = float("inf"); best = (u_init.clone(), [0.5, 0.5, 0.5], 0.6)
    for it in range(iters):
        opt.zero_grad()
        # assign learnable albedo/rough DIRECTLY (not via set_albedo, which
        # would build a plain tensor and cut the gradient to the leaves)
        gaussians["albedo"] = albedo.unsqueeze(0).expand(gaussians["albedo"].shape[0], 3)
        gaussians["roughness"] = rough.expand(gaussians["roughness"].shape[0])
        loss = 0.0
        for t in range(T):
            Ihat = render(gaussians, scene, u_free[t], poses[t], H=H, W=W, light_intensity=light)
            loss = loss + ((Ihat - I_gt[t]) ** 2).mean()
        (loss / T).backward()
        opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best = (u_free.detach().clone(), albedo.detach().tolist(), rough.item())
    return best


def baseline_optical_only(scene, gaussians, forces, poses, I_gt,
                          iters=60, H=64, W=64, light=2.0, E_init=2e4):
    """PR-ENDO-style optical-only baseline: free per-node deformation field +
    optical material (albedo, roughness), NO mechanics (E is not a parameter,
    so it cannot be recovered). Best-loss tracking prevents the free field from
    diverging below its (stiff-FEM) initialization, giving a fair PSNR. E is
    structurally unrecoverable, which is precisely our contribution's point.
    """
    u_free, rec_alb, rec_rough = _free_field_stage1(
        scene, gaussians, forces, poses, I_gt, iters, H, W, light, E_init)
    set_albedo(gaussians, rec_alb)
    gaussians["roughness"] = torch.full_like(gaussians["roughness"], rec_rough)
    ps = []; ss_ = []
    with torch.no_grad():
        for t in range(I_gt.shape[0]):
            I_pred = render(gaussians, scene, u_free[t], poses[t], H=H, W=W, light_intensity=light)
            ps.append(psnr(I_pred.clamp(0, 1), I_gt[t].clamp(0, 1)))
            ss_.append(ssim(I_pred.clamp(0, 1), I_gt[t].clamp(0, 1)))
    return {"E_rec": None, "albedo": rec_alb, "rough": rec_rough,
            "PSNR": sum(ps) / len(ps), "SSIM": sum(ss_) / len(ss_),
            "note": "optical-only, free field, no mechanics -> E not recoverable"}


def baseline_4dgs_then_fem(scene, gaussians, forces, poses, I_gt,
                           iters=60, H=64, W=64, light=2.0, E_init=2e4):
    """Fair decoupled baseline (4DGS-style): Stage 1 fits a FREE per-node
    deformation field to the images (the true 4DGS setting, no mechanics
    constraint), best-loss tracked; Stage 2 fits E to the recovered field by
    FEM-residual least squares. The literature-standard decoupled pipeline and
    a genuinely fair comparison.
    """
    nu = scene["nu_true"]
    nodes = scene["nodes"]; Nn = scene["Nn"]; D = scene["D"]; T = I_gt.shape[0]
    from physics.fem import solve_nh
    u_recovered, rec_alb, rec_rough = _free_field_stage1(
        scene, gaussians, forces, poses, I_gt, iters, H, W, light, E_init)
    # ---- Stage 2: fit E to the recovered free displacement field ----
    # best-loss tracked, ~60 iters with a moderate LR lets E move off its init.
    log_E = torch.tensor(torch.log(torch.tensor(float(E_init))).item(), requires_grad=True)
    optE = torch.optim.Adam([log_E], lr=4e-3)
    best_E_loss = float("inf"); best_E = float(E_init)
    for it in range(60):
        optE.zero_grad()
        E = torch.exp(log_E)
        loss = 0.0
        for t in range(T):
            u = solve_nh(nodes, scene["elems"], E, nu, forces[t], scene["fixed"], D=D)
            loss = loss + ((u - u_recovered[t]) ** 2).mean()
        (loss / T).backward(); optE.step()
        if loss.item() < best_E_loss:
            best_E_loss = loss.item(); best_E = E.item()
    E = torch.tensor(best_E)
    set_albedo(gaussians, rec_alb)
    gaussians["roughness"] = torch.full_like(gaussians["roughness"], rec_rough)
    ps = []
    with torch.no_grad():
        for t in range(T):
            I_pred = render(gaussians, scene, u_recovered[t], poses[t], H=H, W=W, light_intensity=light)
            ps.append(psnr(I_pred.clamp(0, 1), I_gt[t].clamp(0, 1)))
    return {"E_rec": float(best_E),
            "albedo": rec_alb, "rough": rec_rough,
            "PSNR": sum(ps) / len(ps),
            "note": "decoupled: free per-node field (4DGS-style) then fit E"}
