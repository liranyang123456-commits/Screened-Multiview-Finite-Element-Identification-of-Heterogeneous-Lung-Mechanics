"""Heterogeneous-material inverse problem (Upgrade 1).

Recovers a spatially-varying stiffness MAP, parameterized at low dimension as a
stiff inclusion (tumor/polyp model): recover (E_background, E_inclusion,
inclusion_center_xy, inclusion_radius) from the image sequence. This is the
clinically-meaningful inverse problem (localize a stiff lesion), and is
well-posed at 5 parameters vs. the ill-posed per-node field.

Differentiability: the inclusion params -> E_nodes -> per-elem mu/lam -> u* map
is differentiated by finite differences over the 5 params (cheap), since the
forward FEM already supports per-element heterogeneous mu/lam.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from physics.fem import make_heterogeneous_E_field, solve_nh_heterogeneous, solve_nh
from rendering.gaussian_pbr import render, set_albedo


def heterogeneous_forward(scene, gaussians, forces, poses, E_bg, E_inc,
                          inc_center_xy, inc_radius, H=64, W=64, light=2.0,
                          inc_z=None):
    """Render the full sequence for a given inclusion parameterization.
    Returns (T,3,H,W) image tensor (detached — used inside FD)."""
    nodes = scene["nodes"]
    if inc_z is None:
        inc_z = float(nodes[:, 2].mean())
    E_field = make_heterogeneous_E_field(
        nodes, [inc_center_xy[0], inc_center_xy[1], inc_z], inc_radius, E_bg, E_inc)
    imgs = []
    for t in range(len(forces)):
        u = solve_nh_heterogeneous(nodes, scene["elems"], E_field, scene["nu_true"],
                                   forces[t], scene["fixed"], D=scene["D"])
        imgs.append(render(gaussians, scene, u, poses[t], H=H, W=W, light_intensity=light).detach())
    return torch.stack(imgs, 0)


def recover_inclusion(scene, gaussians, forces, poses, I_gt,
                      E_bg_init=5e3, E_inc_init=1e4, inc_xy_init=(0.5, 0.5),
                      inc_r_init=0.2, iters=40, lr=0.05, H=64, W=64, light=2.0,
                      gt_params=None):
    """Recover inclusion parameters by Adam with finite-difference gradients
    over the 5 inclusion parameters (E_bg, E_inc, center_xy, radius).

    We use FD (not the per-node adjoint) because the inclusion is low-dimensional
    (5 params) and the per-node E-field adjoint is expensive at scale; FD over 5
    params is both correct and tractable. The per-node adjoint layer is provided
    in physics.fem for future full-field recovery.
    """
    log_Ebg = torch.tensor(torch.log(torch.tensor(float(E_bg_init))).item(), requires_grad=True)
    log_Einc = torch.tensor(torch.log(torch.tensor(float(E_inc_init))).item(), requires_grad=True)
    cxy = torch.tensor(list(inc_xy_init), dtype=torch.float64, requires_grad=True)
    log_r = torch.tensor(torch.log(torch.tensor(float(inc_r_init))).item(), requires_grad=True)
    opt = torch.optim.Adam([log_Ebg, log_Einc, cxy, log_r], lr=lr)
    history = {"loss": [], "E_bg": [], "E_inc": [], "r": []}

    def unpack():
        return (torch.exp(log_Ebg).item(), torch.exp(log_Einc).item(),
                cxy.detach().tolist(), torch.exp(log_r).item())

    def loss_at(Ebg, Einc, xy, r):
        I_pred = heterogeneous_forward(scene, gaussians, forces, poses, Ebg, Einc, xy, r,
                                       H=H, W=W, light=light)
        return ((I_pred - I_gt) ** 2).mean().item()

    for it in range(iters):
        opt.zero_grad()
        Ebg, Einc, xy, r = unpack()
        L0 = loss_at(Ebg, Einc, xy, r)
        eps_E, eps_xy, eps_r = Ebg * 0.02, 0.02, r * 0.05
        g_Ebg = (loss_at(Ebg + eps_E, Einc, xy, r) - loss_at(Ebg - eps_E, Einc, xy, r)) / (2 * eps_E)
        g_Einc = (loss_at(Ebg, Einc * (1 + 0.02), xy, r) - loss_at(Ebg, Einc * (1 - 0.02), xy, r)) / (2 * Einc * 0.02)
        g_xy0 = (loss_at(Ebg, Einc, [xy[0] + eps_xy, xy[1]], r) - loss_at(Ebg, Einc, [xy[0] - eps_xy, xy[1]], r)) / (2 * eps_xy)
        g_xy1 = (loss_at(Ebg, Einc, [xy[0], xy[1] + eps_xy], r) - loss_at(Ebg, Einc, [xy[0], xy[1] - eps_xy], r)) / (2 * eps_xy)
        g_r = (loss_at(Ebg, Einc, xy, r + eps_r) - loss_at(Ebg, Einc, xy, r - eps_r)) / (2 * eps_r)
        log_Ebg.grad = torch.tensor(Ebg * g_Ebg)
        log_Einc.grad = torch.tensor(Einc * g_Einc)
        cxy.grad = torch.tensor([g_xy0, g_xy1], dtype=torch.float64)
        log_r.grad = torch.tensor(r * g_r)
        opt.step()
        with torch.no_grad():
            cxy.clamp_(0.1, 0.9); log_r.clamp_(torch.log(torch.tensor(0.05)), torch.log(torch.tensor(0.4)))
        Ebg, Einc, xy, r = unpack()
        history["loss"].append(L0); history["E_bg"].append(Ebg)
        history["E_inc"].append(Einc); history["r"].append(r)
        if it % 10 == 0 or it == iters - 1:
            msg = (f"  iter {it:3d} loss={L0:.3e} E_bg={Ebg:.0e} E_inc={Einc:.0e} "
                   f"xy={[round(v,2) for v in xy]} r={r:.3f}")
            if gt_params:
                msg += (f" | GT: E_bg={gt_params['E_bg']:.0e} "
                        f"E_inc={gt_params['E_inc']:.0e} r={gt_params['r']:.3f}")
            print(msg, flush=True)
    Ebg, Einc, xy, r = unpack()
    return {"E_bg": Ebg, "E_inc": Einc, "inc_center": xy, "inc_radius": r,
            "history": history}
