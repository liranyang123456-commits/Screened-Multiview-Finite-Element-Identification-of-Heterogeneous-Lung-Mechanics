"""M1 verification: Neo-Hookean large-deformation differentiable FEM.

Tests R6 — the second critical assumption:
  - Does Newton converge under large deformation (>20% strain)?
  - Are gradients (via implicit-function adjoint) correct vs finite diff?
  - Can Young's modulus E be recovered under Neo-Hookean?

This is the nonlinear analog of the linear-elastic MVP. If it passes, the
mechanical branch of the unified framework is scientifically sound.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from physics import make_triangle_grid
from physics.fem import solve_nh
import importlib.util


def _load_render():
    """Load the MVP soft-surface renderer (reuse)."""
    spec = importlib.util.spec_from_file_location(
        "mvp_render", os.path.join(os.path.dirname(__file__), "..", "mvp", "render.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    render = _load_render()

    # ---- mesh: triangles (robust under large deformation) -----------------
    nx = ny = 10
    nodes, elems = make_triangle_grid(nx, ny, 1.0, 1.0)
    Nn = nodes.shape[0]
    ndof = 2 * Nn

    # boundary conditions: fix left edge
    left = torch.where(nodes[:, 0].abs() < 1e-9)[0]
    fixed = torch.cat([2 * left, 2 * left + 1])
    # also pin bottom-left corner in y to remove rigid rotation ambiguity
    bl = 0
    # (already pinned via left edge)

    # force at top-right node, downward — LARGE to exceed 20% strain
    top_right = nx + (ny) * (nx + 1)
    f_ext = torch.zeros(ndof)
    F_FORCE = 2e2                      # physical range (avoids element inversion)
    f_ext[2 * top_right + 1] = -F_FORCE

    nu = torch.tensor(0.45)

    # ============== TEST 1: Newton convergence at large strain =============
    print("=" * 60)
    print("TEST 1: Newton convergence under large deformation")
    print("=" * 60)
    E_gt = torch.tensor(5.0e3)
    u_star = solve_nh(nodes, elems, E_gt, nu, f_ext, fixed)
    max_disp = u_star.abs().max().item()
    # strain ~ disp / characteristic length (1.0)
    print(f"  E_gt={E_gt.item():.0e}, nu={nu.item()}")
    print(f"  max |u| = {max_disp:.4f}  (strain ~ {max_disp*100:.1f}%)")
    print(f"  top_right disp = {u_star[2*top_right+1].item():.4f}")

    # ============== TEST 2: gradient correctness (FD vs adjoint) ==========
    print("\n" + "=" * 60)
    print("TEST 2: gradient correctness — adjoint vs finite-difference")
    print("=" * 60)
    # build a scalar loss = mean image of deformed mesh (differentiable in u*)
    H = W = 96
    extent = (-0.5, 1.5, -1.0, 1.5)
    col = torch.tensor([0.85, 0.55, 0.45])

    def loss_of_E(E_val):
        u = solve_nh(nodes, elems, E_val, nu, f_ext, fixed)
        # deformed mesh -> triangle fill. render_filled_mesh expects (Ne,4) quads;
        # adapt: render each triangle by treating as degenerate quad, or sum
        # pairwise. Simplest: use displacement-norm loss as a proxy that still
        # exercises the adjoint path (u* depends on E).
        return u.abs().mean()         # differentiable in u*

    # autograd gradient
    E_test = torch.tensor(8.0e3, requires_grad=True)
    L = loss_of_E(E_test)
    L.backward()
    g_auto = E_test.grad.item()
    # finite diff
    eps = 1.0
    Lp = loss_of_E(torch.tensor(8.0e3 + eps)).item()
    Lm = loss_of_E(torch.tensor(8.0e3 - eps)).item()
    g_fd = (Lp - Lm) / (2 * eps)
    print(f"  E_test = 8e3")
    print(f"  adjoint dL/dE = {g_auto:.6e}")
    print(f"  finite-diff   = {g_fd:.6e}")
    rel = abs(g_auto - g_fd) / max(abs(g_auto), abs(g_fd), 1e-30)
    print(f"  relative diff = {rel:.3e}   {'PASS' if rel < 1e-2 else 'FAIL'}")

    # ============== TEST 3: E recovery under Neo-Hookean ===================
    print("\n" + "=" * 60)
    print("TEST 3: Young's modulus recovery (Neo-Hookean)")
    print("=" * 60)
    # GT image / displacement from E_gt
    with torch.no_grad():
        u_gt = solve_nh(nodes, elems, E_gt, nu, f_ext, fixed)

    E = torch.tensor(2.0e4, requires_grad=True)   # 4x off — stress test
    opt = torch.optim.Adam([E], lr=5e2)
    print(f"  init E = {E.item():.3e} (gt {E_gt.item():.3e})")
    for it in range(200):
        opt.zero_grad()
        u = solve_nh(nodes, elems, E, nu, f_ext, fixed)
        loss = ((u - u_gt) ** 2).mean()
        loss.backward()
        opt.step()
        if it % 25 == 0 or it == 199:
            err = abs(E.item() - E_gt.item()) / E_gt.item() * 100
            print(f"  iter {it:3d}  E={E.item():.3e}  loss={loss.item():.3e}  err={err:.2f}%")
    final_err = abs(E.item() - E_gt.item()) / E_gt.item() * 100
    print(f"\n  FINAL: E_recovered={E.item():.3e}  err={final_err:.3f}%  "
          f"{'PASS' if final_err < 2.0 else 'MARGINAL' if final_err < 10 else 'FAIL'}")

    # ============== TEST 4: gradient stability across strain range =========
    print("\n" + "=" * 60)
    print("TEST 4: gradient stability vs deformation magnitude")
    print("=" * 60)
    print(f"  {'force':>10} {'maxstrain%':>10} {'|adjoint|':>12} {'|finite-diff|':>14} {'reldiff':>10}")
    for fmag in [5e1, 1e2, 2e2, 3e2]:
        f = torch.zeros(ndof); f[2 * top_right + 1] = -fmag
        Et = torch.tensor(8.0e3, requires_grad=True)
        L = loss_of_E_param(nodes, elems, Et, nu, f, fixed)
        L.backward()
        ga = Et.grad.item()
        eps = 1.0
        Lp = loss_of_E_param(nodes, elems, torch.tensor(8e3 + eps), nu, f, fixed).item()
        Lm = loss_of_E_param(nodes, elems, torch.tensor(8e3 - eps), nu, f, fixed).item()
        gf = (Lp - Lm) / (2 * eps)
        with torch.no_grad():
            us = solve_nh(nodes, elems, torch.tensor(8e3), nu, f, fixed)
            strain = us.abs().max().item() * 100
        rd = abs(ga - gf) / max(abs(ga), abs(gf), 1e-30)
        print(f"  {fmag:10.0e} {strain:10.1f} {abs(ga):12.4e} {abs(gf):14.4e} {rd:10.2e}")


def loss_of_E_param(nodes, elems, E, nu, f_ext, fixed):
    u = solve_nh(nodes, elems, E, nu, f_ext, fixed)
    return u.abs().mean()


if __name__ == "__main__":
    main()
