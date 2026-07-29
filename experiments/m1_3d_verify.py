"""M1->3D: tetrahedral Neo-Hookean gradient + E recovery verification.

Confirms R6 in 3D: the unified dim-agnostic FEM gives correct gradients and
recovers E for a 3D soft block under compression. This is the actual
dimensionality the endoscopy pipeline will use.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from physics import make_tet_grid
from physics.fem import solve_nh


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    # ---- 3D tet mesh: a soft cube -------------------------------
    nx = ny = nz = 4
    nodes, elems = make_tet_grid(nx, ny, nz, 1.0, 1.0, 1.0)
    Nn = nodes.shape[0]
    D = 3
    print(f"mesh: {Nn} nodes, {elems.shape[0]} tets, dim={D}")

    # BC: fix bottom face (z=0) fully
    bottom = torch.where(nodes[:, 2].abs() < 1e-9)[0]
    fixed = torch.cat([D * bottom, D * bottom + 1, D * bottom + 2])

    # Force: push top-center node down (compression)
    top_nodes_idx = torch.where(nodes[:, 2] > 1.0 - 1e-9)[0]
    top_center = top_nodes_idx[len(top_nodes_idx) // 2]
    f_ext = torch.zeros(D * Nn)
    f_ext[D * top_center + 2] = -5e2          # downward, physical range

    nu = torch.tensor(0.45)

    # ===== TEST 1: Newton convergence =====
    print("\n=== TEST 1: 3D Newton convergence ===")
    E_gt = torch.tensor(5.0e3)
    u_star = solve_nh(nodes, elems, E_gt, nu, f_ext, fixed, D=3)
    maxu = u_star.abs().max().item()
    from physics import deformation_gradient
    Xe = nodes[elems]; xe = Xe + u_star.view(Nn, D)[elems]
    F = deformation_gradient(xe, Xe); J = torch.det(F)
    print(f"  max|u|={maxu:.4f} (strain ~{maxu*100:.1f}%)  min J={J.min().item():.4f} max J={J.max().item():.4f}")

    # ===== TEST 2: gradient correctness =====
    print("\n=== TEST 2: 3D adjoint gradient vs finite-diff ===")
    def loss_E(Ev):
        u = solve_nh(nodes, elems, Ev, nu, f_ext, fixed, D=3)
        return u.abs().mean()
    Et = torch.tensor(8e3, requires_grad=True)
    L = loss_E(Et); L.backward()
    ga = Et.grad.item()
    eps = 1.0
    Lp = loss_E(torch.tensor(8e3 + eps)).item()
    Lm = loss_E(torch.tensor(8e3 - eps)).item()
    gf = (Lp - Lm) / (2 * eps)
    rel = abs(ga - gf) / max(abs(ga), abs(gf), 1e-30)
    print(f"  adjoint dL/dE={ga:.6e}  FD={gf:.6e}  reldiff={rel:.3e}  {'PASS' if rel<1e-2 else 'FAIL'}")

    # ===== TEST 3: E recovery =====
    print("\n=== TEST 3: 3D Young's modulus recovery ===")
    with torch.no_grad():
        u_gt = solve_nh(nodes, elems, E_gt, nu, f_ext, fixed, D=3)
    E = torch.tensor(7e3, requires_grad=True)
    opt = torch.optim.Adam([E], lr=1e2)
    for it in range(150):
        opt.zero_grad()
        u = solve_nh(nodes, elems, E, nu, f_ext, fixed, D=3)
        loss = ((u - u_gt) ** 2).mean()
        loss.backward()
        opt.step()
        if it % 30 == 0 or it == 149:
            err = abs(E.item() - E_gt.item()) / E_gt.item() * 100
            print(f"  iter {it:3d}  E={E.item():.3e}  loss={loss.item():.3e}  err={err:.2f}%")
    ferr = abs(E.item() - E_gt.item()) / E_gt.item() * 100
    print(f"\n  FINAL E_recovered={E.item():.3e}  err={ferr:.2f}%  "
          f"{'PASS' if ferr < 5 else 'MARGINAL' if ferr < 15 else 'FAIL'}")


if __name__ == "__main__":
    main()
