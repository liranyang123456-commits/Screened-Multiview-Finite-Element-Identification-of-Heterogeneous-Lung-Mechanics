"""Test: Newton solver must converge for ALL E in a physical range when given a
reasonable initial guess. Currently fails (cold-start divergence at some E).

This is the failing test for the E-recovery root cause.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from physics import make_triangle_grid, deformation_gradient
from physics.fem import solve_nh, newton_solve

torch.set_default_dtype(torch.float64)


def test_solver_converges_across_E_range():
    """Solver should return physically valid (J>0) solutions for every E in a
    broad range around GT, so optimization can traverse the landscape smoothly."""
    nx = ny = 10
    nodes, elems = make_triangle_grid(nx, ny, 1.0, 1.0)
    Nn = nodes.shape[0]
    left = torch.where(nodes[:, 0].abs() < 1e-9)[0]
    fixed = torch.cat([2 * left, 2 * left + 1])
    tr = nx + ny * (nx + 1)
    f = torch.zeros(2 * Nn); f[2 * tr + 1] = -2e2
    nu = torch.tensor(0.45)

    # reference solution at E_gt=5e3
    u_ref = solve_nh(nodes, elems, torch.tensor(5e3), nu, f, fixed, D=2).detach()

    failures = []
    for Ev in [4.5e3, 4.7e3, 4.8e3, 4.9e3, 5.0e3, 5.1e3, 5.2e3, 5.5e3]:
        u = solve_nh(nodes, elems, torch.tensor(Ev), nu, f, fixed, D=2)
        Xe = nodes[elems]; xe = Xe + u.view(Nn, 2)[elems]
        J = torch.det(deformation_gradient(xe, Xe))
        maxu = u.abs().max().item()
        minj = J.min().item()
        status = "OK" if (minj > 0.1 and maxu < 2.0) else "FAIL"
        print(f"  E={Ev:.2e}  max|u|={maxu:.4f}  min_J={minj:.4f}  {status}")
        if status == "FAIL":
            failures.append(Ev)
    assert not failures, f"solver failed at E={failures} (cold-start divergence)"
    print("PASS: solver converges across full E range")


if __name__ == "__main__":
    test_solver_converges_across_E_range()
