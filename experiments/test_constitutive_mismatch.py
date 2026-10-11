"""Consistency checks for physics/constitutive_mismatch.py.

The mismatch solver is only used to *generate* observations for the
constitutive-mismatch audit; with ``law="isotropic"`` it must reproduce the
production heterogeneous Neo-Hookean solver, and the fiber / viscous laws must
move the solution away from it.
"""
from __future__ import annotations

import unittest

import torch

from physics import make_tet_grid
from physics.constitutive_mismatch import solve_mismatch
from physics.fem import make_heterogeneous_E_field, solve_nh_heterogeneous


def _small_scene():
    torch.set_default_dtype(torch.float64)
    nodes, elems = make_tet_grid(3, 2, 2, lx=0.06, ly=0.04, lz=0.04)
    nodes = nodes.to(torch.float64)
    center = torch.tensor([0.03, 0.02, 0.02], dtype=torch.float64)
    E_nodes = make_heterogeneous_E_field(nodes, center, 0.012, 4000.0, 12000.0)
    nu = torch.tensor(0.45, dtype=torch.float64)
    back = torch.where(nodes[:, 2] <= 0.0)[0]
    fixed = torch.cat([3 * back, 3 * back + 1, 3 * back + 2])
    front = torch.where(nodes[:, 2] >= 0.04 - 1e-9)[0]
    force = torch.zeros(3 * nodes.shape[0], dtype=torch.float64)
    force[3 * front + 2] = -0.02 / len(front)
    force[3 * front + 0] = 0.01 / len(front)
    return nodes, elems, E_nodes, nu, force, fixed


class ConstitutiveMismatchTest(unittest.TestCase):
    def test_isotropic_law_matches_production_solver(self) -> None:
        nodes, elems, E_nodes, nu, force, fixed = _small_scene()
        reference = solve_nh_heterogeneous(nodes, elems, E_nodes, nu, force, fixed)
        mismatch = solve_mismatch(
            nodes, elems, E_nodes, nu, force, fixed, law="isotropic", n_load_steps=4, max_iter=12, tol=1e-10
        )
        self.assertGreater(float(reference.norm()), 0.0)
        relative = float((mismatch - reference).norm() / reference.norm())
        self.assertLess(relative, 1e-7, f"isotropic mismatch solver deviates by {relative:.3e}")

    def test_fiber_and_viscous_laws_change_the_solution(self) -> None:
        nodes, elems, E_nodes, nu, force, fixed = _small_scene()
        reference = solve_nh_heterogeneous(nodes, elems, E_nodes, nu, force, fixed)
        for law in ("fiber", "viscous"):
            solution = solve_mismatch(nodes, elems, E_nodes, nu, force, fixed, law=law, n_load_steps=4, max_iter=8)
            relative = float((solution - reference).norm() / reference.norm())
            self.assertTrue(torch.isfinite(solution).all())
            self.assertGreater(relative, 1e-4, f"{law} law should perturb the displacement (got {relative:.3e})")
            self.assertLess(relative, 1.0, f"{law} law perturbation implausibly large ({relative:.3e})")

    def test_unknown_law_rejected(self) -> None:
        nodes, elems, E_nodes, nu, force, fixed = _small_scene()
        with self.assertRaises(ValueError):
            solve_mismatch(nodes, elems, E_nodes, nu, force, fixed, law="plastic")


if __name__ == "__main__":
    unittest.main()
