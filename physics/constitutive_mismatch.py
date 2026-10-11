"""Finite-strain anisotropic fiber and viscous Neo-Hookean extensions.

These energies are used only for constitutive-mismatch audits. The production
isotropic solver is unchanged.
"""
from __future__ import annotations

import torch

from physics import deformation_gradient, nh_energy_density
from physics.fem import _simplex_vol, node_E_to_per_elem_mu_lam


def mismatch_energy_density(
    F: torch.Tensor,
    mu: torch.Tensor,
    lam: torch.Tensor,
    *,
    a0: torch.Tensor | None = None,
    k1: float = 0.0,
    k2: float = 3.0,
    eta: float = 0.0,
    F_prev: torch.Tensor | None = None,
    dt: float = 1.0,
) -> torch.Tensor:
    """Neo-Hookean plus optional Holzapfel fiber and finite-strain viscous potential."""
    psi = nh_energy_density(F, mu, lam, dim=3)
    if k1 > 0.0 and a0 is not None:
        fa = torch.einsum("nij,j->ni", F, a0)
        stretch = (fa * fa).sum(dim=-1) - 1.0
        active = torch.clamp(stretch, min=0.0)
        psi = psi + (float(k1) / (2.0 * float(k2))) * (torch.exp(float(k2) * active.square()) - 1.0)
    if eta > 0.0 and F_prev is not None:
        C = torch.matmul(F.transpose(-1, -2), F)
        C_prev = torch.matmul(F_prev.transpose(-1, -2), F_prev)
        delta = C - C_prev
        psi = psi + (float(eta) / (2.0 * float(dt))) * (delta * delta).sum(dim=(-2, -1))
    return psi


def _local_force(uloc, Xe, Vol, mu, lam, Ne, **energy_kwargs):
    with torch.enable_grad():
        xloc = Xe + uloc.view(Ne, 4, 3)
        F = deformation_gradient(xloc, Xe)
        psi = mismatch_energy_density(F, mu, lam, **energy_kwargs)
        energy = (Vol * psi).sum()
        return torch.autograd.grad(energy, uloc, create_graph=True)[0]


def _assemble(u, X, elems, mu, lam, **energy_kwargs):
    Nn = X.shape[0]
    Ne = elems.shape[0]
    Xe = X[elems]
    Vol = _simplex_vol(Xe, 3)
    uloc = u.view(Nn, 3)[elems].reshape(Ne, 12).detach().clone().requires_grad_(True)
    force_e = _local_force(uloc, Xe, Vol, mu, lam, Ne, **energy_kwargs)
    fint = torch.zeros(3 * Nn, dtype=X.dtype)
    Kloc = torch.zeros(Ne, 12, 12, dtype=X.dtype)
    for column in range(12):
        seed = torch.zeros(Ne, 12, dtype=X.dtype)
        seed[:, column] = 1.0
        Kloc[:, :, column] = torch.autograd.grad(force_e, uloc, grad_outputs=seed, retain_graph=True)[0]
    dofs = torch.stack([3 * elems[:, node] + axis for node in range(4) for axis in range(3)], dim=1)
    fint.index_put_((dofs.reshape(-1),), force_e.reshape(-1), accumulate=True)
    stiffness = torch.zeros(3 * Nn, 3 * Nn, dtype=X.dtype)
    row = dofs.unsqueeze(-1).expand(-1, -1, 12).reshape(-1)
    col = dofs.unsqueeze(-2).expand(-1, 12, -1).reshape(-1)
    stiffness.index_put_((row, col), Kloc.reshape(-1), accumulate=True)
    F_now = deformation_gradient(Xe + u.view(Nn, 3)[elems], Xe).detach()
    return fint, stiffness, F_now


def solve_mismatch(
    X: torch.Tensor,
    elems: torch.Tensor,
    E_nodes: torch.Tensor,
    nu: torch.Tensor,
    f_ext: torch.Tensor,
    fixed_dofs: torch.Tensor,
    *,
    law: str,
    n_load_steps: int = 6,
    max_iter: int = 12,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Load-stepped Newton solve for isotropic, fiber, or viscous finite strain."""
    if law not in {"isotropic", "fiber", "viscous"}:
        raise ValueError(law)
    mu, lam, _ = node_E_to_per_elem_mu_lam(E_nodes, elems, nu)
    mu_scale = float(mu.median())
    a0 = torch.tensor([1.0, 0.0, 0.0], dtype=X.dtype)
    energy = {
        "a0": a0 if law == "fiber" else None,
        "k1": 0.6 * mu_scale if law == "fiber" else 0.0,
        "k2": 3.0,
        "eta": 0.2 * mu_scale if law == "viscous" else 0.0,
        "F_prev": None,
        "dt": 1.0 / n_load_steps,
    }
    Nn = X.shape[0]
    u = torch.zeros(3 * Nn, dtype=X.dtype)
    free = torch.tensor([i for i in range(3 * Nn) if i not in set(fixed_dofs.tolist())], dtype=torch.long)
    for step in range(1, n_load_steps + 1):
        load = (step / n_load_steps) * f_ext
        for _ in range(max_iter):
            fint, stiffness, F_now = _assemble(u, X, elems, mu, lam, **energy)
            residual = load - fint
            residual_norm = float(residual[free].detach().norm())
            load_norm = float(load[free].detach().norm())
            if residual_norm < tol * (1.0 + load_norm):
                break
            block = stiffness[free][:, free]
            block = block + 1e-8 * torch.eye(block.shape[0], dtype=X.dtype) * (block.diag().abs().mean() + 1.0)
            delta = torch.zeros_like(u)
            delta[free] = torch.linalg.solve(block, residual[free])
            u = u + delta
        energy["F_prev"] = F_now.detach()
    return u.detach()
