"""Dimension-agnostic differentiable Neo-Hookean FEM (2D triangles / 3D tets).

Unified assembly: each simplex element has (D+1) nodes, (D+1)*D local dofs.
Internal force is computed as the autodiff gradient of the per-element strain
energy (guaranteed consistent, symmetric tangent). The Newton solve + implicit-
differentiation adjoint work identically for D=2 and D=3.

Public API:
    solve_nh(X, elems, E, nu, f_ext, fixed_dofs, dim) -> u* (differentiable in E,nu,f)
"""
from __future__ import annotations
import torch
import torch.autograd
from . import (deformation_gradient, nh_energy_density)


def _simplex_vol(X: torch.Tensor, D: int) -> torch.Tensor:
    """Rest volume/area of each simplex. X: (Ne, D+1, D). Returns (Ne,).

    Volume = |det([v1-v0, v2-v0, ..., vD-v0])| / D!
    """
    cols = torch.stack([X[:, k + 1] - X[:, 0] for k in range(D)], dim=-1)  # (Ne,D,D)
    det = torch.linalg.det(cols)
    fact = 1.0
    for i in range(2, D + 1):
        fact *= i
    return det.abs() / fact


def _shape_grads(X: torch.Tensor, D: int) -> torch.Tensor:
    """Reference shape-function gradients dN_k/dX (constant per simplex).

    For a simplex with vertices v0..vD, the gradients are the rows of
    inv([v1-v0 | v2-v0 | ... | vD-v0]) for nodes 1..D, and node0 is the
    negative sum. Returns (Ne, D+1, D).
    """
    cols = torch.stack([X[:, k + 1] - X[:, 0] for k in range(D)], dim=-1)  # (Ne,D,D)
    inv = torch.linalg.inv(cols)                                          # (Ne,D,D)
    # base shape coefficients: node0 = -sum(others), node_k = e_k for k>=1
    dN = torch.zeros(X.shape[0], D + 1, D, dtype=X.dtype, device=X.device)
    dN[:, 0] = -inv.sum(dim=-2)                       # node0 grad = -sum rows
    dN[:, 1:] = inv.transpose(-2, -1)                 # node_k grad = k-th row
    return dN


def _local_fint(uloc, Xe, dN, Vol, mu, lam, Ne, D):
    """Per-element internal force = d(Vol_e psi(F_e))/d u_e, single backward."""
    with torch.enable_grad():
        xloc = Xe + uloc.view(Ne, D + 1, D)
        F = deformation_gradient(xloc, Xe)
        psi = nh_energy_density(F, mu, lam, dim=D)
        pel = (Vol * psi).sum()
        fe = torch.autograd.grad(pel, uloc, create_graph=True)[0]   # (Ne,(D+1)*D)
    return fe


def _assemble_tangent(Xe, dN, Vol, elems, mu, lam, Nn, Ne, u_nodes, D):
    """Global tangent stiffness = d(fint)/du (symmetric Hessian), double autodiff."""
    ndof_local = (D + 1) * D
    with torch.enable_grad():
        uloc = u_nodes[elems].reshape(Ne, ndof_local).detach().clone().requires_grad_(True)
        Kloc = torch.zeros(Ne, ndof_local, ndof_local, dtype=Xe.dtype, device=Xe.device)
        fe = _local_fint(uloc, Xe, dN, Vol, mu, lam, Ne, D)
        for c in range(ndof_local):
            grad_out = torch.zeros(Ne, ndof_local, dtype=Xe.dtype, device=Xe.device)
            grad_out[:, c] = 1.0
            gi = torch.autograd.grad(fe, uloc, grad_outputs=grad_out,
                                     retain_graph=True)[0]
            Kloc[:, :, c] = gi
    K = torch.zeros(D * Nn, D * Nn, dtype=Xe.dtype, device=Xe.device)
    # local dof layout: [n0x,n0y,(n0z), n1x,... ]
    loc_dofs = torch.stack(
        [D * elems[:, n] + a for n in range(D + 1) for a in range(D)], dim=1)  # (Ne,ndof_local)
    g_i = loc_dofs.unsqueeze(-1).expand(-1, -1, ndof_local).reshape(-1)
    g_j = loc_dofs.unsqueeze(-2).expand(-1, ndof_local, -1).reshape(-1)
    K.index_put_((g_i, g_j), Kloc.reshape(-1), accumulate=True)
    return K


def assemble_internal_force_and_tangent(u_flat, X, elems, mu, lam, Nn, D):
    """Returns fint (D*Nn,), K (D*Nn, D*Nn) — energy-consistent, symmetric K."""
    Ne = elems.shape[0]
    u_nodes = u_flat.view(Nn, D)
    Xe = X[elems]
    dN = _shape_grads(Xe, D)
    Vol = _simplex_vol(Xe, D)
    ndof_local = (D + 1) * D
    uloc = u_nodes[elems].reshape(Ne, ndof_local).detach().clone().requires_grad_(True)
    fint_e = _local_fint(uloc, Xe, dN, Vol, mu, lam, Ne, D)         # (Ne,ndof_local)
    fint = torch.zeros(D * Nn, dtype=u_flat.dtype, device=u_flat.device)
    loc_dofs = torch.stack(
        [D * elems[:, n] + a for n in range(D + 1) for a in range(D)], dim=1).reshape(-1)
    fint.index_put_((loc_dofs,), fint_e.reshape(-1), accumulate=True)
    K = _assemble_tangent(Xe, dN, Vol, elems, mu, lam, Nn, Ne, u_nodes, D)
    return fint, K


def _newton_from_guess(u0, X, elems, mu, lam, f_ext, fixed_dofs, D,
                       max_iter=50, tol=1e-8):
    """Newton iterations from an initial guess u0. Returns converged u."""
    Nn = X.shape[0]
    u = u0
    all_dofs = torch.arange(D * Nn, device=X.device)
    free = all_dofs[~torch.isin(all_dofs, fixed_dofs)]
    for _ in range(max_iter):
        fint, K = assemble_internal_force_and_tangent(u, X, elems, mu, lam, Nn, D)
        R = f_ext - fint                       # balance: f_int = f_ext
        R_free = R[free]
        if R_free.norm() < tol:
            break
        Kff = K[free][:, free]
        Kff = Kff + 1e-9 * torch.eye(Kff.shape[0], dtype=X.dtype, device=X.device) * (Kff.diag().abs().mean() + 1e-12)
        du_free = torch.linalg.solve(Kff, R_free)
        du = torch.zeros_like(u)
        du[free] = du_free
        u = u + du
    return u


def newton_solve(X, elems, mu, lam, f_ext, fixed_dofs, D,
                 max_iter=50, tol=1e-8, n_load_steps=8, u_init=None):
    """Solve NH equilibrium with load stepping + warm start for robustness.

    The full external load is applied in `n_load_steps` increments; each
    increment's solve is warm-started from the previous increment's solution.
    This avoids cold-start Newton divergence for strongly nonlinear regimes
    (the root cause of spurious 'bifurcation' behavior in E-recovery). If
    u_init is given, it seeds the first increment.
    """
    Nn = X.shape[0]
    u = u_init.clone() if u_init is not None else torch.zeros(D * Nn, dtype=X.dtype, device=X.device)
    for s in range(1, n_load_steps + 1):
        f_s = (s / n_load_steps) * f_ext
        u = _newton_from_guess(u, X, elems, mu, lam, f_s, fixed_dofs, D,
                               max_iter=max_iter, tol=tol)
    # final polish at full load
    u = _newton_from_guess(u, X, elems, mu, lam, f_ext, fixed_dofs, D,
                           max_iter=max_iter, tol=tol)
    return u


class NHFEMLayer(torch.autograd.Function):
    """Differentiable NH equilibrium solve, dim-agnostic.

    Conventions: R(u;theta) = f_int(u;theta) - f_ext = 0.
      dR/du = K (tangent), dR/dtheta = d(f_int)/dtheta.
      du*/dtheta = -K^{-1} d(f_int)/dtheta.
      Adjoint lambda = K^{-T} (dL/du);  dL/dtheta = -lambda^T d(f_int)/dtheta.
    """

    @staticmethod
    def forward(ctx, X, elems, E, nu, f_ext, fixed_dofs, D):
        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        u_star = newton_solve(X, elems, mu, lam, f_ext, fixed_dofs, int(D))
        ctx.D = int(D)
        ctx.save_for_backward(X, elems, E, nu, f_ext, fixed_dofs, u_star)
        return u_star

    @staticmethod
    def backward(ctx, grad_output):
        X, elems, E, nu, f_ext, fixed_dofs, u_star = ctx.saved_tensors
        D = ctx.D
        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        Nn = X.shape[0]
        all_dofs = torch.arange(D * Nn, device=X.device)
        free = all_dofs[~torch.isin(all_dofs, fixed_dofs)]
        _, K = assemble_internal_force_and_tangent(u_star, X, elems, mu, lam, Nn, D)
        Kff = K[free][:, free]
        Kff = Kff + 1e-9 * torch.eye(Kff.shape[0], dtype=X.dtype, device=X.device) * (Kff.diag().abs().mean() + 1e-12)
        dLdu = grad_output.clone().to(dtype=Kff.dtype)
        dLdu[fixed_dofs] = 0.0
        lam_vec = torch.zeros_like(dLdu)
        lam_vec[free] = torch.linalg.solve(Kff.t(), dLdu[free])
        eps = 1e-3
        def fint_at(E_v, nu_v):
            mu_v = E_v / (2 * (1 + nu_v)); lam_v = E_v * nu_v / ((1 + nu_v) * (1 - 2 * nu_v))
            f, _ = assemble_internal_force_and_tangent(u_star, X, elems, mu_v, lam_v, Nn, D)
            return f
        dfint_dE = (fint_at(E * (1 + eps), nu) - fint_at(E * (1 - eps), nu)) / (2 * E * eps)
        dfint_dnu = (fint_at(E, nu + eps) - fint_at(E, nu - eps)) / (2 * eps)
        dL_dE = -(lam_vec * dfint_dE).sum()
        dL_dnu = -(lam_vec * dfint_dnu).sum()
        dL_df = lam_vec.to(dtype=grad_output.dtype)
        return None, None, dL_dE, dL_dnu, dL_df, None, None


def solve_nh(X, elems, E, nu, f_ext, fixed_dofs, D=2):
    """Differentiable NH equilibrium solve. u* is differentiable in (E,nu,f_ext).
    D=2 (triangles) or D=3 (tetrahedra). E is a scalar (homogeneous material)."""
    return NHFEMLayer.apply(X, elems, E, nu, f_ext, fixed_dofs, D)


# ===========================================================================
# HETEROGENEOUS-MATERIAL FEM (per-node E field -> per-element mu/lam)
# ===========================================================================

def node_E_to_per_elem_mu_lam(E_nodes, elems, nu):
    """Convert a per-node Young's-modulus field E_nodes (Nn,) to per-element
    (mu_e, lam_e) vectors (Ne,) by averaging E over each element's 4 nodes then
    applying the standard (mu,lam)<-(E,nu) relations with scalar nu.

    This is the standard FEM treatment of heterogeneous linear-elastic/Neo-
    Hookean materials: material parameters live at quadrature points (here the
    element centroid), obtained by nodal averaging. Enables recovery of a
    spatially-varying stiffness map, not just a single E.
    """
    E_per_elem = E_nodes[elems].mean(dim=1)              # (Ne,) avg of 4 node Es
    mu = E_per_elem / (2 * (1 + nu))
    lam = E_per_elem * nu / ((1 + nu) * (1 - 2 * nu))
    return mu, lam, E_per_elem


class NHFEMLayerHeterogeneous(torch.autograd.Function):
    """Differentiable NH equilibrium solve with a PER-NODE E field.

    Forward: u* = Newton solve with per-element mu/lam derived from E_nodes.
    Backward: adjoint with per-element mu/lam sensitivities, scattered back to
    nodes via dE_e/dE_nodes = 1/4 (each element averages its 4 node Es).
    Differentiable in E_nodes, nu, f_ext.
    """

    @staticmethod
    def forward(ctx, X, elems, E_nodes, nu, f_ext, fixed_dofs, D):
        mu, lam, _ = node_E_to_per_elem_mu_lam(E_nodes, elems, nu)
        u_star = newton_solve(X, elems, mu, lam, f_ext, fixed_dofs, int(D))
        ctx.D = int(D)
        ctx.save_for_backward(X, elems, E_nodes, nu, f_ext, fixed_dofs, u_star)
        return u_star

    @staticmethod
    def backward(ctx, grad_output):
        X, elems, E_nodes, nu, f_ext, fixed_dofs, u_star = ctx.saved_tensors
        D = ctx.D
        Nn = X.shape[0]; Ne = elems.shape[0]
        all_dofs = torch.arange(D * Nn, device=X.device)
        free = all_dofs[~torch.isin(all_dofs, fixed_dofs)]
        mu, lam, E_per_elem = node_E_to_per_elem_mu_lam(E_nodes, elems, nu)
        _, K = assemble_internal_force_and_tangent(u_star, X, elems, mu, lam, Nn, D)
        Kff = K[free][:, free]
        Kff = Kff + 1e-9 * torch.eye(Kff.shape[0], dtype=X.dtype, device=X.device) * (Kff.diag().abs().mean() + 1e-12)
        dLdu = grad_output.clone().to(dtype=Kff.dtype)
        dLdu[fixed_dofs] = 0.0
        lam_vec = torch.zeros_like(dLdu)
        lam_vec[free] = torch.linalg.solve(Kff.t(), dLdu[free])
        # per-element dfint/dE_e via FD on per-element E (vectorized: perturb ALL elements at once)
        eps = 1e-3
        def fint_at_elem(Ee_v):
            mu_v = Ee_v / (2 * (1 + nu)); lam_v = Ee_v * nu / ((1 + nu) * (1 - 2 * nu))
            f, _ = assemble_internal_force_and_tangent(u_star, X, elems, mu_v, lam_v, Nn, D)
            return f
        # dfint_dE_per_elem: (Ne, D*Nn) — perturb each element's E independently
        # vectorized FD: perturb the whole field by +eps on element e, for all e at once
        # via a (Ne,) diagonal perturbation broadcast — but that mixes elements.
        # Simpler+correct: per-element sensitivity = d(fint)/dE_e approximated by
        # perturbing ONLY that element. Cheap because Ne is small (~hundreds).
        dfint_dE_per_elem = torch.zeros(Ne, D * Nn, dtype=X.dtype, device=X.device)
        base_fint = fint_at_elem(E_per_elem).detach()
        for e in range(Ne):
            Ee_p = E_per_elem.clone(); Ee_p[e] *= (1 + eps)
            fp = fint_at_elem(Ee_p)
            dfint_dE_per_elem[e] = (fp - base_fint) / (E_per_elem[e] * eps + 1e-12)
        # dL/dE_e = -lambda^T dfint/dE_e  -> (Ne,)
        dL_dE_per_elem = -(lam_vec.unsqueeze(0) * dfint_dE_per_elem).sum(dim=1)
        # scatter per-element gradient to nodes: dE_e/dE_node = 1/4 (avg of 4 nodes)
        dL_dE_nodes = torch.zeros(Nn, dtype=X.dtype, device=X.device)
        node_contrib = dL_dE_per_elem / (D + 1)            # each elem contributes 1/4 to each of its 4 nodes
        for k in range(D + 1):
            dL_dE_nodes.index_add_(0, elems[:, k], node_contrib)
        # nu gradient (scalar): FD on nu with current E field
        def fint_at_nu(nu_v):
            mu_v = E_per_elem / (2 * (1 + nu_v)); lam_v = E_per_elem * nu_v / ((1 + nu_v) * (1 - 2 * nu_v))
            f, _ = assemble_internal_force_and_tangent(u_star, X, elems, mu_v, lam_v, Nn, D)
            return f
        dfint_dnu = (fint_at_nu(nu + eps) - fint_at_nu(nu - eps)) / (2 * eps)
        dL_dnu = -(lam_vec * dfint_dnu).sum()
        dL_df = lam_vec.to(dtype=grad_output.dtype)
        return None, None, dL_dE_nodes, dL_dnu, dL_df, None, None


def solve_nh_heterogeneous(X, elems, E_nodes, nu, f_ext, fixed_dofs, D=3):
    """Differentiable NH equilibrium solve with a PER-NODE E field.

    E_nodes: (Nn,) tensor of per-node Young's moduli (heterogeneous material).
    Differentiable in E_nodes, nu, f_ext. This is the heterogeneous-material
    generalization of solve_nh; it lets the inverse problem recover a stiffness
    MAP rather than a single scalar.
    """
    return NHFEMLayerHeterogeneous.apply(X, elems, E_nodes, nu, f_ext, fixed_dofs, D)


def make_heterogeneous_E_field(nodes, inclusion_center, inclusion_radius,
                               E_background, E_inclusion, soft_width=None):
    """Create a per-node E field with a stiff inclusion (e.g. a tumor/polyp):
    E_nodes = E_background outside the inclusion ball, E_inclusion inside.
    Returns (Nn,) tensor.

    If soft_width is given (or E_background/E_inclusion are tensors requiring
    grad), uses a smooth sigmoid transition of width soft_width so the field is
    differentiable in (E_background, E_inclusion, inclusion_radius). Otherwise
    uses a hard step (for GT scene generation)."""
    d = (nodes - torch.as_tensor(inclusion_center, dtype=nodes.dtype)
         if not torch.is_tensor(inclusion_center) else nodes - inclusion_center).norm(dim=1)
    needs_grad = any(torch.is_tensor(t) and t.requires_grad
                     for t in [E_background, E_inclusion, inclusion_radius])
    if soft_width is None and not needs_grad:
        # hard step (GT generation)
        inside = d < inclusion_radius
        E_nodes = torch.full((nodes.shape[0],), float(E_background), dtype=nodes.dtype)
        E_nodes = torch.where(inside, torch.tensor(float(E_inclusion), dtype=nodes.dtype), E_nodes)
        return E_nodes
    # soft transition (differentiable)
    sw = float(soft_width) if soft_width is not None else 0.05
    # inside_weight ~ 1 inside, 0 outside, smooth
    inside_weight = torch.sigmoid((inclusion_radius - d) / sw)
    E_bg_t = torch.as_tensor(float(E_background) if not torch.is_tensor(E_background)
                             else 0, dtype=nodes.dtype) if not torch.is_tensor(E_background) else E_background
    E_inc_t = E_inclusion if torch.is_tensor(E_inclusion) else torch.tensor(float(E_inclusion), dtype=nodes.dtype)
    if not torch.is_tensor(E_background):
        E_bg_t = torch.tensor(float(E_background), dtype=nodes.dtype)
    return E_bg_t * (1 - inside_weight) + E_inc_t * inside_weight
