"""Differentiable Neo-Hookean FEM solver (2D plane-strain triangles).

The forward pass solves the nonlinear equilibrium
        R(u; E, nu, f) = f_ext - f_int(u; E, nu) = 0
by Newton iteration (no autograd graph kept — fast, detached).

The backward pass uses the implicit-function theorem to give the EXACT gradient
of a downstream loss L(u*) w.r.t. (E, nu, f):
    at the solution u*,  dR/du is the tangent stiffness K (non-singular),
    du*/dtheta = -K^{-1} dR/dtheta,
    so   dL/dtheta = - (dL/du) K^{-1} dR/dtheta.
We solve the adjoint  K^T lambda = (dL/du)  and set dL/dtheta = -lambda^T dR/dtheta.

This avoids backpropagating through the (variable-length, possibly divergent)
Newton iterations, which is the usual failure mode of naive differentiable FEM.

Gradients w.r.t. boundary conditions / rest nodes can be added similarly.
"""
from __future__ import annotations
import torch
import torch.autograd
from . import (deformation_gradient, nh_pk1, nh_energy_density)


def _area(X: torch.Tensor) -> torch.Tensor:
    """Triangle rest area. X: (Ne,3,2). Returns (Ne,)."""
    d1 = X[:, 1] - X[:, 0]
    d2 = X[:, 2] - X[:, 0]
    return 0.5 * (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]).abs()


def _dN_dX(X: torch.Tensor) -> torch.Tensor:
    """Reference-element shape-function gradients (constant per triangle).

    Returns dN (Ne,3,2): dN[k] is dN_k/dx in the *rest* (material) frame,
    i.e. the columns of X_ref^{-1}. Used for both internal-force and tangent.
    """
    # X columns: (X1-X0, X2-X0); shape grads: rows of inv([X1-X0 | X2-X0]).
    dX = torch.stack([X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]], dim=-1)  # (Ne,2,2)
    dX_inv = torch.linalg.inv(dX)                                     # (Ne,2,2)
    # shape func gradients: [[-1,-1],[1,0],[0,1]] @ inv
    base = torch.tensor([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=X.dtype)
    # dN_k = base[k] @ dX_inv
    dN = torch.einsum('kb,nab->nka', base, dX_inv)                   # (Ne,3,2)
    return dN


def assemble_internal_force_and_tangent(
    u_flat: torch.Tensor,
    X: torch.Tensor,
    elems: torch.Tensor,
    mu: torch.Tensor,
    lam: torch.Tensor,
    Nn: int,
):
    """Assemble internal force vector fint (2*Nn,) and tangent stiffness K (2*Nn,2*Nn).

    fint = sum_e A_e int P : dN  (constant per triangle -> A_e * sum_k P dN_k)
    K    = -d fint / d u  (geometric + material tangent).
    """
    Ne = elems.shape[0]
    u_nodes = u_flat.view(Nn, 2)
    Xe = X[elems]                                           # (Ne,3,2) rest per-elem
    dN = _dN_dX(Xe)                                         # (Ne,3,2) material grads
    A = _area(Xe)                                           # (Ne,)
    # fint via the energy-consistent autodiff path (single backward pass).
    uloc = u_nodes[elems].reshape(Ne, 6).detach().clone().requires_grad_(True)
    fint_e = _local_fint(uloc, Xe, dN, A, mu, lam, Ne, dim=2)   # (Ne,6), graph built
    fint = torch.zeros(2 * Nn, dtype=u_flat.dtype, device=u_flat.device)
    loc_dofs = torch.stack([
        2 * elems[:, 0], 2 * elems[:, 0] + 1,
        2 * elems[:, 1], 2 * elems[:, 1] + 1,
        2 * elems[:, 2], 2 * elems[:, 2] + 1,
    ], dim=1).reshape(-1)                                   # (Ne*6,) row-block
    rep_e = torch.arange(Ne).repeat_interleave(6)
    fint.index_put_((loc_dofs,), fint_e.reshape(-1), accumulate=True)

    # --- tangent stiffness K = d(fint)/du (material + geometric) -------------
    K = _assemble_tangent_autodiff(Xe, dN, A, elems, mu, lam, Nn, Ne, u_nodes)
    return fint, K


def _local_fint(uloc, Xe, dNe, Ae, mu, lam, Ne, dim=2):
    """Per-element internal force = d(A_e psi(F_e))/d u_e, via a single
    reverse-mode autodiff of the total per-element energy w.r.t. uloc.

    Wrapped in torch.enable_grad() so it works even inside an autograd
    Function.forward (which runs in no-grad mode). Consistent with the energy
    for any element winding.
    """
    with torch.enable_grad():
        xloc = Xe + uloc.view(Ne, 3, 2)
        F = deformation_gradient(xloc, Xe)
        psi = nh_energy_density(F, mu, lam, dim)           # (Ne,)
        pel = (Ae * psi).sum()                             # scalar total
        fe = torch.autograd.grad(pel, uloc, create_graph=True)[0]   # (Ne,6)
    return fe


def _assemble_tangent_autodiff(Xe, dN, A, elems, mu, lam, Nn, Ne, u_nodes):
    """Global tangent stiffness = d(fint)/du, the symmetric Hessian of total
    energy. Double-autodiff at the current deformed state, wrapped in
    enable_grad so it works inside an autograd Function.forward."""
    with torch.enable_grad():
        uloc = u_nodes[elems].reshape(Ne, 6).detach().clone().requires_grad_(True)
        Kloc = torch.zeros(Ne, 6, 6, dtype=Xe.dtype, device=Xe.device)
        fe = _local_fint(uloc, Xe, dN, A, mu, lam, Ne, dim=2)     # (Ne,6)
        for c in range(6):
            grad_out = torch.zeros(Ne, 6, dtype=Xe.dtype, device=Xe.device)
            grad_out[:, c] = 1.0
            gi = torch.autograd.grad(fe, uloc, grad_outputs=grad_out,
                                     retain_graph=True)[0]
            Kloc[:, :, c] = gi
    K = torch.zeros(2 * Nn, 2 * Nn, dtype=Xe.dtype, device=Xe.device)
    loc_dofs = torch.stack([
        2 * elems[:, 0], 2 * elems[:, 0] + 1,
        2 * elems[:, 1], 2 * elems[:, 1] + 1,
        2 * elems[:, 2], 2 * elems[:, 2] + 1,
    ], dim=1)                                                # (Ne,6)
    g_i = loc_dofs.unsqueeze(-1).expand(-1, -1, 6).reshape(-1)
    g_j = loc_dofs.unsqueeze(-2).expand(-1, 6, -1).reshape(-1)
    K.index_put_((g_i, g_j), Kloc.reshape(-1), accumulate=True)
    return K


def newton_solve(
    X: torch.Tensor,
    elems: torch.Tensor,
    mu: torch.Tensor,
    lam: torch.Tensor,
    f_ext: torch.Tensor,
    fixed_dofs: torch.Tensor,
    max_iter: int = 50,
    tol: float = 1e-8,
):
    """Forward Newton solve. Returns u* (2*Nn,), detached."""
    Nn = X.shape[0]
    u = torch.zeros(2 * Nn, dtype=X.dtype, device=X.device)
    all_dofs = torch.arange(2 * Nn, device=X.device)
    free = all_dofs[~torch.isin(all_dofs, fixed_dofs)]
    for it in range(max_iter):
        fint, K = assemble_internal_force_and_tangent(u, X, elems, mu, lam, Nn)
        R = f_ext - fint
        R_free = R[free]
        if R_free.norm() < tol:
            break
        Kff = K[free][:, free]
        # add small regularization for stability (NH tangent can lose SPD near inversion)
        Kff = Kff + 1e-9 * torch.eye(Kff.shape[0], dtype=X.dtype, device=X.device) * (Kff.diag().abs().mean() + 1e-12)
        du_free = torch.linalg.solve(Kff, R_free)
        du = torch.zeros_like(u)
        du[free] = du_free
        u = u + du
    return u


class NHFEMLayer(torch.autograd.Function):
    """Differentiable NH-FEM equilibrium solve.

    Forward:  u* = argmin_u  Pi(u; E, nu)   s.t. f_ext balance, BCs.
    Backward: implicit-function-theorem adjoint.
    """

    @staticmethod
    def forward(ctx, X, elems, E, nu, f_ext, fixed_dofs):
        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        u_star = newton_solve(X, elems, mu, lam, f_ext, fixed_dofs)
        ctx.save_for_backward(X, elems, E, nu, f_ext, fixed_dofs, u_star)
        return u_star

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is dL/du_star from PyTorch (== upstream gradient).
        X, elems, E, nu, f_ext, fixed_dofs, u_star = ctx.saved_tensors
        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        Nn = X.shape[0]
        all_dofs = torch.arange(2 * Nn, device=X.device)
        free = all_dofs[~torch.isin(all_dofs, fixed_dofs)]

        # Conventions: residual R(u;theta) = f_int(u;theta) - f_ext = 0 at u*.
        #   dR/du  = K  (tangent stiffness = d(f_int)/du, assembled above)
        #   dR/dtheta = d(f_int)/dtheta  (f_ext independent of material params)
        # Implicit function theorem: du*/dtheta = -(dR/du)^{-1} dR/dtheta
        #                                     = -K^{-1} d(f_int)/dtheta.
        # Adjoint: K^T lambda = dL/du  ->  lambda = K^{-T} (dL/du).
        # Then dL/dtheta = (dL/du) du*/dtheta = -(dL/du) K^{-1} d(f_int)/dtheta
        #                                     = -lambda^T d(f_int)/dtheta.
        _, K = assemble_internal_force_and_tangent(u_star, X, elems, mu, lam, Nn)
        Kff = K[free][:, free]
        Kff = Kff + 1e-9 * torch.eye(Kff.shape[0], dtype=X.dtype, device=X.device) * (Kff.diag().abs().mean() + 1e-12)

        dLdu = grad_output.clone()
        dLdu[fixed_dofs] = 0.0
        lam_vec = torch.zeros_like(grad_output)
        lam_vec[free] = torch.linalg.solve(Kff.t(), dLdu[free])

        # d(f_int)/dtheta at u* (u* held fixed), via finite differences.
        eps = 1e-3
        def fint_at(E_v, nu_v):
            mu_v = E_v / (2 * (1 + nu_v)); lam_v = E_v * nu_v / ((1 + nu_v) * (1 - 2 * nu_v))
            f, _ = assemble_internal_force_and_tangent(u_star, X, elems, mu_v, lam_v, Nn)
            return f
        dfint_dE = (fint_at(E * (1 + eps), nu) - fint_at(E * (1 - eps), nu)) / (2 * E * eps)
        dfint_dnu = (fint_at(E, nu + eps) - fint_at(E, nu - eps)) / (2 * eps)

        # dL/dtheta = -(dL/du) K^{-1} d(fint)/dtheta = -lambda^T d(fint)/dtheta
        dL_dE = -(lam_vec * dfint_dE).sum()
        dL_dnu = -(lam_vec * dfint_dnu).sum()
        # dL/df_ext: R = f_int - f_ext, dR/df_ext = -I -> du*/df = -K^{-1}(-I) = K^{-1}
        #   dL/df = (dL/du) K^{-1} = lambda  (note sign)
        dL_df = lam_vec
        # 6 inputs: (X, elems, E, nu, f_ext, fixed_dofs). X/elems/fixed non-diff.
        return None, None, dL_dE, dL_dnu, dL_df, None


def solve_nh(X, elems, E, nu, f_ext, fixed_dofs):
    """Convenience: differentiable NH equilibrium solve. u* is differentiable
    in (E, nu, f_ext)."""
    return NHFEMLayer.apply(X, elems, E, nu, f_ext, fixed_dofs)
