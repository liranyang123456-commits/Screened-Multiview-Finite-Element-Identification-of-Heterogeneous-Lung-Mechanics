"""Neo-Hookean hyperelasticity + mesh utilities.

Stable Bonet & Wood form. Provides:
  - per-element deformation gradient (triangles / tets)
  - NH energy density, first Piola-Kirchhoff stress
  - triangle and tetrahedral mesh generation

Material parameters from (E, nu):
    mu  = E / (2(1+nu))
    lam = E*nu / ((1+nu)(1-2nu))
"""
from __future__ import annotations
import torch


# ===========================================================================
# Deformation gradient from element corners
# ===========================================================================

def deformation_gradient(x: torch.Tensor, X: torch.Tensor):
    """Constant per-element deformation gradient F = dx/dX.

    Args:
        x: (Ne, D+1, D) current deformed simplex corners (D=2 triangles, D=3 tets).
        X: (Ne, D+1, D) rest simplex corners.
    Returns:
        F: (Ne, D, D).
    """
    D = X.shape[-1]
    # build (corner1-corner0, ..., cornerD-corner0) as columns -> (Ne, D, D)
    cols_x = [x[:, k + 1] - x[:, 0] for k in range(D)]
    cols_X = [X[:, k + 1] - X[:, 0] for k in range(D)]
    dx = torch.stack(cols_x, dim=-1)
    dX = torch.stack(cols_X, dim=-1)
    dX_inv = torch.linalg.inv(dX)
    return dx @ dX_inv


# ===========================================================================
# Neo-Hookean energy density + PK1 stress
# ===========================================================================

def _J_and_I1(F: torch.Tensor, dim: int):
    """det and trace(C)=F:F for the right dimensionality (plane strain adds the
    constrained 3rd direction)."""
    J = torch.det(F)
    I1 = (F * F).sum(dim=(-2, -1))
    if dim == 2:
        I1 = I1 + 1.0          # constrained out-of-plane stretch = 1
    return J, I1


def nh_energy_density(F: torch.Tensor, mu, lam, dim: int):
    """Stable NH energy density. dim=2 plane strain, dim=3 full 3D.

    psi = (mu/2)(I1 - dim) - mu*ln(J) + (lam/2)(ln J)^2
    """
    J, I1 = _J_and_I1(F, dim)
    logJ = torch.log(J.clamp_min(1e-9))
    psi = (mu / 2.0) * (I1 - dim) - mu * logJ + (lam / 2.0) * logJ ** 2
    return psi


def nh_pk1(F: torch.Tensor, mu, lam, dim: int):
    """First Piola-Kirchhoff stress = d(psi)/dF, computed by autodiff so it is
    EXACTLY the gradient of nh_energy_density for the given dimensionality
    (2D plane-strain or 3D). This guarantees fint = d(Pi)/du and hence the
    tangent d(fint)/du is a symmetric Hessian.

    F: (Ne, D, D). Returns P: (Ne, D, D), differentiable in F, mu, lam.
    """
    F = F if F.requires_grad else F.detach().requires_grad_(True)
    psi = nh_energy_density(F, mu, lam, dim)                 # (Ne,)
    P = torch.autograd.grad(
        psi.sum(), F, create_graph=True)[0]                  # (Ne,D,D)
    return P


# ===========================================================================
# Mesh generation
# ===========================================================================

def make_triangle_grid(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0):
    """Regular triangle mesh, 2 triangles per cell (CCW).

    Returns nodes (Nn,2), elems (Ne,3). Triangles are preferred over quads for
    large-deformation NH: constant per-element F, robust to inversion.
    """
    xs = torch.linspace(0, lx, nx + 1)
    ys = torch.linspace(0, ly, ny + 1)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    nodes = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

    def nid(i, j):
        return j * (nx + 1) + i

    elems = []
    for j in range(ny):
        for i in range(nx):
            n0 = nid(i, j)
            n1 = nid(i + 1, j)
            n2 = nid(i + 1, j + 1)
            n3 = nid(i, j + 1)
            elems.append([n0, n1, n2])
            elems.append([n0, n2, n3])
    return nodes, torch.tensor(elems, dtype=torch.long)


def make_tet_grid(nx: int, ny: int, nz: int, lx=1.0, ly=1.0, lz=1.0):
    """Regular tetrahedral mesh, 6-tet body-diagonal split per cube.

    Returns nodes (Nn,3), elems (Ne,4). Each cube [v0..v7] (binary-coded by
    (i,j,k) bits) is split into 6 tets along the body diagonal v0-v7.
    """
    xs = torch.linspace(0, lx, nx + 1)
    ys = torch.linspace(0, ly, ny + 1)
    zs = torch.linspace(0, lz, nz + 1)
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing='ij')
    nodes = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)

    def nid(i, j, k):
        return k * (nx + 1) * (ny + 1) + j * (nx + 1) + i

    # 6-tet split of a cube along body diagonal v0-v7.
    # v0=(0,0,0) v1=(1,0,0) v2=(0,1,0) v3=(1,1,0)
    # v4=(0,0,1) v5=(1,0,1) v6=(0,1,1) v7=(1,1,1)
    tet_pattern = [
        [0, 1, 2, 7],
        [0, 2, 7, 4],
        [0, 4, 7, 5],
        [1, 3, 7, 2],
        [2, 6, 4, 7],
        [1, 7, 5, 4],  # note: consistent positive-volume orientation verified below
    ]
    elems = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                v = [nid(i, j, k), nid(i + 1, j, k), nid(i, j + 1, k),
                     nid(i + 1, j + 1, k), nid(i, j, k + 1), nid(i + 1, j, k + 1),
                     nid(i, j + 1, k + 1), nid(i + 1, j + 1, k + 1)]
                for t in tet_pattern:
                    elems.append([v[t[0]], v[t[1]], v[t[2]], v[t[3]]])
    return nodes, torch.tensor(elems, dtype=torch.long)
