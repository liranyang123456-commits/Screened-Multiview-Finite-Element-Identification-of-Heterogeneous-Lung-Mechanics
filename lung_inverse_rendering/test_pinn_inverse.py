from __future__ import annotations

import torch

from physics.fem import assemble_internal_force, node_E_to_per_elem_mu_lam


def test_assemble_internal_force_backprops_to_displacement_and_modulus() -> None:
    nodes = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    elems = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    E_nodes = torch.tensor([5000.0, 5000.0, 5200.0, 5100.0], dtype=torch.float64, requires_grad=True)
    nu = torch.tensor(0.45, dtype=torch.float64)
    u = torch.zeros(12, dtype=torch.float64, requires_grad=True)
    u = u + 0.01 * torch.arange(12, dtype=torch.float64)
    mu, lam, _ = node_E_to_per_elem_mu_lam(E_nodes, elems, nu)
    fint = assemble_internal_force(u, nodes, elems, mu, lam, 4, 3)
    loss = fint.square().sum()
    grads = torch.autograd.grad(loss, [u, E_nodes], allow_unused=True)
    assert grads[0] is not None and torch.isfinite(grads[0]).all()
    assert grads[1] is not None and torch.isfinite(grads[1]).all()
