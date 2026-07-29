"""Parametric multi-anatomy FEM geometry generator.

Generates realistic-shaped tissue meshes (tetrahedral) for different anatomical
regions, to build a diverse simulation benchmark ("different scenes, different
parts"). Each generator returns (nodes, elems, surface_tris, region_tags) where
region_tags mark which nodes are on the deformable surface vs. fixed back, so
the simulator can apply anatomically-plausible boundary conditions.

Anatomy types:
  polyp      — dome/bump on a flat base (colon polyp, the C3VD use case)
  liver_lobe — elongated wedge with rounded top (liver lobe)
  vessel     — half-tube / trough (blood vessel side wall)
  stomach    — gently curved flat sheet (stomach wall)
  flat_block — the original block (baseline / control)

All meshes are tetrahedral via physics.make_tet_grid, then reshaped. The
surface-triangle extraction is anatomy-aware (the imaged front surface).
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from physics import make_tet_grid


def _surface_tris_by_centroid(nodes, elems, side="front", frac=0.25):
    """Extract boundary triangles via per-tet node-count on a z-slab whose
    boundary is set by the tet CENTROID distribution (robust to deformed fronts).

    A tet contributes its 3 front-most (or back-most) nodes as a surface triangle
    if exactly 3 of its 4 nodes lie in the front slab and the slab is defined by
    centroid quantile. Falls back gracefully when shapes are curved.
    """
    zc = nodes[elems, 2].mean(dim=1)                  # (Ne,) tet centroids
    if side == "front":
        z_cut = torch.quantile(zc, frac)
        slab = nodes[:, 2] <= z_cut
    else:
        z_cut = torch.quantile(zc, 1.0 - frac)
        slab = nodes[:, 2] >= z_cut
    n_in = slab[elems].sum(dim=1)
    tris = []
    for e_idx in torch.where(n_in == 3)[0].tolist():
        tet = elems[e_idx]
        surf = [int(n.item()) for n in tet if slab[n]]
        if len(surf) == 3:
            tris.append(surf)
    return torch.tensor(tris, dtype=torch.long) if tris else torch.zeros((0, 3), dtype=torch.long)


# =============================================================================
# Anatomy generators
# =============================================================================

def make_flat_block(nx=6, ny=6, nz=3, lx=1.0, ly=1.0, lz=0.5):
    """Control: flat block (original)."""
    nodes, elems = make_tet_grid(nx, ny, nz, lx, ly, lz)
    return nodes, elems, _surface_tris_by_centroid(nodes, elems, "front", 0.25), "flat_block"


def make_polyp(nx=8, ny=8, nz=4, base=1.0, thickness=0.4,
               dome_height=0.25, dome_radius=0.25, dome_center=(0.5, 0.5)):
    """Colon polyp: flat base + hemispherical dome on the front (z=0) surface.
    The dome is the deformable lesion; tool presses on the dome apex.
    """
    nodes, elems = make_tet_grid(nx, ny, nz, base, base, thickness)
    # raise front-surface nodes inside the dome footprint
    cx, cy = dome_center
    dx = nodes[:, 0] - cx
    dy = nodes[:, 1] - cy
    r = torch.sqrt(dx ** 2 + dy ** 2)
    z_min = nodes[:, 2].min()
    on_front = nodes[:, 2] < (z_min + 1e-4)
    # hemispherical bump: z -= dome_height * sqrt(max(0, 1-(r/R)^2))
    inside = (r < dome_radius) & on_front
    bump = torch.sqrt((1 - (r / dome_radius) ** 2).clamp_min(0))
    nodes[:, 2] = torch.where(inside, nodes[:, 2] - dome_height * bump, nodes[:, 2])
    return nodes, elems, _surface_tris_by_centroid(nodes, elems, "front", 0.25), "polyp"


def make_liver_lobe(nx=8, ny=6, nz=4, length=1.2, width=0.7, thickness=0.45):
    """Liver lobe: elongated wedge with a rounded top (front surface tilted)."""
    nodes, elems = make_tet_grid(nx, ny, nz, length, width, thickness)
    # taper the +y end (narrower lobe tip)
    y_norm = nodes[:, 1] / width
    taper = 1.0 - 0.4 * (y_norm ** 2)
    nodes[:, 0] = (nodes[:, 0] - length / 2) * taper + length / 2
    # dome the front (z=0) surface along x
    z_min = nodes[:, 2].min()
    x_norm = nodes[:, 0] / length
    dome = 0.18 * torch.sin(3.14159 * x_norm) * (nodes[:, 2] < z_min + 1e-4)
    nodes[:, 2] = nodes[:, 2] - dome
    return nodes, elems, _surface_tris_by_centroid(nodes, elems, "front", 0.25), "liver_lobe"


def make_vessel_tube(nx=10, ny=6, nz=4, length=1.2, width=0.6, depth=0.4,
                     trough_depth=None):
    """Blood vessel: a trough/half-pipe — the front (z=0) surface is scooped
    along the x axis to form a concave vessel lumen wall. trough_depth is
    auto-capped below the z-layer spacing to keep all tets non-degenerate."""
    nodes, elems = make_tet_grid(nx, ny, nz, length, width, depth)
    z_spacing = depth / nz
    if trough_depth is None:
        trough_depth = 0.6 * z_spacing                  # safe (< 1 layer)
    trough_depth = min(trough_depth, 0.7 * z_spacing)
    z_min = nodes[:, 2].min()
    on_front = nodes[:, 2] < z_min + 1e-4
    y_norm = (nodes[:, 1] / width) * 2 - 1           # -1..1
    scoop = trough_depth * (1 - y_norm ** 2)          # deepest at y center
    nodes[:, 2] = torch.where(on_front, nodes[:, 2] + scoop, nodes[:, 2])
    return nodes, elems, _surface_tris_by_centroid(nodes, elems, "front", 0.25), "vessel_tube"


def make_stomach_wall(nx=8, ny=8, nz=3, length=1.0, width=1.0, thickness=0.35,
                      curvature=0.12):
    """Stomach wall: gently globally curved sheet (single large radius fold)."""
    nodes, elems = make_tet_grid(nx, ny, nz, length, width, thickness)
    z_min = nodes[:, 2].min()
    x_norm = nodes[:, 0] / length - 0.5
    y_norm = nodes[:, 1] / width - 0.5
    fold = curvature * (x_norm ** 2 + y_norm ** 2)
    on_front = nodes[:, 2] < z_min + 1e-4
    nodes[:, 2] = torch.where(on_front, nodes[:, 2] - fold, nodes[:, 2])
    return nodes, elems, _surface_tris_by_centroid(nodes, elems, "front", 0.25), "stomach_wall"


ANATOMY_REGISTRY = {
    "flat_block": make_flat_block,
    "polyp": make_polyp,
    "liver_lobe": make_liver_lobe,
    "vessel_tube": make_vessel_tube,
    "stomach_wall": make_stomach_wall,
}


def build_anatomy_scene(anatomy="polyp", E_true=5e3, nu_true=0.45, **kw):
    """Build a scene dict (compatible with simulator.scene convention) for the
    given anatomy. Returns scene dict with nodes/elems/surface_tris/fixed/..."""
    nodes, elems, surface_tris, name = ANATOMY_REGISTRY[anatomy](**kw)
    Nn = nodes.shape[0]; D = 3
    z_max = nodes[:, 2].max().item()
    back = torch.where(nodes[:, 2] > z_max - 1e-3)[0]
    fixed = torch.cat([D * back, D * back + 1, D * back + 2])
    return {
        "nodes": nodes, "elems": elems, "Nn": Nn, "D": D,
        "fixed": fixed, "surface_tris": surface_tris,
        "E_true": torch.tensor(float(E_true), dtype=torch.float64),
        "nu_true": torch.tensor(float(nu_true), dtype=torch.float64),
        "lx": float(nodes[:, 0].max()), "ly": float(nodes[:, 1].max()),
        "lz": float(nodes[:, 2].max() - nodes[:, 2].min()),
        "anatomy": name,
    }


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    for anat in ANATOMY_REGISTRY:
        s = build_anatomy_scene(anat)
        from physics import deformation_gradient
        # verify no element inversion at rest (all J ~ 1)
        Xe = s["nodes"][s["elems"]]
        cols = torch.stack([Xe[:, k + 1] - Xe[:, 0] for k in range(3)], dim=-1)
        J0 = torch.linalg.det(cols)
        print(f"{anat:14s}: Nn={s['Nn']:4d} Ne={s['elems'].shape[0]:4d} "
              f"surface_tris={s['surface_tris'].shape[0]:4d} "
              f"fixed={s['fixed'].shape[0]//3:4d} nodes "
              f"J0[min]={J0.min():.3f} z[{s['nodes'][:,2].min():.3f},{s['nodes'][:,2].max():.3f}]")
