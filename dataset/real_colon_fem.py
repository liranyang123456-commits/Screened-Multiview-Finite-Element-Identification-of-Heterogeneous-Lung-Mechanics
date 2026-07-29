"""Real colon mesh -> FEM tetrahedral domain.

C3VD reference meshes are open (non-watertight) colon surfaces with ~100k-230k
vertices. To use them as FEM geometry we:
  1. load + downsample a segment mesh
  2. crop a local tissue patch (a roughly-flat region the endoscope would see)
  3. remesh the patch to a coarse, regular triangulation
  4. extrude along the patch normal to form a thin volumetric shell
  5. tetrahedralize the shell into a Neo-Hookean FEM domain

The result is a real-colon-shaped FEM patch (not a procedural block), which we
render with our Gaussian-PBR pipeline. This grounds the benchmark in real
endoscopic geometry while keeping the mechanics fully known (synthetic E/nu).
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch


def load_colon_segment(obj_path, voxel=0.8):
    """Load a C3VD colon segment mesh, downsample. Returns open3d TriangleMesh."""
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(obj_path)
    if len(m.vertices) == 0:
        return None
    m = m.simplify_vertex_clustering(voxel, o3d.geometry.SimplificationContraction.Average)
    m.compute_vertex_normals()
    return m


def crop_patch(mesh, center=None, radius=None, n_target=400, max_radius_frac=0.15,
               absolute_radius_cap=None):
    """Crop a local patch of the surface around `center`. Returns (verts (n,3),
    faces (m,3), center, normal). The patch is roughly flat -> good FEM base.

    radius is chosen as the min of (15th-percentile distance, absolute_radius_cap)
    so that huge meshes (e.g. the full-colon model) don't yield oversized patches.
    The cap keeps the patch at a consistent endoscopic-FOV-like scale across
    segments, which is what makes the resulting FEM domains comparable.
    """
    import open3d as o3d
    V = np.asarray(mesh.vertices)
    if center is None:
        center = V.mean(axis=0)
    d = np.linalg.norm(V - center, axis=1)
    if radius is None:
        pct_radius = np.percentile(d, max_radius_frac * 100)
        if absolute_radius_cap is None:
            radius = pct_radius
        else:
            radius = min(pct_radius, absolute_radius_cap)
    keep = d < radius
    # submesh: keep triangles whose all 3 verts are kept
    F = np.asarray(mesh.triangles)
    keep_faces = keep[F].all(axis=1)
    # relabel
    idx_map = -np.ones(len(V), dtype=int)
    idx_map[keep] = np.arange(keep.sum())
    Fp = idx_map[F[keep_faces]]
    Vp = V[keep]
    # patch normal via SVD (dominant plane)
    Vc = Vp - Vp.mean(axis=0)
    _, _, Vt = np.linalg.svd(Vc, full_matrices=False)
    normal = Vt[-1]                            # smallest variance direction
    if normal[2] < 0:
        normal = -normal                        # point towards camera (-z-ish)
    return Vp, Fp, Vp.mean(axis=0), normal


def patch_to_tet_shell(Vp, Fp, normal, grid_nx=7, grid_ny=7, thickness_frac=0.3,
                       extent=None):
    """Flatten the patch to its tangent plane, build a regular tetrahedral block
    with the patch's in-plane extent, then re-embed in 3D oriented along the
    patch basis. This gives a well-conditioned FEM domain shaped to the real
    colon patch (real aspect ratio, real orientation), used as the deformable
    tissue. Returns (nodes (Nn,3), elems (Ne,4))."""
    from physics import make_tet_grid
    Vc = Vp - Vp.mean(axis=0)
    _, _, Vt = np.linalg.svd(Vc, full_matrices=False)
    plane = Vc @ Vt[:2].T                       # (n,2) in-plane coords
    if extent is None:
        mn, mx = plane.min(0), plane.max(0)
        lx = max(mx[0] - mn[0], 1.0)
        ly = max(mx[1] - mn[1], 1.0)
    else:
        lx, ly = extent
    lz = thickness_frac * min(lx, ly)
    nodes, elems = make_tet_grid(grid_nx, grid_ny, 3, lx, ly, lz)
    # orient the block along the patch plane basis, place at patch centroid
    # build rotation: block xy -> Vt[0],Vt[1]; block z -> normal
    R = np.eye(3)
    R[:, 0] = Vt[0]; R[:, 1] = Vt[1]; R[:, 2] = np.cross(Vt[0], Vt[1])
    # normalize columns
    R = R / np.linalg.norm(R, axis=0, keepdims=True)
    # recenter block at origin then place at patch centroid
    nodes_np = nodes.numpy()
    nodes_np = nodes_np - np.array([lx / 2, ly / 2, lz / 2])
    nodes_np = nodes_np @ R.T + Vp.mean(axis=0)
    return torch.from_numpy(nodes_np), elems


def build_real_colon_scene(obj_path, segment_name="colon", E_true=5e3,
                           nu_true=0.45, grid=(7, 7), voxel=0.8, seed=0,
                           patch_radius_cap=35.0):
    """Full pipeline: C3VD obj -> FEM scene dict. Returns scene dict compatible
    with simulator/renderer, with anatomy='real_colon_{segment}'.

    patch_radius_cap: absolute cap (in mesh units) on the crop radius, so huge
    meshes (full colon) yield patches comparable to smaller segments. Default
    35 matches the typical segment patch radius (~30)."""
    import open3d as o3d
    from simulator.anatomy import _surface_tris_by_centroid
    rng = np.random.RandomState(seed)
    mesh = load_colon_segment(obj_path, voxel=voxel)
    if mesh is None or len(mesh.vertices) < 50:
        # fallback to flat block
        from simulator.anatomy import make_flat_block
        nodes, elems, srf, _ = make_flat_block()
    else:
        V = np.asarray(mesh.vertices)
        # pick a random visible-ish patch center (interior point)
        center = V[rng.randint(len(V))]
        Vp, Fp, c, n = crop_patch(mesh, center=center,
                                  absolute_radius_cap=patch_radius_cap)
        nodes, elems = patch_to_tet_shell(Vp, Fp, n, grid_nx=grid[0], grid_ny=grid[1])
        srf = _surface_tris_by_centroid(nodes, elems, "front", 0.25)
    # standardize: translate so min corner at origin, scale to ~unit
    nodes = nodes.double()
    nodes = nodes - nodes.min(0).values
    s = nodes.max(0).values
    scale = float(s[:2].max())                  # normalize so max in-plane dim = 1
    nodes = nodes / scale
    Nn = nodes.shape[0]; D = 3
    # Use z quantiles (not thresholds) because real-patch z is continuous, not
    # layered like a regular grid. Fix the back ~20% slab, mark the front ~20%
    # slab as the contact/imaged surface (via surface_tris, already centroid-based).
    z_q_hi = torch.quantile(nodes[:, 2], 0.80)
    back = torch.where(nodes[:, 2] > z_q_hi)[0]
    fixed = torch.cat([D * back, D * back + 1, D * back + 2])
    return {
        "nodes": nodes, "elems": elems, "Nn": Nn, "D": D,
        "fixed": fixed, "surface_tris": srf,
        "E_true": torch.tensor(float(E_true), dtype=torch.float64),
        "nu_true": torch.tensor(float(nu_true), dtype=torch.float64),
        "lx": float(nodes[:, 0].max()), "ly": float(nodes[:, 1].max()),
        "lz": float(nodes[:, 2].max() - nodes[:, 2].min()),
        "anatomy": f"real_colon_{segment_name}",
    }


if __name__ == "__main__":
    import glob
    c3vd = "dataset/C3VD"
    segs = sorted(glob.glob(os.path.join(c3vd, "*_model.obj")))
    print(f"found {len(segs)} colon segments")
    for obj in segs[:3]:                          # test 3 segments
        name = os.path.basename(obj).replace("_model.obj", "")
        try:
            s = build_real_colon_scene(obj, segment_name=name, seed=0)
            # verify non-degenerate + solvable
            Xe = s["nodes"][s["elems"]]
            cols = torch.stack([Xe[:, k + 1] - Xe[:, 0] for k in range(3)], dim=-1)
            J0 = torch.linalg.det(cols)
            print(f"{name:10s}: Nn={s['Nn']:4d} Ne={s['elems'].shape[0]:4d} "
                  f"srf={s['surface_tris'].shape[0]:4d} sing={int((J0.abs()<1e-8).sum())} "
                  f"bbox={torch.round(s['nodes'].max(0).values,decimals=2).tolist()}")
        except Exception as e:
            print(f"{name:10s}: FAILED {e}")
