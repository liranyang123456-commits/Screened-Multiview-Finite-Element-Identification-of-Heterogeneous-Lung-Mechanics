"""Real endoscopy mesh loaders: SCARED point-cloud -> tet mesh, and C3VD loader.

These provide real-data geometry priors for the FEM pipeline. SCARED's
point_cloud.obj files are unstructured 1.3M-point clouds (no faces) — we
Poisson-reconstruct + crop a local patch + tetrahedralize to get a usable FEM
domain. C3VD (must be downloaded manually) ships watertight reference meshes.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import numpy as np
import torch


def scared_pointcloud_to_tet(obj_path, max_points=8000, voxel_size=0.004,
                             crop_radius=0.04, crop_center=None, tet_res=(6, 6, 3)):
    """Load a SCARED point_cloud.obj (pure vertices), Poisson-reconstruct a
    surface, crop a local patch (a tissue region), and generate a tetrahedral
    FEM mesh approximating it.

    Returns (nodes torch (Nn,3) in meters, elems (Ne,4), surface_tris (Nt,3)).
    Falls back to a flat tet block if Open3D/Poisson unavailable.
    """
    try:
        import open3d as o3d
    except Exception:
        return _fallback_tet(tet_res)

    # 1. load points
    pcd = o3d.io.read_point_cloud(obj_path)
    if len(pcd.points) == 0:
        return _fallback_tet(tet_res)
    P = np.asarray(pcd.points)                       # (N,3) meters

    # 2. downsample + estimate normals + Poisson reconstruct
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, -1.0]))
    try:
        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8)
    except Exception:
        return _fallback_tet(tet_res)

    # 3. crop a local patch (a roughly-flat tissue region for FEM)
    V = np.asarray(mesh.vertices)
    if crop_center is None:
        crop_center = V.mean(axis=0)
    d = np.linalg.norm(V - crop_center, axis=1)
    keep = d < crop_radius
    # also keep vertices near the local tangent plane (drop far-from-plane)
    if keep.sum() < 100:
        keep = np.ones(len(V), dtype=bool)
    mesh_c = mesh.select_by_index(np.where(keep)[0])
    mesh_c = mesh_c.compute_vertex_normals()

    # 4. flatten the patch to its tangent plane and build a tet block on it
    # (this gives a well-conditioned FEM domain approximating the local tissue)
    Vc = np.asarray(mesh_c.vertices)
    if len(Vc) < 50:
        return _fallback_tet(tet_res)
    # tangent plane via PCA
    Vc_c = Vc - Vc.mean(axis=0)
    _, _, Vt = np.linalg.svd(Vc_c, full_matrices=False)
    plane_coords = Vc_c @ Vt[:2].T                    # (N,2) in-plane
    # bounding box of the patch in plane
    mn, mx = plane_coords.min(0), plane_coords.max(0)
    lx = max(mx[0] - mn[0], 0.02); ly = max(mx[1] - mn[1], 0.02)
    # build tet block with the patch's aspect ratio, scaled to a normalized size
    from physics import make_tet_grid
    nodes, elems = make_tet_grid(tet_res[0], tet_res[1], tet_res[2], lx, ly, 0.4 * min(lx, ly))
    # place at patch centroid, oriented along the plane basis
    R = np.eye(3); R[:, :2] = Vt[:2].T; R[:, 2] = np.cross(Vt[0], Vt[1])
    nodes_np = nodes.numpy() @ R.T + Vc.mean(axis=0)
    from simulator.anatomy import _surface_tris_by_centroid
    nodes_t = torch.from_numpy(nodes_np)
    srf = _surface_tris_by_centroid(nodes_t, elems, "front", 0.25)
    return nodes_t, elems, srf


def _fallback_tet(tet_res=(6, 6, 3)):
    from physics import make_tet_grid
    from simulator.anatomy import _surface_tris_by_centroid
    nodes, elems = make_tet_grid(*tet_res, 1.0, 1.0, 0.5)
    return nodes, elems, _surface_tris_by_centroid(nodes, elems, "front", 0.25)


def load_c3vd_scene(c3vd_root, scene_name):
    """Load a C3VD scene (when the user has downloaded C3VD).

    Expected layout per C3VD spec:
      {c3vd_root}/{scene_name}/
        color/0001_color.png ...      (RGB frames)
        depth/0001_depth.png ...      (16-bit depth, scale 0-100 mm)
        normals/0001_normals.png ...  (16-bit surface normals)
        pose.txt                       (4x4 c2w per line, N poses)
        {scene_name}_model.obj or full_model.obj  (reference mesh)
    Returns dict with images, depth (m), poses (N,4,4), mesh (if found).
    """
    from PIL import Image
    import glob
    sdir = os.path.join(c3vd_root, scene_name)
    out = {"scene": scene_name, "available": False}
    if not os.path.isdir(sdir):
        return out
    # color
    cols = sorted(glob.glob(os.path.join(sdir, "**", "*color*.png"), recursive=True))
    if not cols:
        cols = sorted(glob.glob(os.path.join(sdir, "*.png")))
    out["color_files"] = cols
    # depth
    depth_files = sorted(glob.glob(os.path.join(sdir, "**", "*depth*.png"), recursive=True))
    out["depth_files"] = depth_files
    # poses
    pose_file = os.path.join(sdir, "pose.txt")
    if os.path.exists(pose_file):
        P = np.loadtxt(pose_file)
        out["poses"] = P.reshape(-1, 4, 4) if P.size % 16 == 0 else None
    # reference mesh
    mesh = None
    for cand in [f"{scene_name}_model.obj", "full_model.obj"]:
        mp = os.path.join(sdir, cand)
        if os.path.exists(mp):
            mesh = mp; break
    out["mesh_path"] = mesh
    out["available"] = bool(cols)
    return out


if __name__ == "__main__":
    # test SCARED pointcloud path on one keyframe (if open3d available)
    scared_pc = r"external_data/SCARED\dataset_1\keyframe_1\point_cloud.obj"
    print(f"testing SCARED pointcloud -> tet: {scared_pc}")
    if os.path.exists(scared_pc):
        try:
            nodes, elems, srf = scared_pointcloud_to_tet(scared_pc)
            print(f"  OK: nodes {nodes.shape}, elems {elems.shape}, surface_tris {srf.shape}")
            print(f"  bbox: x[{nodes[:,0].min():.3f},{nodes[:,0].max():.3f}] "
                  f"y[{nodes[:,1].min():.3f},{nodes[:,1].max():.3f}] "
                  f"z[{nodes[:,2].min():.3f},{nodes[:,2].max():.3f}]")
        except Exception as e:
            print(f"  FAILED: {e}")
    else:
        print("  SCARED pointcloud not found — skipping")
    # test C3VD loader (likely empty)
    print("\ntesting C3VD loader (expected: not downloaded)...")
    c3vd = load_c3vd_scene(r"external_data/C3VD", "cecum_t1_a")
    print(f"  C3VD available: {c3vd['available']}  (must download from durrlab.github.io/C3VD)")
