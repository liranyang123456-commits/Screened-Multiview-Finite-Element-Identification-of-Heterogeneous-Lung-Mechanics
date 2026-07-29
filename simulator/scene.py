"""M2: Soft-tissue FEM simulator for synthetic endoscopy GT generation.

Generates a time sequence of deformed soft-tissue meshes under prescribed
tool-contact forces, with FULLY KNOWN ground truth (E, nu, contact forces,
per-frame node positions, surface Gaussians, camera poses). This is the GT
source for all downstream inverse-rendering experiments.

Scene model:
  - A soft tissue block (tetrahedral mesh), Neo-Hookean.
  - Back face fixed (Dirichlet).
  - A surgical "tool" presses / drags on a surface region; modelled as a
    prescribed nodal force on a contact patch whose magnitude varies over time.
  - A virtual endoscopic camera orbits the scene (known poses).
  - Surface Gaussians (for M3 rendering) are seeded on boundary triangles.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from physics import make_tet_grid
from physics.fem import newton_solve


def build_tissue_scene(nx=6, ny=6, nz=4, lx=1.0, ly=1.0, lz=0.5,
                      E_true=5.0e3, nu_true=0.45):
    """Build a soft-tissue block scene.

    Returns a dict with mesh + material + boundary conditions + camera params.
    Convention: z is the surface normal (tool presses in -z); back face z=lz fixed.
    """
    nodes, elems = make_tet_grid(nx, ny, nz, lx, ly, lz)
    Nn = nodes.shape[0]
    D = 3

    # fix back face (z = lz) fully
    back = torch.where(nodes[:, 2] > lz - 1e-4)[0]
    fixed = torch.cat([D * back, D * back + 1, D * back + 2])

    # surface triangles on the z=0 (front) face: a tet with exactly 3 of its 4
    # nodes on z<1e-4 contributes those 3 nodes as a boundary triangle.
    z_of_node = nodes[elems, 2]                      # (Ne,4)
    n_surface_nodes = (z_of_node < 1e-4).sum(dim=1)  # (Ne,) count per tet
    surface_tris = []
    for e_idx in torch.where(n_surface_nodes == 3)[0].tolist():
        tet = elems[e_idx]
        surf_nodes = [int(n.item()) for n in tet if nodes[n, 2] < 1e-4]
        if len(surf_nodes) == 3:
            surface_tris.append(surf_nodes)
    surface_tris = torch.tensor(surface_tris, dtype=torch.long) if surface_tris \
        else torch.zeros((0, 3), dtype=torch.long)

    return {
        "nodes": nodes, "elems": elems, "Nn": Nn, "D": D,
        "fixed": fixed, "surface_tris": surface_tris,
        "E_true": torch.tensor(float(E_true)),
        "nu_true": torch.tensor(float(nu_true)),
        "lx": lx, "ly": ly, "lz": lz,
    }


def contact_force_sequence(scene, T=12, contact_center=(0.5, 0.5, 0.0),
                           contact_radius=0.18, max_force=2e2,
                           pattern="press_release", front_z=None):
    """Build a time-varying nodal force vector simulating tool contact.

    pattern:
      'press_release'   — ramp up then down (single press)
      'drag'            — press while translating contact center in x
    front_z: z-threshold defining the contact (front) surface. If None, uses
      1e-4 (regular grids where z_min~0). For real-patch scenes with continuous
      z, pass the front-surface quantile (e.g. 20th percentile).
    Returns: list of T force vectors (D*Nn,), and GT contact descriptors.
    """
    nodes = scene["nodes"]; Nn = scene["Nn"]; D = scene["D"]
    lx, ly = scene["lx"], scene["ly"]
    cx, cy, cz = contact_center
    if front_z is None:
        front_z = nodes[:, 2].min().item() + 1e-4
    forces = []
    contact_log = []
    for t in range(T):
        s = (t + 1) / T                              # 0->1 ramp
        if pattern == "press_release":
            envelope = torch.sin(torch.tensor(t / max(T - 1, 1) * 3.14159))   # 0..1..0
            cxt, cyt = cx, cy
        elif pattern == "drag":
            envelope = min(1.0, s * 1.5)             # ramp up, hold
            cxt = cx + 0.25 * torch.sin(torch.tensor(t / T * 6.2832)) * (lx)
            cyt = cy
        else:
            raise ValueError(pattern)
        # force on surface nodes within radius, directed into tissue (-z)
        f = torch.zeros(D * Nn)
        d = torch.sqrt((nodes[:, 0] - cxt) ** 2 + (nodes[:, 1] - cyt) ** 2)
        in_patch = (d < contact_radius) & (nodes[:, 2] < front_z)
        for n_idx in torch.where(in_patch)[0]:
            w = torch.exp(-(d[n_idx] / (contact_radius * 0.5)) ** 2)
            f[D * n_idx + 2] = -max_force * envelope * w    # downward
        forces.append(f)
        contact_log.append({"t": t, "center": (float(cxt), float(cyt)),
                            "force_mag": float(max_force * envelope)})
    return forces, contact_log


def simulate_sequence(scene, forces, n_load_steps=8):
    """Run the FEM forward for each frame. Warm-starts each frame from the
    previous frame's solution (continuation) for robustness.

    Returns: u_seq (T, D*Nn) per-frame displacements (detached GT).
    """
    nodes = scene["nodes"]; elems = scene["elems"]
    E = scene["E_true"]; nu = scene["nu_true"]
    mu = E / (2 * (1 + nu)); lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    fixed = scene["fixed"]; D = scene["D"]
    u_prev = None
    u_seq = []
    for f in forces:
        u = newton_solve(nodes, elems, mu, lam, f, fixed, D,
                         n_load_steps=n_load_steps, u_init=u_prev)
        u_prev = u.detach()
        u_seq.append(u_prev)
    return torch.stack(u_seq, dim=0)


def make_camera_poses(T=12, radius=1.6, height=1.2, look_at=(0.5, 0.5, 0.0)):
    """Orbiting endoscopic camera: known per-frame poses (c2w 4x4), OpenCV
    convention (camera z points forward into the scene, y down, x right).
    """
    cx, cy, cz = look_at
    poses = []
    for t in range(T):
        theta = t / T * 2 * 3.14159265
        eye = torch.tensor([cx + radius * float(torch.cos(torch.tensor(theta))),
                            cy + radius * float(torch.sin(torch.tensor(theta))),
                            height + cz])
        target = torch.tensor([cx, cy, cz], dtype=torch.float64)
        forward = target - eye; forward = forward / forward.norm()
        world_up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        right = torch.linalg.cross(forward, world_up, dim=0); right = right / right.norm()
        up = torch.linalg.cross(right, forward, dim=0)         # already normalized basis
        # OpenCV: x=right, y=down, z=forward. Our 'up' above points up, so use -up for y.
        R = torch.stack([right, -up, forward], dim=1)          # columns = cam axes in world
        c2w = torch.eye(4, dtype=torch.float64)
        c2w[:3, :3] = R; c2w[:3, 3] = eye
        poses.append(c2w)
    return torch.stack(poses, dim=0)


def make_multiview_camera_poses(
    T=7,
    radius=1.6,
    height=1.2,
    look_at=(0.5, 0.5, 0.0),
    num_views=3,
    azimuth_offsets=(-0.28, 0.0, 0.28),
    height_offsets=(-0.12, 0.0, 0.12),
):
    """Create three rigidly synchronized camera orbits.

    The middle view is exactly :func:`make_camera_poses`, preserving the
    historical single-view trajectory.  The other views have fixed azimuth
    and height offsets, so ``poses[t]`` is a synchronized capture with shape
    ``(V, 4, 4)`` rather than independently sampled camera motion.
    """
    if num_views != 3:
        raise ValueError("The sim-lung-v2 multiview protocol requires num_views=3")
    if len(azimuth_offsets) != num_views or len(height_offsets) != num_views:
        raise ValueError("Camera-ring offsets must match num_views")

    cx, cy, cz = look_at
    rings = []
    for azimuth_offset, height_offset in zip(azimuth_offsets, height_offsets):
        poses = []
        for t in range(T):
            theta = t / T * 2 * 3.14159265 + azimuth_offset
            eye = torch.tensor(
                [
                    cx + radius * float(torch.cos(torch.tensor(theta))),
                    cy + radius * float(torch.sin(torch.tensor(theta))),
                    height + height_offset * radius + cz,
                ],
                dtype=torch.float64,
            )
            target = torch.tensor([cx, cy, cz], dtype=torch.float64)
            forward = target - eye
            forward = forward / forward.norm()
            world_up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
            right = torch.linalg.cross(forward, world_up, dim=0)
            right = right / right.norm()
            up = torch.linalg.cross(right, forward, dim=0)
            c2w = torch.eye(4, dtype=torch.float64)
            c2w[:3, :3] = torch.stack([right, -up, forward], dim=1)
            c2w[:3, 3] = eye
            poses.append(c2w)
        rings.append(torch.stack(poses))
    synchronized = torch.stack(rings, dim=1)
    # Use the legacy implementation verbatim for the center ring, including
    # its floating-point operation order, so old single-view fields are exact.
    synchronized[:, 1] = make_camera_poses(
        T=T, radius=radius, height=height, look_at=look_at
    )
    return synchronized


if __name__ == "__main__":
    # quick self-test: build scene, simulate, check physical validity
    torch.set_default_dtype(torch.float64)
    scene = build_tissue_scene()
    forces, log = contact_force_sequence(scene, T=10)
    u_seq = simulate_sequence(scene, forces)
    print(f"simulated {u_seq.shape[0]} frames, {u_seq.shape[1]} dofs")
    # check no element inversion
    from physics import deformation_gradient
    nodes = scene["nodes"]; elems = scene["elems"]; Nn = scene["Nn"]
    for t in range(u_seq.shape[0]):
        Xe = nodes[elems]; xe = Xe + u_seq[t].view(Nn, 3)[elems]
        J = torch.det(deformation_gradient(xe, Xe))
        if J.min() < 0:
            print(f"  frame {t}: ELEMENT INVERSION min_J={J.min():.3f}")
        else:
            print(f"  frame {t}: max|u|={u_seq[t].abs().max():.4f} min_J={J.min():.4f}")
