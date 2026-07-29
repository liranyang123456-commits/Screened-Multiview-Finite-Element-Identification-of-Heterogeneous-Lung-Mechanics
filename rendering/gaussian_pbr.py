"""M3+M4: differentiable Gaussian-splat renderer with PBR shading, bound to a
deforming FEM mesh.

Design:
  - Surface Gaussians are seeded on boundary triangles of the FEM mesh.
  - Each Gaussian's center is a barycentric combination of the 3 triangle
    vertices, so it deforms with the mesh (M4 coupling).
  - Each Gaussian carries: position (driven), color (albedo), roughness,
    scale, opacity. Lighting = co-located point light at camera (endoscopy).
  - Rendering: project Gaussians to screen, splat with alpha compositing,
    apply diffuse + specular (Cook-Torrance-ish) shading per Gaussian.
  - All differentiable in (node positions, albedo, roughness, light, poses).

This is a minimal but physically-grounded differentiable renderer sufficient
to test optical+mechanical joint inverse rendering. It does NOT implement SSS
(SSS is a documented future extension; see spec R3).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _per_tri_von_mises_strain(xe, Xe):
    """Per-triangle von-Mises equivalent strain from the Green-Lagrange strain.

    xe, Xe: (Nt,3,3) deformed/rest triangle vertices. Returns (Nt,) scalar
    von-Mises strain. Uses the small-strain approximation E = sym(F-I) (valid
    for the moderate strains in our scenes); fully differentiable.
    """
    # F per triangle (2D in-plane, since triangles are surface elements)
    dX = torch.stack([Xe[:, 1] - Xe[:, 0], Xe[:, 2] - Xe[:, 0]], dim=-1)[:, :2, :]  # (Nt,2,2)
    dx = torch.stack([xe[:, 1] - xe[:, 0], xe[:, 2] - xe[:, 0]], dim=-1)[:, :2, :]
    Finv_X = torch.linalg.inv(dX)
    F = dx[:, :, :2] @ Finv_X                                  # (Nt,2,2) in-plane F
    eps = 0.5 * (F + F.transpose(-2, -1)) - torch.eye(2, dtype=xe.dtype)  # (Nt,2,2)
    # plane-stress von Mises equivalent strain
    ex, ey = eps[:, 0, 0], eps[:, 1, 1]
    gxy = eps[:, 0, 1]
    vm = torch.sqrt(ex ** 2 + ey ** 2 - ex * ey + 3 * gxy ** 2 + 1e-12)
    return vm


def seed_surface_gaussians(scene, gaussians_per_tri=4, albedo=(0.85, 0.45, 0.4)):
    """Seed Gaussians on the surface triangles of the scene.

    Returns a dict of per-Gaussian attributes referencing their host triangle
    + barycentric coords, plus learnable optical params (albedo, roughness).
    """
    tris = scene["surface_tris"]                  # (Nt,3)
    Nt = tris.shape[0]
    # barycentric sample points (small grid within the triangle)
    pts = []
    for i in range(gaussians_per_tri):
        a = (i + 0.5) / gaussians_per_tri
        # spread along the triangle
        pts.append((a * 0.6, (1 - a) * 0.6, max(0.0, 1 - a * 0.6 - (1 - a) * 0.6)))
    bary = torch.tensor(pts, dtype=torch.float64)        # (G_per_tri,3)
    Ng = Nt * gaussians_per_tri
    # host triangle index per gaussian
    host_tri = torch.arange(Nt).repeat_interleave(gaussians_per_tri)   # (Ng,)
    bary_all = bary.repeat(Nt, 1)                                        # (Ng,3)
    return {
        "host_tri": host_tri,        # (Ng,) index into surface_tris
        "bary": bary_all,            # (Ng,3) barycentric weights
        "albedo": torch.full((Ng, 3), 0.0, dtype=torch.float64).fill_(0),
        "roughness": torch.full((Ng,), 0.5, dtype=torch.float64),
        "scale": torch.full((Ng,), 0.05, dtype=torch.float64),
        "opacity": torch.full((Ng,), 0.95, dtype=torch.float64),
    }


def set_albedo(gaussians, albedo_rgb):
    """Uniform albedo assignment (for GT scenes)."""
    a = torch.tensor(albedo_rgb, dtype=torch.float64)
    gaussians["albedo"] = a.unsqueeze(0).expand(gaussians["albedo"].shape[0], 3).clone()


def gaussian_centers(gaussians, scene, u_flat):
    """Compute world-space Gaussian centers from FEM node positions + bary coords.

    Differentiable in u_flat. Returns centers (Ng,3).
    """
    tris = scene["surface_tris"]
    nodes = scene["nodes"]
    deformed = nodes + u_flat.view(scene["Nn"], scene["D"])
    host = gaussians["host_tri"]                       # (Ng,)
    tri_verts = deformed[tris[host]]                   # (Ng,3,3)
    centers = (gaussians["bary"].unsqueeze(-1) * tri_verts).sum(dim=1)   # (Ng,3)
    return centers


def project(centers, c2w, focal=200, H=128, W=128):
    """Project world points to screen (differentiable). c2w: (4,4) cam-to-world.
    Returns uv (N,2) in pixels, depth (N,), valid mask (N,)."""
    w2c = torch.linalg.inv(c2w)
    R = w2c[:3, :3]; t = w2c[:3, 3]
    cam = centers @ R.t() + t.unsqueeze(0)             # (N,3) in cam space
    z = cam[:, 2]
    valid = z > 1e-3
    z_safe = torch.where(valid, z, torch.ones_like(z))
    u = focal * cam[:, 0] / z_safe + W / 2.0
    v = focal * cam[:, 1] / z_safe + H / 2.0
    return torch.stack([u, v], dim=1), z, valid


def project_with_foreground_confidence(
    centers,
    c2w,
    focal=200,
    H=128,
    W=128,
    depth_softness=0.02,
):
    """Project points and estimate same-pixel foreground confidence.

    This is deliberately separate from the legacy ``valid``/visibility mask.
    Each point is assigned the nearest depth at its rounded image pixel and a
    soft confidence ``exp(-(z-z_near)/depth_softness)``.  Thus an in-frame
    point can be geometrically valid while having low foreground confidence.
    """
    if depth_softness <= 0:
        raise ValueError("depth_softness must be positive")
    uv, depth, valid = project(centers, c2w, focal=focal, H=H, W=W)
    in_frame = (
        valid
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < W)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < H)
    )
    pixel_x = uv[:, 0].round().to(torch.long).clamp(0, W - 1)
    pixel_y = uv[:, 1].round().to(torch.long).clamp(0, H - 1)
    pixel_index = pixel_y * W + pixel_x
    z_buffer = torch.full(
        (H * W,), torch.inf, dtype=depth.dtype, device=depth.device
    )
    if hasattr(z_buffer, "scatter_reduce_"):
        z_buffer.scatter_reduce_(
            0,
            pixel_index[in_frame],
            depth[in_frame],
            reduce="amin",
            include_self=True,
        )
    else:  # pragma: no cover - compatibility with old PyTorch
        for index in torch.where(in_frame)[0]:
            pixel = pixel_index[index]
            z_buffer[pixel] = torch.minimum(z_buffer[pixel], depth[index])
    nearest_depth = z_buffer[pixel_index]
    nearest_depth = torch.where(
        in_frame, nearest_depth, torch.zeros_like(nearest_depth)
    )
    depth_delta = (depth - nearest_depth).clamp_min(0)
    confidence = torch.exp(-depth_delta / depth_softness)
    confidence = torch.where(in_frame, confidence, torch.zeros_like(confidence))
    return uv, depth, in_frame, nearest_depth, confidence


def render(gaussians, scene, u_flat, c2w, focal=200, H=128, W=128,
           light_intensity=2.0, ambient=0.25, sss_weight=0.0, brdf="blinnphong"):
    """Differentiable render of the deformed scene from pose c2w.

    Returns image (3,H,W). Co-located light at camera (endoscopy assumption).

    brdf:
      'blinnphong'  (default) — stable Blinn-Phong; the method we report. The
                    roughness->shininess mapping is simple and well-conditioned
                    at our render resolution, which is what makes E/albedo
                    identifiable in the joint inverse problem.
      'cooktorrance'— Cook-Torrance microfacet (GGX). More physical but, at
                    64x64 / 6 frames, the roughness is not identifiable and
                    destabilizes E recovery (documented in the paper).
    sss_weight: subsurface-scattering glow term weight (default 0; the term is
                    implemented but, like SSS coefficients, not identifiable at
                    our scale — see paper Limitations).
    """
    centers = gaussian_centers(gaussians, scene, u_flat)        # (Ng,3)
    uv, depth, valid = project(centers, c2w, focal, H, W)
    tris = scene["surface_tris"]
    nodes = scene["nodes"] + u_flat.view(scene["Nn"], scene["D"])
    host = gaussians["host_tri"]
    v0 = nodes[tris[host, 1]] - nodes[tris[host, 0]]
    v1 = nodes[tris[host, 2]] - nodes[tris[host, 0]]
    normals = torch.cross(v0, v1, dim=-1)                       # (Ng,3)
    nrm = normals.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    normals = normals / nrm
    cam_pos = c2w[:3, 3]
    view_dir = cam_pos.unsqueeze(0) - centers                   # (Ng,3) towards cam
    view_dir = view_dir / view_dir.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    ndotl = (normals * view_dir).sum(dim=-1).clamp(0, 1)        # (Ng,)
    albedo = gaussians["albedo"]                                # (Ng,3) in [0,1]
    rough = gaussians["roughness"].clamp(1e-3, 1.0)             # (Ng,)

    # ---- MECHANO-OPTICAL CROSS-COUPLING (Upgrade 2) ----
    # Compute per-Gaussian von-Mises strain (from host triangle deformation)
    # and apply two clinically-motivated coupling terms:
    #  (a) strain->roughness: stretched tissue microstructure roughens
    #      rough_eff = rough0 + k_strain * eps_vm
    #  (b) stress->perfusion: compression blanches tissue (whitens)
    #      albedo_eff = albedo0 + k_perf * sigma_hydro  (towards white)
    # These make the optical appearance DEPEND on the mechanical state — a
    # genuine multi-field coupling, not just geometric position bridging.
    coupling_k_strain = gaussians.get("coupling_k_strain", torch.tensor(0.0, dtype=albedo.dtype))
    coupling_k_perf = gaussians.get("coupling_k_perf", torch.tensor(0.0, dtype=albedo.dtype))
    if float(coupling_k_strain) != 0.0 or float(coupling_k_perf) != 0.0:
        # per-host-triangle Green-Lagrange strain -> von Mises scalar
        Xe0 = scene["nodes"][tris]                              # (Nt,3,3) rest
        xe0 = nodes[tris]                                       # (Nt,3,3) deformed
        eps_vm = _per_tri_von_mises_strain(xe0, Xe0)            # (Nt,)
        eps_host = eps_vm[host]                                 # (Ng,)
        if float(coupling_k_strain) != 0.0:
            rough = (rough + coupling_k_strain * eps_host).clamp(1e-3, 1.0)
        if float(coupling_k_perf) != 0.0:
            # hydrostatic stress ~ -mean strain (compression -> positive perf shift)
            perf_shift = coupling_k_perf * eps_host.abs().clamp(0, 0.3)
            albedo = (albedo + perf_shift.unsqueeze(-1)).clamp(0, 1)

    if brdf == "cooktorrance":
        ndotv = ndotl
        h = view_dir
        ndoth = (normals * h).sum(dim=-1).clamp(0, 1)
        a2 = rough ** 2
        denom = (ndoth ** 2 * (a2 - 1.0) + 1.0).clamp_min(1e-6)
        D = a2 / (3.14159265 * denom ** 2)
        G = ndotv.clamp(0, 1)
        F = 0.04 + 0.96 * (1.0 - ndotv).clamp(0, 1) ** 5
        spec_brdf = (D * G * F) / (4.0 * ndotv.clamp_min(1e-3) + 1e-6)
        spec = spec_brdf.unsqueeze(-1) * light_intensity
        diffuse = albedo * (light_intensity * ndotl).unsqueeze(-1) / 3.14159265
        shaded = albedo * ambient + diffuse + spec * 0.3
    else:  # blinnphong (reported method)
        spec = ndotl ** (2.0 / (rough ** 2 + 1e-6))
        shaded = albedo * (ambient + light_intensity * ndotl).unsqueeze(-1) \
                 + 0.3 * light_intensity * spec.unsqueeze(-1)

    if sss_weight > 0.0:
        sigma_a = gaussians.get("sigma_a", torch.full((albedo.shape[0], 3), 0.5,
                                                      dtype=albedo.dtype, device=albedo.device))
        sigma_s = gaussians.get("sigma_s", torch.full((albedo.shape[0], 3), 1.0,
                                                      dtype=albedo.dtype, device=albedo.device))
        sss_strength = torch.sigmoid(sigma_s - sigma_a)
        sss_glow = sss_strength * (1.0 - ndotl).clamp(0, 1).unsqueeze(-1) * (1.0 + albedo)
        shaded = shaded + sss_weight * sss_glow
    shaded = shaded.clamp(0, 2.5)

    # ---- VECTORIZED splat (no Python per-Gaussian loop) ----
    # For speed we use depth-sorted over-operator compositing reduced via
    # a single cumulative product over the Gaussian axis (sorted by depth).
    # Build a (Ng, H, W) alpha map, then composite front-to-back with
    # torch.cumprod on the transmittance — fully differentiable, no in-place.
    order = torch.argsort(depth, descending=False)             # near -> far
    scale_px = (focal * gaussians["scale"] / depth.clamp_min(1e-3)).clamp(0.5, 30.0)
    opacity = gaussians["opacity"]
    ux = uv[:, 0]; vx = uv[:, 1]                                # (Ng,)
    s = scale_px                                                # (Ng,)
    # pixel coordinate grids
    ys = torch.arange(H, dtype=centers.dtype, device=centers.device).view(H, 1)
    xs = torch.arange(W, dtype=centers.dtype, device=centers.device).view(1, W)
    # To keep memory bounded we splat each gaussian only into a local window.
    # Use full-grid dense alpha (Ng x H x W) which is fine for Ng<=~600, 64x64.
    dx = xs.unsqueeze(0) - ux.view(-1, 1, 1)                    # (Ng,1,W) broadcast
    dy = ys.unsqueeze(0) - vx.view(-1, 1, 1)                    # (Ng,H,1) broadcast
    d2 = (dx * dx + dy * dy)                                    # (Ng,H,W)
    a = torch.exp(-0.5 * d2 / (s.view(-1, 1, 1) ** 2 + 1e-12)) \
        * opacity.view(-1, 1, 1) * valid.view(-1, 1, 1).to(centers.dtype)   # (Ng,H,W)
    a = a[order]                                                # sort near->far
    shaded_s = shaded[order]                                    # (Ng,3)

    # front-to-back compositing: T_g = prod_{g'<g}(1-a_g'); img += T_g*a_g*color
    one_minus_a = 1.0 - a                                       # (Ng,H,W)
    T = torch.cumprod(one_minus_a, dim=0)                       # transmittance up to g (excl a_g)
    T_prev = torch.cat([torch.ones_like(T[:1]), T[:-1]], dim=0) # T before gaussian g
    w = T_prev * a                                              # (Ng,H,W) weight
    # weighted color sum: sum_g w_g * color_g  -> (H,W,3) then -> (3,H,W)
    img = torch.einsum('ghw,gc->chw', w, shaded_s)              # (3,H,W)
    return img
