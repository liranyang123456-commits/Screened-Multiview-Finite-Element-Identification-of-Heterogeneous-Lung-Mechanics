"""Smoke tests for the CT-conditioned lung benchmark primitives."""
from __future__ import annotations

import numpy as np

from lung_inverse_rendering.ct_geometry import make_synthetic_ct_surrogate
from lung_inverse_rendering.ct_loader import lung_air_mask, mask_to_surface


def test_synthetic_ct_surrogate_has_renderable_fem_scene() -> None:
    scene = make_synthetic_ct_surrogate("lung_geometry_test", seed=9, E_true=5e3)
    assert scene["geometry_source"] == "synthetic_ct_surrogate"
    assert scene["nodes"].shape[1] == 3
    assert scene["elems"].shape[1] == 4
    assert len(scene["surface_tris"]) > 0
    assert len(scene["fixed"]) > 0


def test_hu_mask_to_surface_without_patient_data() -> None:
    volume = np.full((24, 32, 32), -100.0, dtype=np.float32)
    z, y, x = np.ogrid[:24, :32, :32]
    left = (x - 10) ** 2 + (y - 16) ** 2 + (z - 12) ** 2 < 7**2
    right = (x - 22) ** 2 + (y - 16) ** 2 + (z - 12) ** 2 < 7**2
    volume[left | right] = -800.0
    vertices, faces = mask_to_surface(lung_air_mask(volume), (1.0, 1.0, 1.0))
    assert len(vertices) > 20
    assert len(faces) > 20


def test_hu_mask_excludes_border_connected_exterior_air() -> None:
    volume = np.full((12, 32, 32), -800.0, dtype=np.float32)
    volume[:, 5:-5, 5:-5] = -100.0
    volume[:, 10:22, 11:21] = -800.0
    mask = lung_air_mask(volume)
    assert not mask[:, 0, :].any()
    assert not mask[:, :, 0].any()
    assert mask[:, 12:20, 13:19].any()


def test_hu_mask_excludes_posterior_table_air_component() -> None:
    volume = np.full((12, 48, 48), -100.0, dtype=np.float32)
    volume[:, 10:34, 8:21] = -800.0
    volume[:, 10:34, 27:40] = -800.0
    volume[:, 42:46, 5:43] = -800.0
    mask = lung_air_mask(volume)
    assert mask[:, 12:30, 10:19].any()
    assert mask[:, 42:46, :].sum() == 0
