from __future__ import annotations

import numpy as np

from experiments.export_ion_ct_figure_assets import (
    crop_to_mask,
    overlay_mask,
    representative_slice,
    window_to_uint8,
)


def test_window_to_uint8_clips_declared_lung_window() -> None:
    image = np.asarray([[-1200.0, -1000.0, -400.0, 200.0, 800.0]])
    output = window_to_uint8(image)
    assert output.dtype == np.uint8
    assert output[0, 0] == output[0, 1] == 0
    assert output[0, -1] == output[0, -2] == 255
    assert 0 < output[0, 2] < 255


def test_representative_slice_and_crop_remove_empty_border() -> None:
    volume = np.zeros((3, 40, 50), dtype=np.float32)
    mask = np.zeros_like(volume, dtype=bool)
    mask[1, 15:25, 18:32] = True
    mask[2, 18:22, 20:25] = True
    assert representative_slice(volume, mask) == 1
    cropped, cropped_mask = crop_to_mask(volume[1], mask[1], margin_fraction=0.0)
    assert cropped.shape[0] < volume.shape[1]
    assert cropped.shape[1] < volume.shape[2]
    assert cropped_mask.any()


def test_overlay_changes_only_masked_pixels() -> None:
    image = np.full((4, 4), 100, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    overlay = overlay_mask(image, mask)
    assert overlay.shape == (4, 4, 3)
    assert np.array_equal(overlay[0, 0], [100, 100, 100])
    assert not np.array_equal(overlay[1, 1], [100, 100, 100])
