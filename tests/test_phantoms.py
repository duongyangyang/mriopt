"""Unit tests for Phase 1: phantom generation."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phantoms.generator import (
    generate_phantom,
    generate_batch,
    label_map_to_rgb,
    save_phantom,
    LABEL_BACKGROUND,
    LABEL_WM,
    LABEL_GM,
    LABEL_CSF,
    IMG_SIZE,
)


def test_shape_and_dtype():
    lm = generate_phantom(seed=0)
    assert lm.shape == (IMG_SIZE, IMG_SIZE)
    assert lm.dtype == np.uint8


def test_valid_labels_only():
    lm = generate_phantom(seed=1)
    valid = {LABEL_BACKGROUND, LABEL_WM, LABEL_GM, LABEL_CSF}
    assert set(np.unique(lm)).issubset(valid)


def test_all_tissue_classes_present():
    # With default config, a phantom should contain all three tissue types
    lm = generate_phantom(seed=2)
    present = set(np.unique(lm))
    assert LABEL_WM in present
    assert LABEL_GM in present
    assert LABEL_CSF in present


def test_reproducibility_same_seed():
    lm1 = generate_phantom(seed=123)
    lm2 = generate_phantom(seed=123)
    assert np.array_equal(lm1, lm2)


def test_uniqueness_different_seeds():
    lm1 = generate_phantom(seed=10)
    lm2 = generate_phantom(seed=11)
    assert not np.array_equal(lm1, lm2)


def test_not_rectangular():
    # A simple sanity check that tissue masks aren't axis-aligned rectangles:
    # for a rectangle, every row within the bounding box would have identical
    # left/right extents. We check that row-wise extents vary (irregular boundary).
    lm = generate_phantom(seed=5)
    wm_mask = lm == LABEL_WM
    rows_with_wm = np.where(wm_mask.any(axis=1))[0]
    widths = []
    for r in rows_with_wm:
        cols = np.where(wm_mask[r])[0]
        widths.append(cols.max() - cols.min())
    widths = np.array(widths)
    # A rectangle would have near-constant width across all rows; real organic
    # shapes should show meaningful variation.
    assert widths.std() > 3


def test_label_map_to_rgb_shape():
    lm = generate_phantom(seed=3)
    rgb = label_map_to_rgb(lm)
    assert rgb.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert rgb.dtype == np.uint8


def test_save_and_reload(tmp_path):
    lm = generate_phantom(seed=4)
    out_dir = tmp_path / "phantom_test"
    save_phantom(lm, str(out_dir))
    assert (out_dir / "phantom.png").exists()
    assert (out_dir / "label_map.npy").exists()
    loaded = np.load(out_dir / "label_map.npy")
    assert np.array_equal(loaded, lm)


def test_generate_batch_uniqueness(tmp_path):
    out_dirs = generate_batch(n=5, out_root=str(tmp_path), base_seed=100)
    assert len(out_dirs) == 5
    maps = [np.load(os.path.join(d, "label_map.npy")) for d in out_dirs]
    # Check all pairwise different
    for i in range(len(maps)):
        for j in range(i + 1, len(maps)):
            assert not np.array_equal(maps[i], maps[j])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
