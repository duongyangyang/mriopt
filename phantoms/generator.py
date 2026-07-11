"""
Synthetic Brain Phantom Generator
==================================

Generates stylized 2D synthetic brain-like phantoms for educational MRI
simulation purposes. These are NOT real anatomical images — they are
procedurally generated label maps using ellipses, circles, and irregular
perturbed shapes meant to loosely resemble a brain cross-section with
three tissue classes (WM, GM, CSF).

Label convention:
    0 = background
    1 = White Matter (WM)
    2 = Gray Matter (GM)
    3 = Cerebrospinal Fluid (CSF)

Outputs per phantom:
    phantom.png    - RGB visualization of the label map
    label_map.npy  - raw uint8 label array, shape (256, 256)
"""

from __future__ import annotations

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataclasses import dataclass

IMG_SIZE = 256

LABEL_BACKGROUND = 0
LABEL_WM = 1
LABEL_GM = 2
LABEL_CSF = 3

# Colors for visualization (RGB)
LABEL_COLORS = {
    LABEL_BACKGROUND: (0, 0, 0),
    LABEL_WM: (220, 220, 220),
    LABEL_GM: (150, 150, 190),
    LABEL_CSF: (60, 120, 220),
}


@dataclass
class PhantomConfig:
    """Configuration ranges used to randomize phantom generation."""
    size: int = IMG_SIZE
    brain_radius_range: tuple = (90, 110)
    brain_aspect_range: tuple = (0.85, 1.15)
    wm_shrink_range: tuple = (0.55, 0.72)  # WM boundary as fraction of brain radius
    n_ventricles_range: tuple = (2, 4)
    ventricle_radius_range: tuple = (8, 22)
    boundary_noise_amplitude: tuple = (4, 10)
    boundary_noise_freq: tuple = (3, 7)
    n_gm_blobs_range: tuple = (3, 6)


def _radial_perturbation(n_points: int, amplitude: float, freq: int, rng: np.random.Generator) -> np.ndarray:
    """
    Create a smooth periodic perturbation over angles [0, 2*pi) using a
    random sum of low-frequency sinusoids. This produces organic, irregular
    (non-circular) boundaries without needing external noise libraries.
    """
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    perturbation = np.zeros(n_points)
    n_harmonics = rng.integers(2, 5)
    for _ in range(n_harmonics):
        k = rng.integers(1, freq + 1)
        phase = rng.uniform(0, 2 * np.pi)
        amp = amplitude * rng.uniform(0.3, 1.0) / n_harmonics
        perturbation += amp * np.sin(k * theta + phase)
    return perturbation


def _irregular_polygon_mask(
    size: int,
    center: tuple,
    base_radius_x: float,
    base_radius_y: float,
    rotation_deg: float,
    amplitude: float,
    freq: int,
    rng: np.random.Generator,
    n_points: int = 200,
) -> np.ndarray:
    """
    Build a filled binary mask from an ellipse whose radius is perturbed by
    smooth noise as a function of angle, then rotated. This yields an
    irregular, organic blob rather than a perfect ellipse or rectangle.
    """
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    perturb = _radial_perturbation(n_points, amplitude, freq, rng)

    rx = base_radius_x + perturb
    ry = base_radius_y + perturb

    x = rx * np.cos(theta)
    y = ry * np.sin(theta)

    rot = np.deg2rad(rotation_deg)
    x_rot = x * np.cos(rot) - y * np.sin(rot)
    y_rot = x * np.sin(rot) + y * np.cos(rot)

    pts = np.stack([x_rot + center[0], y_rot + center[1]], axis=1).astype(np.int32)

    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def generate_phantom(seed: int, size: int = IMG_SIZE, config: PhantomConfig = None) -> np.ndarray:
    """
    Generate a single synthetic brain phantom label map.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility. Same seed -> same phantom.
    size : int
        Output image size (size x size).
    config : PhantomConfig, optional
        Randomization ranges. Uses defaults if not provided.

    Returns
    -------
    label_map : np.ndarray, shape (size, size), dtype uint8
        Values: 0=background, 1=WM, 2=GM, 3=CSF
    """
    if config is None:
        config = PhantomConfig(size=size)

    rng = np.random.default_rng(seed)

    center = (size // 2 + rng.integers(-5, 6), size // 2 + rng.integers(-5, 6))
    brain_radius = rng.uniform(*config.brain_radius_range)
    aspect = rng.uniform(*config.brain_aspect_range)
    rotation = rng.uniform(0, 180)
    amp = rng.uniform(*config.boundary_noise_amplitude)
    freq = rng.integers(*config.boundary_noise_freq)

    # Outer brain contour -> everything inside is WM initially
    brain_mask = _irregular_polygon_mask(
        size, center, brain_radius, brain_radius * aspect, rotation, amp, freq, rng
    )

    label_map = np.zeros((size, size), dtype=np.uint8)
    label_map[brain_mask] = LABEL_WM

    # Gray matter: a thick irregular ring near the outer boundary of the brain,
    # built by shrinking the brain contour and taking the WM-minus-shrunk region.
    gm_shrink = rng.uniform(0.78, 0.90)
    gm_amp = amp * rng.uniform(0.6, 1.0)
    gm_freq = freq
    gm_inner_mask = _irregular_polygon_mask(
        size, center, brain_radius * gm_shrink, brain_radius * aspect * gm_shrink,
        rotation, gm_amp, gm_freq, rng
    )
    gm_ring = brain_mask & ~gm_inner_mask
    label_map[gm_ring] = LABEL_GM

    # Additional scattered GM blobs near the cortical ring for organic look
    n_gm_blobs = rng.integers(*config.n_gm_blobs_range)
    for _ in range(n_gm_blobs):
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(0.6, 0.95) * brain_radius
        bx = int(center[0] + dist * np.cos(angle))
        by = int(center[1] + dist * np.sin(angle))
        if not (0 <= bx < size and 0 <= by < size):
            continue
        blob_r = rng.uniform(6, 14)
        blob_mask = _irregular_polygon_mask(
            size, (bx, by), blob_r, blob_r * rng.uniform(0.7, 1.3),
            rng.uniform(0, 180), blob_r * 0.3, rng.integers(2, 5), rng
        )
        blob_mask &= brain_mask
        label_map[blob_mask] = LABEL_GM

    # WM interior (inside the shrunk inner contour, excluding ventricles later)
    wm_shrink = rng.uniform(*config.wm_shrink_range)
    wm_amp = amp * rng.uniform(0.5, 0.9)
    interior_mask = _irregular_polygon_mask(
        size, center, brain_radius * wm_shrink, brain_radius * aspect * wm_shrink,
        rotation, wm_amp, freq, rng
    )
    interior_mask &= brain_mask
    label_map[interior_mask] = LABEL_WM

    # CSF ventricles: small irregular blobs near the center
    n_ventricles = rng.integers(*config.n_ventricles_range)
    for _ in range(n_ventricles):
        offset_angle = rng.uniform(0, 2 * np.pi)
        offset_dist = rng.uniform(0, 0.25) * brain_radius
        vx = int(center[0] + offset_dist * np.cos(offset_angle))
        vy = int(center[1] + offset_dist * np.sin(offset_angle))
        v_r = rng.uniform(*config.ventricle_radius_range)
        vent_mask = _irregular_polygon_mask(
            size, (vx, vy), v_r, v_r * rng.uniform(0.5, 1.4),
            rng.uniform(0, 180), v_r * 0.3, rng.integers(2, 5), rng
        )
        vent_mask &= brain_mask
        label_map[vent_mask] = LABEL_CSF

    # Also carve a thin central CSF sulcus-like line for extra realism (optional, subtle)
    if rng.uniform() < 0.5:
        line_len = brain_radius * rng.uniform(0.4, 0.7)
        line_angle = rng.uniform(0, np.pi)
        x0 = int(center[0] - line_len / 2 * np.cos(line_angle))
        y0 = int(center[1] - line_len / 2 * np.sin(line_angle))
        x1 = int(center[0] + line_len / 2 * np.cos(line_angle))
        y1 = int(center[1] + line_len / 2 * np.sin(line_angle))
        line_canvas = np.zeros((size, size), dtype=np.uint8)
        cv2.line(line_canvas, (x0, y0), (x1, y1), 1, thickness=rng.integers(2, 4))
        line_mask = line_canvas.astype(bool) & brain_mask
        label_map[line_mask] = LABEL_CSF

    return label_map


def label_map_to_rgb(label_map: np.ndarray) -> np.ndarray:
    """Convert a label map to an RGB image for visualization/saving as PNG."""
    rgb = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for label, color in LABEL_COLORS.items():
        rgb[label_map == label] = color
    return rgb


def save_phantom(label_map: np.ndarray, out_dir: str) -> None:
    """Save phantom.png (visualization) and label_map.npy (raw labels) to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    rgb = label_map_to_rgb(label_map)
    # cv2 expects BGR
    cv2.imwrite(os.path.join(out_dir, "phantom.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(os.path.join(out_dir, "label_map.npy"), label_map)


def generate_batch(n: int, out_root: str, base_seed: int = 0, config: PhantomConfig = None) -> list:
    """
    Generate n unique phantoms, each saved into out_root/phantom_XXXX/.

    Returns list of output directories.
    """
    out_dirs = []
    for i in range(n):
        seed = base_seed + i
        label_map = generate_phantom(seed=seed, config=config)
        out_dir = os.path.join(out_root, f"phantom_{i:04d}")
        save_phantom(label_map, out_dir)
        out_dirs.append(out_dir)
    return out_dirs


if __name__ == "__main__":
    out_dirs = generate_batch(n=100, out_root="/home/claude/mri_project/data/phantoms", base_seed=42)
    print(f"Generated {len(out_dirs)} phantoms into data/phantoms/")
