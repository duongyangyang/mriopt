"""
Contrast-to-Noise Ratio (CNR) Metrics
=======================================

CNR = |mean(A) - mean(B)| / noise_std

Tissue means are computed from ground-truth label masks (WM, GM, CSF).
Noise std is estimated from the background region of the simulated image
(background has zero true signal, so its std reflects the noise floor).
If the background is empty or the image is noiseless, a small epsilon
is used to avoid division by zero, and callers may supply their own
noise_std (e.g., a known injected-noise level) instead.
"""

from __future__ import annotations

import numpy as np

from phantoms.generator import LABEL_BACKGROUND, LABEL_WM, LABEL_GM, LABEL_CSF

EPS = 1e-8


def tissue_mean(image: np.ndarray, label_map: np.ndarray, label: int) -> float:
    """Mean signal intensity within a tissue mask."""
    mask = label_map == label
    if not mask.any():
        return 0.0
    return float(image[mask].mean())


def estimate_noise_std(image: np.ndarray, label_map: np.ndarray) -> float:
    """
    Estimate noise standard deviation from the background region.
    Falls back to a tiny epsilon if background is absent or the
    image is perfectly noiseless (std == 0), to avoid div-by-zero.
    """
    bg_mask = label_map == LABEL_BACKGROUND
    if not bg_mask.any():
        return EPS
    std = float(image[bg_mask].std())
    return std if std > EPS else EPS


def compute_cnr(image: np.ndarray, label_map: np.ndarray, label_a: int, label_b: int,
                 noise_std: float | None = None) -> float:
    """
    CNR between two tissue classes.

    If noise_std is not provided, it is estimated from the background
    region of the image.
    """
    mean_a = tissue_mean(image, label_map, label_a)
    mean_b = tissue_mean(image, label_map, label_b)
    if noise_std is None:
        noise_std = estimate_noise_std(image, label_map)
    return abs(mean_a - mean_b) / noise_std


def cnr_wm_gm(image: np.ndarray, label_map: np.ndarray, noise_std: float | None = None) -> float:
    return compute_cnr(image, label_map, LABEL_WM, LABEL_GM, noise_std)


def cnr_gm_csf(image: np.ndarray, label_map: np.ndarray, noise_std: float | None = None) -> float:
    return compute_cnr(image, label_map, LABEL_GM, LABEL_CSF, noise_std)


def cnr_wm_csf(image: np.ndarray, label_map: np.ndarray, noise_std: float | None = None) -> float:
    return compute_cnr(image, label_map, LABEL_WM, LABEL_CSF, noise_std)


def all_cnr_metrics(image: np.ndarray, label_map: np.ndarray, noise_std: float | None = None) -> dict:
    """Compute all three pairwise CNR metrics in one call."""
    if noise_std is None:
        noise_std = estimate_noise_std(image, label_map)
    return {
        "CNR_WM_GM": cnr_wm_gm(image, label_map, noise_std),
        "CNR_GM_CSF": cnr_gm_csf(image, label_map, noise_std),
        "CNR_WM_CSF": cnr_wm_csf(image, label_map, noise_std),
    }
