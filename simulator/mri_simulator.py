"""
Spin-Echo MRI Simulator
========================

Simulates MRI signal intensity images from tissue label maps using the
classic spin-echo signal equation:

    S = PD * (1 - exp(-TR / T1)) * exp(-TE / T2)

Tissue parameters (educational / illustrative values, not clinical):

    White Matter (WM): T1=850ms,  T2=80ms,   PD=0.70
    Gray Matter  (GM): T1=1300ms, T2=100ms,  PD=0.85
    CSF:               T1=4000ms, T2=2000ms, PD=1.00
"""

from __future__ import annotations

import numpy as np

from phantoms.generator import LABEL_BACKGROUND, LABEL_WM, LABEL_GM, LABEL_CSF

TISSUE_PARAMS = {
    LABEL_BACKGROUND: {"T1": 1.0, "T2": 1.0, "PD": 0.0},  # no signal
    LABEL_WM: {"T1": 850.0, "T2": 80.0, "PD": 0.70},
    LABEL_GM: {"T1": 1300.0, "T2": 100.0, "PD": 0.85},
    LABEL_CSF: {"T1": 4000.0, "T2": 2000.0, "PD": 1.00},
}


def get_phantom_tissue_params(phantom_seed: int, std_frac: float = 0.05) -> dict:
    """
    Per-phantom tissue parameters: each phantom gets its own (T1, T2, PD)
    per tissue class, sampled around the base TISSUE_PARAMS with std_frac
    relative std (default 5%). This models realistic subject-to-subject
    tissue variability instead of a single global constant signal per
    tissue class, so mean(WM)/mean(GM) genuinely differ across phantoms.
    """
    rng = np.random.default_rng(phantom_seed)
    params = {}
    for label, base in TISSUE_PARAMS.items():
        if base["PD"] == 0.0:
            params[label] = dict(base)  # background stays zero-signal
            continue
        params[label] = {
            "T1": base["T1"] * (1 + rng.normal(0, std_frac)),
            "T2": base["T2"] * (1 + rng.normal(0, std_frac)),
            "PD": base["PD"] * (1 + rng.normal(0, std_frac)),
        }
    return params


def spin_echo_signal(pd: float, t1: float, t2: float, tr: float, te: float) -> float:
    """Spin-echo signal equation: S = PD * (1 - exp(-TR/T1)) * exp(-TE/T2)."""
    return pd * (1.0 - np.exp(-tr / t1)) * np.exp(-te / t2)


def simulate_mri(
    label_map: np.ndarray,
    tr: float,
    te: float,
    gaussian_noise_std: float = 0.0,
    rician_noise_std: float = 0.0,
    seed: int | None = None,
    tissue_params: dict | None = None,
) -> np.ndarray:
    """
    Simulate an MRI image from a tissue label map at given TR/TE.

    tissue_params : dict, optional
        Override TISSUE_PARAMS (e.g. per-phantom variability from
        get_phantom_tissue_params). Defaults to the global TISSUE_PARAMS.
    """
    if tissue_params is None:
        tissue_params = TISSUE_PARAMS
    image = np.zeros(label_map.shape, dtype=np.float64)
    for label, params in tissue_params.items():
        mask = label_map == label
        if not mask.any():
            continue
        s = spin_echo_signal(params["PD"], params["T1"], params["T2"], tr, te)
        image[mask] = s

    rng = np.random.default_rng(seed)

    if gaussian_noise_std > 0:
        image = image + rng.normal(0.0, gaussian_noise_std, size=image.shape)

    if rician_noise_std > 0:
        # Rician noise: sqrt((S + n1)^2 + n2^2), n1,n2 ~ N(0, sigma)
        n1 = rng.normal(0.0, rician_noise_std, size=image.shape)
        n2 = rng.normal(0.0, rician_noise_std, size=image.shape)
        image = np.sqrt((image + n1) ** 2 + n2 ** 2)

    return image


def batch_simulate(
    label_map: np.ndarray,
    tr_values: np.ndarray,
    te_values: np.ndarray,
    gaussian_noise_std: float = 0.0,
    rician_noise_std: float = 0.0,
    seed: int | None = None,
) -> dict:
    """
    Simulate MRI images for a full grid of (TR, TE) combinations.

    Returns
    -------
    dict mapping (tr, te) -> image (np.ndarray)
    """
    results = {}
    for tr in tr_values:
        for te in te_values:
            results[(float(tr), float(te))] = simulate_mri(
                label_map, tr, te,
                gaussian_noise_std=gaussian_noise_std,
                rician_noise_std=rician_noise_std,
                seed=seed,
            )
    return results
