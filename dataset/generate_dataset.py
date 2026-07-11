"""
Dataset Generation (Anchor-based design)
============================================
Mỗi mẫu: (ảnh chụp ở TR0/TE0, TR0, TE0, J0, TR_c, TE_c) -> label = J_c

Chạy: PYTHONPATH=. python3 dataset/generate_dataset.py
"""
import os, sys
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from simulator.mri_simulator import simulate_mri, get_phantom_tissue_params
from metrics.cnr import cnr_wm_gm, estimate_noise_std

N_PHANTOMS = 100
LAMBDA = 1e-5
N_PE = 128
GAUSSIAN_NOISE_STD = 0.02
PHANTOM_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "phantoms")
OUT_ROOT = os.path.dirname(__file__)
IMG_DIR = os.path.join(OUT_ROOT, "images")

TR_VALUES = np.arange(200, 4000 + 1, 100)   # lưới đầy đủ để chọn ngẫu nhiên từ đó
TE_VALUES = np.arange(10, 200 + 1, 5)

N_ANCHORS_PER_PHANTOM = 15     # số điểm neo (ảnh) mỗi phantom
N_CANDIDATES_PER_ANCHOR = 30   # số candidate mỗi anchor
CONSISTENCY_FRAC = 0.1         # tỷ lệ mẫu "trùng" (candidate = anchor) để ép model học tính nhất quán
RNG_SEED = 123


def scan_time(tr, n_pe=N_PE):
    return tr * n_pe


def compute_J(image, label_map, tr):
    noise_std = estimate_noise_std(image, label_map)
    cnr = cnr_wm_gm(image, label_map, noise_std=noise_std)
    return cnr - LAMBDA * scan_time(tr)


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    records = []

    for p_idx in range(N_PHANTOMS):
        label_map = np.load(os.path.join(PHANTOM_ROOT, f"phantom_{p_idx:04d}", "label_map.npy"))
        tissue_params = get_phantom_tissue_params(phantom_seed=p_idx, std_frac=0.20)

        # chọn ngẫu nhiên các điểm anchor cho phantom này
        anchor_trs = rng.choice(TR_VALUES, size=N_ANCHORS_PER_PHANTOM, replace=False)
        anchor_tes = rng.choice(TE_VALUES, size=N_ANCHORS_PER_PHANTOM, replace=False)

        for a_idx in range(N_ANCHORS_PER_PHANTOM):
            tr0, te0 = float(anchor_trs[a_idx]), float(anchor_tes[a_idx])

            seed0 = p_idx * 100000 + int(tr0) * 100 + int(te0)
            image0 = simulate_mri(label_map, tr0, te0, gaussian_noise_std=GAUSSIAN_NOISE_STD,
                                   seed=seed0, tissue_params=tissue_params)
            j0 = compute_J(image0, label_map, tr0)

            img_name = f"p{p_idx:04d}_anchor_tr{int(tr0):04d}_te{int(te0):03d}.png"
            img_norm = np.clip(image0 / (image0.max() + 1e-8) * 255, 0, 255).astype(np.uint8)
            Image.fromarray(img_norm).save(os.path.join(IMG_DIR, img_name))

            # sample candidate ngẫu nhiên (khác anchor)
            cand_trs = rng.choice(TR_VALUES, size=N_CANDIDATES_PER_ANCHOR, replace=True)
            cand_tes = rng.choice(TE_VALUES, size=N_CANDIDATES_PER_ANCHOR, replace=True)

            for c_idx in range(N_CANDIDATES_PER_ANCHOR):
                tr_c, te_c = float(cand_trs[c_idx]), float(cand_tes[c_idx])
                seed_c = p_idx * 100000 + int(tr_c) * 100 + int(te_c) + 7
                image_c = simulate_mri(label_map, tr_c, te_c, gaussian_noise_std=GAUSSIAN_NOISE_STD,
                                        seed=seed_c, tissue_params=tissue_params)
                j_c = compute_J(image_c, label_map, tr_c)

                records.append({
                    "phantom": p_idx, "image_path": f"images/{img_name}",
                    "TR0": tr0, "TE0": te0, "J0": j0,
                    "TRc": tr_c, "TEc": te_c, "Jc": j_c,
                })

            # thêm mẫu "trùng" (candidate = anchor) để ép tính nhất quán
            if rng.uniform() < CONSISTENCY_FRAC:
                records.append({
                    "phantom": p_idx, "image_path": f"images/{img_name}",
                    "TR0": tr0, "TE0": te0, "J0": j0,
                    "TRc": tr0, "TEc": te0, "Jc": j0,
                })

        print(f"phantom {p_idx:04d} done ({N_ANCHORS_PER_PHANTOM} anchors)")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUT_ROOT, "metadata.csv"), index=False)
    print(f"\nDone. {len(df)} rows -> dataset/metadata.csv, "
          f"{N_PHANTOMS * N_ANCHORS_PER_PHANTOM} ảnh -> dataset/images/")


if __name__ == "__main__":
    main()
