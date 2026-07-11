"""
Phase 8 Evaluation: CNN-recommended (TR*, TE*) vs Ground-Truth (Anchor-based model)
=======================================================================================
Dùng model mới (có TR0,TE0,J0 làm anchor tường minh) để đánh giá lại xem
%J lost có cải thiện so với thiết kế cũ (không anchor) hay không.

Chạy:
    PYTHONPATH=. python3 optimization/evaluate_recommendations.py \
        --test_phantoms 90,91,92,93,94,95,96,97,98,99 \
        --tr0 200 --te0 10
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from simulator.mri_simulator import simulate_mri, get_phantom_tissue_params
from metrics.cnr import cnr_wm_gm, estimate_noise_std
from optimization.recommend_params import load_model, load_image, recommend, TR_CANDIDATES, TE_CANDIDATES, _default_device

LAMBDA = 1e-5
N_PE = 128
GAUSSIAN_NOISE_STD = 0.02
PHANTOM_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "phantoms")

TR_GRID_FULL = np.arange(200, 4000 + 1, 100)
TE_GRID_FULL = np.arange(10, 200 + 1, 5)


def scan_time(tr, n_pe=N_PE):
    return tr * n_pe


def lookup_true_j(true_df: pd.DataFrame, tr: float, te: float) -> float:
    """Tra cứu J thật gần nhất trong true_df cho (TR, TE) đề xuất."""
    sub = true_df.copy()
    sub["dist"] = (sub["TR"] - tr).abs() + (sub["TE"] - te).abs() * 10
    return float(sub.loc[sub["dist"].idxmin(), "J"])


def compute_true_grid(phantom_idx: int) -> pd.DataFrame:
    """Tính J thật cho toàn bộ lưới TR/TE của 1 phantom, dùng simulator thật (ground truth)."""
    label_map = np.load(os.path.join(PHANTOM_ROOT, f"phantom_{phantom_idx:04d}", "label_map.npy"))
    tissue_params = get_phantom_tissue_params(phantom_seed=phantom_idx, std_frac=0.20)
    records = []
    for tr in TR_GRID_FULL:
        for te in TE_GRID_FULL:
            seed = phantom_idx * 100000 + int(tr) * 100 + int(te)
            image = simulate_mri(label_map, tr, te, gaussian_noise_std=GAUSSIAN_NOISE_STD,
                                  seed=seed, tissue_params=tissue_params)
            noise_std = estimate_noise_std(image, label_map)
            cnr = cnr_wm_gm(image, label_map, noise_std=noise_std)
            j = cnr - LAMBDA * scan_time(tr)
            records.append({"TR": tr, "TE": te, "J": j})
    return pd.DataFrame(records), label_map, tissue_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="training/checkpoints/best_model.pt")
    parser.add_argument("--test_phantoms", default="90,91,92,93,94,95,96,97,98,99")
    parser.add_argument("--tr0", type=float, default=200.0, help="TR của ảnh anchor 'đã chụp'")
    parser.add_argument("--te0", type=float, default=10.0, help="TE của ảnh anchor 'đã chụp'")
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--out_dir", default="optimization/evaluation")
    parser.add_argument("--top_k", type=int, default=2, help="Số ứng viên đa-dạng/top-K đề xuất")
    parser.add_argument("--min_dist_frac", type=float, default=0.3,
                         help="Khoảng cách chuẩn hóa tối thiểu giữa các ứng viên top-K (cho selection=greedy)")
    parser.add_argument("--selection", default="te_stratified", choices=["greedy", "te_stratified"],
                         help="greedy: tách theo (TR,TE); te_stratified: ép mỗi bin TE có 1 ứng viên (bắt chế độ T2)")
    parser.add_argument("--te_split", default=None,
                         help="Ngưỡng TE chia bin (vd '15,60'). Mặc định: tự chọn theo top_k.")
    args = parser.parse_args()

    te_split = None
    if args.te_split:
        te_split = [float(x) for x in args.te_split.split(",")]

    os.makedirs(args.out_dir, exist_ok=True)
    test_phantoms = [int(x) for x in args.test_phantoms.split(",")]

    model, j_mean, j_std = load_model(args.checkpoint, args.device)
    if j_mean is None:
        raise ValueError("Checkpoint không chứa j_mean/j_std -- cần train lại bằng training/train.py mới")

    records = []
    for p_idx in test_phantoms:
        true_df, label_map, tissue_params = compute_true_grid(p_idx)
        true_best = true_df.loc[true_df["J"].idxmax()]
        j_true_max = float(true_best["J"])

        # tính J0 thật tại điểm anchor (TR0, TE0) -- giả định biết được (đã segment ảnh anchor)
        seed0 = p_idx * 100000 + int(args.tr0) * 100 + int(args.te0)
        image0 = simulate_mri(label_map, args.tr0, args.te0, gaussian_noise_std=GAUSSIAN_NOISE_STD,
                               seed=seed0, tissue_params=tissue_params)
        noise_std0 = estimate_noise_std(image0, label_map)
        j0_true = cnr_wm_gm(image0, label_map, noise_std=noise_std0) - LAMBDA * scan_time(args.tr0)

        # lưu ảnh anchor tạm để load lại qua load_image (giữ nhất quán pipeline)
        from PIL import Image
        tmp_img_path = os.path.join(args.out_dir, f"_tmp_anchor_p{p_idx}.png")
        img_norm = np.clip(image0 / (image0.max() + 1e-8) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(img_norm).save(tmp_img_path)
        image_tensor = load_image(tmp_img_path)

        pred = recommend(model, image_tensor, args.device, args.tr0, args.te0, j0_true, j_mean, j_std,
                         top_k=args.top_k, min_dist_frac=args.min_dist_frac,
                         selection=args.selection, te_split=te_split)
        cands = pred["candidates"]

        # J thật + J_lost cho từng ứng viên; chọn best-of-K (oracle) theo J thật
        cand_rows = []
        for c in cands:
            j_true_c = lookup_true_j(true_df, c["TR"], c["TE"])
            j_lost_c = 100.0 * (j_true_max - j_true_c) / (abs(j_true_max) + 1e-8)
            cand_rows.append({
                "TR": c["TR"], "TE": c["TE"], "J_pred": c["J_pred"],
                "J_true": j_true_c, "pct_J_lost": j_lost_c,
            })
        top1 = cand_rows[0]
        best = max(cand_rows, key=lambda r: r["J_true"])

        records.append({
            "phantom": p_idx,
            "TR_true": float(true_best["TR"]), "TE_true": float(true_best["TE"]), "J_true_max": j_true_max,
            "TR_pred_top1": top1["TR"], "TE_pred_top1": top1["TE"],
            "J_lost_top1": top1["pct_J_lost"],
            "TR_pred_best": best["TR"], "TE_pred_best": best["TE"],
            "J_lost_best_of_k": best["pct_J_lost"],
            "n_candidates": len(cand_rows),
            "selection_mode": args.selection,
            "J0_anchor": j0_true,
            "candidates": json.dumps(cand_rows),
        })
        cand_str = " | ".join(f"({c['TR']:.0f},{c['TE']:.0f})Jl={c['pct_J_lost']:.1f}%" for c in cand_rows)
        print(f"phantom {p_idx}: true=({true_best['TR']:.0f},{true_best['TE']:.0f}) "
              f"top1=({top1['TR']:.0f},{top1['TE']:.0f}) Jl@top1={top1['pct_J_lost']:.1f}% "
              f"best=({best['TR']:.0f},{best['TE']:.0f}) Jl@bestK={best['pct_J_lost']:.1f}% "
              f"[{cand_str}]")

        os.remove(tmp_img_path)

    result_df = pd.DataFrame(records)
    result_df.to_csv(os.path.join(args.out_dir, "recommendation_evaluation.csv"), index=False)

    print(f"\n=== Summary over held-out phantoms (selection={args.selection}, top_k={args.top_k}) ===")
    print(f"TR* abs error top1 (ms): mean={abs(result_df['TR_true']-result_df['TR_pred_top1']).mean():.1f}, "
          f"median={abs(result_df['TR_true']-result_df['TR_pred_top1']).median():.1f}")
    print(f"%% J lost @top1:        mean={result_df['J_lost_top1'].mean():.1f}%%, "
          f"median={result_df['J_lost_top1'].median():.1f}%%, max={result_df['J_lost_top1'].max():.1f}%%")
    print(f"%% J lost @best-of-K:   mean={result_df['J_lost_best_of_k'].mean():.1f}%%, "
          f"median={result_df['J_lost_best_of_k'].median():.1f}%%, max={result_df['J_lost_best_of_k'].max():.1f}%%")
    improved = (result_df['J_lost_best_of_k'] < result_df['J_lost_top1']).sum()
    print(f"best-of-K cải thiện so với top1 ở {improved}/{len(result_df)} phantom")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].scatter(result_df["TR_true"], result_df["TR_pred_top1"], color="tab:blue",
                    label="top-1", alpha=0.7)
    axes[0].scatter(result_df["TR_true"], result_df["TR_pred_best"], color="tab:green",
                    marker="D", label="best-of-K", alpha=0.7)
    lims = [TR_CANDIDATES.min(), TR_CANDIDATES.max()]
    axes[0].plot(lims, lims, "k--", alpha=0.5, label="y = x (perfect)")
    axes[0].set_xlabel("True TR* (ms)")
    axes[0].set_ylabel("CNN-recommended TR* (ms)")
    axes[0].legend(fontsize=8)

    axes[1].scatter(result_df["TE_true"], result_df["TE_pred_top1"], color="tab:blue",
                    label="top-1", alpha=0.7)
    axes[1].scatter(result_df["TE_true"], result_df["TE_pred_best"], color="tab:green",
                    marker="D", label="best-of-K", alpha=0.7)
    lims_te = [TE_CANDIDATES.min(), TE_CANDIDATES.max()]
    axes[1].plot(lims_te, lims_te, "k--", alpha=0.5, label="y = x (perfect)")
    axes[1].set_xlabel("True TE* (ms)")
    axes[1].set_ylabel("CNN-recommended TE* (ms)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "true_vs_recommended_scatter.png"), dpi=130)
    plt.close(fig)

    print(f"\nSaved: {args.out_dir}/recommendation_evaluation.csv, true_vs_recommended_scatter.png")


if __name__ == "__main__":
    main()
