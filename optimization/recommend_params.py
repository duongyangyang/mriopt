"""
Phase 8: Parameter Recommendation (Anchor-based model)
==========================================================
f(image, TR0, TE0, J0, TRc, TEc) -> Jc_predicted

Chạy:
    PYTHONPATH=. python3 optimization/recommend_params.py \
        --image dataset/images/p0000_anchor_tr0200_te010.png \
        --tr0 200 --te0 10 --j0 1.23
"""
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.cnn_model import MRIParamNet
from training.dataset import TR_MIN, TR_MAX, TE_MIN, TE_MAX

TR_CANDIDATES = np.arange(200, 4000 + 1, 50)
TE_CANDIDATES = np.arange(10, 200 + 1, 5)


def _default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(checkpoint_path: str, device: str):
    model = MRIParamNet(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        j_mean = checkpoint.get("j_mean")
        j_std = checkpoint.get("j_std")
    else:
        state_dict = checkpoint
        j_mean, j_std = None, None
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, j_mean, j_std


def load_image(image_path: str) -> torch.Tensor:
    img = np.array(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0)  # (1, H, W)


def recommend(model, image_tensor: torch.Tensor, device: str,
              tr0: float, te0: float, j0: float, j_mean: float, j_std: float,
              tr_candidates=TR_CANDIDATES, te_candidates=TE_CANDIDATES,
              batch_size: int = 512, top_k: int = 1, min_dist_frac: float = 0.3,
              selection: str = "greedy", te_split=None):
    """
    Quét toàn bộ lưới candidate (TRc, TEc), giữ cố định (image, TR0, TE0, J0),
    dự đoán Jc cho từng candidate. Trả về:
      - (TR*, TE*) = argmax Jc_pred (top-1, backward-compat).
      - `candidates`: list top-K ứng viên theo `selection`:
          * "greedy": chọn J cao nhất, rồi các điểm xa (TR,TE) chuẩn hóa > min_dist_frac
            -- đa dạng nhưng nếu model collapse TE thì chỉ tách theo TR.
          * "te_stratified": chia candidate theo K bin TE (cạnh giới bởi TE_MIN,
            te_split, TE_MAX) lấy argmax mỗi bin -- ÉP có ứng viên TE dài để bắt
            chế độ T2 (true optimum TE>10).
    """
    TR_grid, TE_grid = np.meshgrid(tr_candidates.astype(float), te_candidates.astype(float))
    trc_flat, tec_flat = TR_grid.ravel(), TE_grid.ravel()

    tr0_norm = (tr0 - TR_MIN) / (TR_MAX - TR_MIN)
    te0_norm = (te0 - TE_MIN) / (TE_MAX - TE_MIN)
    j0_norm = (j0 - j_mean) / (j_std + 1e-8)
    trc_norm = (trc_flat - TR_MIN) / (TR_MAX - TR_MIN)
    tec_norm = (tec_flat - TE_MIN) / (TE_MAX - TE_MIN)

    n = len(trc_flat)
    params_all = np.stack([
        np.full(n, tr0_norm), np.full(n, te0_norm), np.full(n, j0_norm),
        trc_norm, tec_norm,
    ], axis=1)
    params_all = torch.tensor(params_all, dtype=torch.float32)

    preds = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_img = image_tensor.unsqueeze(0).repeat(end - start, 1, 1, 1).to(device)
            batch_params = params_all[start:end].to(device)
            pred_norm = model(batch_img, batch_params).cpu().numpy()
            preds[start:end] = pred_norm * j_std + j_mean  # denormalize

    J_grid = preds.reshape(TE_grid.shape)
    idx_te, idx_tr = np.unravel_index(np.argmax(J_grid), J_grid.shape)
    tr_star, te_star, j_max = tr_candidates[idx_tr], te_candidates[idx_te], J_grid[idx_te, idx_tr]

    if selection == "te_stratified":
        edges = _te_edges(top_k, te_split)
        candidates = _te_stratified_topk(trc_flat, tec_flat, preds, edges)
    else:
        candidates = _diverse_topk(
            trc_flat, tec_flat, preds,
            tr_range=(TR_MAX - TR_MIN), te_range=(TE_MAX - TE_MIN),
            top_k=top_k, min_dist_frac=min_dist_frac,
        )

    return {
        "TR_grid": TR_grid, "TE_grid": TE_grid, "J_grid": J_grid,
        "TR_star": float(tr_star), "TE_star": float(te_star), "J_max": float(j_max),
        "candidates": candidates,
    }


def _diverse_topk(tr_flat, te_flat, j_pred, tr_range, te_range, top_k, min_dist_frac):
    """Greedy selection: lấy candidate J cao nhất, rồi các candidate tiếp theo
    cách (theo khoảng cách chuẩn hóa TR/TE) tất cả ứng viên đã chọn > min_dist_frac."""
    order = np.argsort(j_pred)[::-1]
    picked = []
    for idx in order:
        tr, te = tr_flat[idx], te_flat[idx]
        ok = True
        for (ptr, pte, _) in picked:
            d = abs(tr - ptr) / tr_range + abs(te - pte) / te_range
            if d < min_dist_frac:
                ok = False
                break
        if ok:
            picked.append((float(tr), float(te), float(j_pred[idx])))
            if len(picked) >= top_k:
                break
    return [{"TR": t[0], "TE": t[1], "J_pred": t[2]} for t in picked]


def _te_edges(top_k, te_split):
    """Cạnh các bin TE. te_split = ngưỡng nội bộ (vd [15] -> bin [10,15],(15,200]).
    Nếu None, tự chọn theo top_k: k=2->[15], k=3->[15,60] (isolate TE≤15 khỏi T2)."""
    if te_split is None:
        if top_k == 2:
            te_split = [15.0]
        elif top_k == 3:
            te_split = [15.0, 60.0]
        else:
            qs = np.linspace(0, 1, top_k + 1)[1:-1]
            te_split = list(qs * (TE_MAX - TE_MIN) + TE_MIN)
    edges = [TE_MIN] + list(te_split) + [TE_MAX]
    return sorted(set(edges))


def _te_stratified_topk(tr_flat, te_flat, j_pred, te_edges):
    """Lấy argmax J_pred trong mỗi bin TE (bin cuối đóng cả 2 đầu). Đảm bảo có
    ứng viên TE dài để bắt chế độ T2 dù model bias về TE ngắn."""
    cands = []
    n_bins = len(te_edges) - 1
    for i in range(n_bins):
        lo, hi = te_edges[i], te_edges[i + 1]
        if i == n_bins - 1:
            mask = (te_flat >= lo) & (te_flat <= hi)
        else:
            mask = (te_flat >= lo) & (te_flat < hi)
        if not mask.any():
            continue
        idxs = np.where(mask)[0]
        best = idxs[np.argmax(j_pred[idxs])]
        cands.append((float(tr_flat[best]), float(te_flat[best]), float(j_pred[best])))
    return [{"TR": c[0], "TE": c[1], "J_pred": c[2]} for c in cands]


def plot_recommendation(result, tr0, te0, out_path):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(
        result["J_grid"], aspect="auto", origin="lower",
        extent=[TR_CANDIDATES.min(), TR_CANDIDATES.max(), TE_CANDIDATES.min(), TE_CANDIDATES.max()],
        cmap="viridis",
    )
    ax.scatter([result["TR_star"]], [result["TE_star"]], color="red", marker="*", s=250,
               edgecolor="white", label=f"Recommended (TR*={result['TR_star']:.0f}, TE*={result['TE_star']:.0f})")
    ax.scatter([tr0], [te0], color="orange", marker="o", s=100,
               edgecolor="white", label=f"Anchor (TR0={tr0:.0f}, TE0={te0:.0f})")
    ax.set_xlabel("TR (ms)")
    ax.set_ylabel("TE (ms)")
    ax.set_title("Predicted J landscape (CNN surrogate, anchor-based)")
    fig.colorbar(im, ax=ax, label="Predicted J")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to anchor MRI image (png)")
    parser.add_argument("--tr0", type=float, required=True, help="TR of the acquired image")
    parser.add_argument("--te0", type=float, required=True, help="TE of the acquired image")
    parser.add_argument("--j0", type=float, required=True,
                         help="J tại điểm anchor (tính từ CNR đo trên ảnh đã segment)")
    parser.add_argument("--checkpoint", default="training/checkpoints/best_model.pt")
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--out_dir", default="optimization/recommendations")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model, j_mean, j_std = load_model(args.checkpoint, args.device)
    if j_mean is None:
        raise ValueError("Checkpoint không chứa j_mean/j_std -- cần train lại bằng training/train.py mới")

    image_tensor = load_image(args.image)
    result = recommend(model, image_tensor, args.device, args.tr0, args.te0, args.j0, j_mean, j_std)

    print(f"Recommended: TR*={result['TR_star']:.0f} ms, TE*={result['TE_star']:.0f} ms, "
          f"predicted J={result['J_max']:.4f}")

    base_name = os.path.splitext(os.path.basename(args.image))[0]
    plot_path = os.path.join(args.out_dir, f"{base_name}_recommendation.png")
    plot_recommendation(result, args.tr0, args.te0, plot_path)
    print(f"Heatmap saved to {plot_path}")

    with open(os.path.join(args.out_dir, f"{base_name}_recommendation.txt"), "w") as f:
        f.write(f"image: {args.image}\n")
        f.write(f"TR0: {args.tr0}\nTE0: {args.te0}\nJ0: {args.j0}\n")
        f.write(f"TR_star: {result['TR_star']}\nTE_star: {result['TE_star']}\n")
        f.write(f"predicted_J_max: {result['J_max']}\n")


if __name__ == "__main__":
    main()