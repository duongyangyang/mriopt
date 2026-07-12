"""
Train CNN (Anchor-based) với ranking-aware loss -- ListNet / pairwise hinge
==========================================================================
f(image, TR0, TE0, J0, TRc, TEc) -> Jc

Khác với train.py (chỉ MSE):
  - Dataset listwise: mỗi item = 1 anchor + danh sách candidate của nó
    (MRIParamListDataset), ép model dùng anchor để SẮP XẾP candidate thay vì
    hồi quy về trung bình toàn cục -- trực tiếp nhắm vào tác vụ argmax và mặt
    J đa cực trị.
  - Loss = masked_MSE + λ_rank * pairwise_hinge(theo anchor).
  - Val metric phụ: Spearman trung bình/giữa pred và true J trong mỗi list.

Chạy Mac (Apple Silicon):
    PYTHONPATH=. python3 training/train_rank.py --epochs 30 --device mps

Các tùy chọn mới:
    --mse_weight: trọng số nhân cho term masked MSE trong loss tổng
    --save_metric: chọn checkpoint tốt nhất theo 'mse' hoặc 'rank'
"""
import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.dataset import MRIParamDataset, MRIParamListDataset
from models.cnn_model import MRIParamNet
# Tái dùng logic split/evaluate/device đã có (không đè baseline train.py)
from training.train import split_by_phantom, evaluate, get_default_device

torch.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore", message="Plan failed with a cudnnException")


# -------------------------- ranking utilities --------------------------
def _rankdata_1d(x: np.ndarray) -> np.ndarray:
    """Ranks (average ties) cho mảng 1D -- thay scipy.stats.rankdata."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    # gộp ties: trung bình rank cho các giá trị bằng nhau
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    return ranks


def masked_mse(pred, target, mask):
    diff2 = (pred - target) ** 2 * mask
    return diff2.sum() / mask.sum().clamp(min=1.0)


def pairwise_hinge(pred, target, mask, margin):
    """
    pred/target/mask: (B, L). Với mỗi anchor, với mọi cặp (i,j) có
    true_i > true_j (và cả hai mask=1), hinge = max(0, margin - (pred_i - pred_j)).
    Đường chéo tự loại (true_i == true_j).
    """
    pi = pred.unsqueeze(2)       # (B, L, 1) = pred_i
    pj = pred.unsqueeze(1)       # (B, 1, L) = pred_j
    ti = target.unsqueeze(2)
    tj = target.unsqueeze(1)
    mi = mask.unsqueeze(2)
    mj = mask.unsqueeze(1)
    valid = (mi * mj) * (ti > tj).float()      # (B, L, L)
    hinge = torch.clamp(margin - (pi - pj), min=0.0)
    n_valid = valid.sum().clamp(min=1.0)
    return (hinge * valid).sum() / n_valid


def val_spearman(pred, target, mask):
    """Spearman trung bình trên các list có >=2 candidate hợp lệ."""
    p = pred.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    m = mask.detach().cpu().numpy()
    corrs = []
    for b in range(p.shape[0]):
        idx = m[b] > 0
        if idx.sum() < 2:
            continue
        pr = _rankdata_1d(p[b][idx])
        tr = _rankdata_1d(t[b][idx])
        pr = pr - pr.mean()
        tr = tr - tr.mean()
        denom = np.sqrt((pr ** 2).sum() * (tr ** 2).sum())
        if denom > 0:
            corrs.append((pr * tr).sum() / denom)
    return float(np.mean(corrs)) if corrs else float("nan")


@torch.no_grad()
def eval_listwise(model, loader, device, desc="Val"):
    """Trả (masked_mse, spearman) trên list dataset."""
    model.eval()
    mses, spearmans = [], []
    for image, params, target, mask in tqdm(loader, desc=desc, leave=False):
        image, params, target, mask = image.to(device), params.to(device), target.to(device), mask.to(device)
        pred = _forward_list(model, image, params, len(target))
        mses.append(masked_mse(pred, target, mask).item())
        spearmans.append(val_spearman(pred, target, mask))
    if not mses:
        return float("nan"), float("nan")
    return float(np.mean(mses)), float(np.nanmean(spearmans))


def _forward_list(model, image, params, B):
    """image: (B,1,H,W) dùng chung cho L candidate; params: (B,L,5) -> pred (B,L)."""
    L = params.shape[1]
    H, W = image.shape[-2], image.shape[-1]
    image = image.unsqueeze(1).expand(B, L, 1, H, W).reshape(B * L, 1, H, W)
    params = params.reshape(B * L, 5)
    return model(image, params).reshape(B, L)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="dataset/metadata.csv")
    parser.add_argument("--dataset_root", default="dataset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8, help="Số anchor/batch")
    parser.add_argument("--list_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mse_weight", type=float, default=1.0,
                        help="Trọng số nhân cho masked MSE trong loss tổng: loss = mse_weight * mse + lambda_rank * rank")
    parser.add_argument("--lambda_rank", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.1, help="Margin hinge (đơn vị J chuẩn hóa)")
    parser.add_argument("--save_metric", choices=["mse", "rank"], default="mse",
                        help="Chọn checkpoint tốt nhất theo validation MSE hoặc validation Spearman rank")
    parser.add_argument("--device", default=get_default_device())
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_dir", default="training/checkpoints")
    parser.add_argument("--ckpt_name", default="best_model_rank.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device: {args.device}")
    full_df = pd.read_csv(args.metadata)

    train_df, val_df, test_df, train_p, val_p, test_p = split_by_phantom(full_df, seed=args.seed)
    print(f"Rows: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    all_j_train = pd.concat([train_df["J0"], train_df["Jc"]])
    j_mean, j_std = all_j_train.mean(), all_j_train.std()
    print(f"J normalization stats (from train only): mean={j_mean:.4f}, std={j_std:.4f}")

    train_ds = MRIParamListDataset(train_df, args.dataset_root, list_size=args.list_size,
                                    j_mean=j_mean, j_std=j_std)
    val_ds = MRIParamListDataset(val_df, args.dataset_root, list_size=args.list_size,
                                  j_mean=j_mean, j_std=j_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=(args.device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(args.device == "cuda"))

    model = MRIParamNet(pretrained=True).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    margin_t = torch.tensor(args.margin, device=args.device)

    history = {"epoch": [], "train_loss": [], "train_mse": [], "train_rank": [],
               "val_loss": [], "val_spearman": [], "lr": [], "epoch_time_sec": []}
    if args.save_metric == "mse":
        best_score = float("inf")
    else:
        best_score = -float("inf")
    import time as _time

    for epoch in range(1, args.epochs + 1):
        epoch_start = _time.time()
        model.train()
        tot_loss = tot_mse = tot_rank = 0.0
        n_lists = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        for image, params, target, mask in pbar:
            B = len(target)
            image = image.to(args.device)
            params = params.to(args.device)
            target = target.to(args.device)
            mask = mask.to(args.device)

            optimizer.zero_grad()
            pred = _forward_list(model, image, params, B)

            mse = masked_mse(pred, target, mask)
            rank = pairwise_hinge(pred, target, mask, margin_t)
            loss = args.mse_weight * mse + args.lambda_rank * rank
            loss.backward()
            optimizer.step()

            tot_loss += loss.item() * B
            tot_mse += mse.item() * B
            tot_rank += rank.item() * B
            n_lists += B
            pbar.set_postfix(loss=f"{loss.item():.4f}", mse=f"{mse.item():.4f}", rank=f"{rank.item():.4f}")

        train_loss = tot_loss / max(n_lists, 1)
        train_mse = tot_mse / max(n_lists, 1)
        train_rank = tot_rank / max(n_lists, 1)

        val_mse, val_sp = eval_listwise(model, val_loader, args.device, desc=f"Epoch {epoch} [val]")
        val_loss = val_mse  # theo dõi theo MSE chuẩn hóa (phần regression)
        scheduler.step(val_loss)

        epoch_time = _time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_mse"].append(train_mse)
        history["train_rank"].append(train_rank)
        history["val_loss"].append(val_loss)
        history["val_spearman"].append(val_sp)
        history["lr"].append(current_lr)
        history["epoch_time_sec"].append(epoch_time)
        pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "training_history_rank.csv"), index=False)

        print(f"Epoch {epoch}/{args.epochs} - loss={train_loss:.4f} mse={train_mse:.4f} rank={train_rank:.4f} "
              f"| val_mse={val_loss:.4f} val_spearman={val_sp:.4f} lr={current_lr:.2e} time={epoch_time:.1f}s")

        if args.save_metric == "mse":
            improved = np.isfinite(val_loss) and val_loss < best_score
            current_metric = val_loss
        else:
            improved = np.isfinite(val_sp) and val_sp > best_score
            current_metric = val_sp

        if improved:
            best_score = current_metric
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_spearman": val_sp,
                "train_loss": train_loss,
                "j_mean": j_mean,
                "j_std": j_std,
                "mse_weight": args.mse_weight,
                "lambda_rank": args.lambda_rank,
                "margin": args.margin,
                "save_metric": args.save_metric,
                "train_phantoms": sorted(train_p),
                "val_phantoms": sorted(val_p),
                "test_phantoms": sorted(test_p),
                "args": vars(args),
            }, os.path.join(args.out_dir, args.ckpt_name))
            if args.save_metric == "mse":
                print(f"  -> saved new best model (val_mse={current_metric:.4f}, val_spearman={val_sp:.4f})")
            else:
                print(f"  -> saved new best model (val_spearman={current_metric:.4f}, val_mse={val_loss:.4f})")

    # ---- Final test eval (single-sample, tái dùng evaluate() từ train.py) ----
    ckpt_path = os.path.join(args.out_dir, args.ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"\nWARNING: checkpoint {ckpt_path} chưa được lưu (val rỗng?) -- bỏ qua final test eval.")
        return
    test_ds_single = MRIParamDataset(test_df, args.dataset_root, j_mean=j_mean, j_std=j_std)
    test_loader_single = DataLoader(test_ds_single, batch_size=64, shuffle=False,
                                     num_workers=args.num_workers,
                                     pin_memory=(args.device == "cuda"))
    checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    mae, rmse, r2 = evaluate(model, test_loader_single, args.device, test_ds_single, desc="Final test eval")
    print(f"\nTest set (phantom-level held-out, {len(test_p)} phantoms): "
          f"MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

    summary = {
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_mse": float(checkpoint["val_loss"]),
        "best_val_spearman": float(checkpoint["val_spearman"]),
        "test_MAE": float(mae),
        "test_RMSE": float(rmse),
        "test_R2": float(r2),
        "j_mean": float(j_mean),
        "j_std": float(j_std),
        "lambda_rank": args.lambda_rank,
        "margin": args.margin,
        "n_train_anchors": len(train_ds),
        "n_val_anchors": len(val_ds),
        "train_phantoms": sorted(train_p),
        "val_phantoms": sorted(val_p),
        "test_phantoms": sorted(test_p),
        "hyperparameters": vars(args),
    }
    with open(os.path.join(args.out_dir, "training_summary_rank.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nĐã lưu: {args.out_dir}/{args.ckpt_name}, training_history_rank.csv, "
          f"training_summary_rank.json")


if __name__ == "__main__":
    main()
