"""
Train CNN (Anchor-based design) -- split theo PHANTOM để tránh data leakage
================================================================================
f(image, TR0, TE0, J0, TRc, TEc) -> Jc

Chạy local (CPU, test nhanh):
    PYTHONPATH=. python3 training/train.py --epochs 2 --batch_size 32

Chạy GPU cloud:
    PYTHONPATH=. python3 training/train.py --epochs 30 --batch_size 128 --device cuda

Chạy Mac (Apple Silicon):
    PYTHONPATH=. python3 training/train.py --epochs 30 --batch_size 32 --device mps --num_workers 0
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.dataset import MRIParamDataset
from models.cnn_model import MRIParamNet

# ------------------------------------------------------------------
# Fix cảnh báo cuDNN "Plan failed with a cudnnException ... CUDNN_STATUS_NOT_SUPPORTED"
# Cảnh báo này xuất hiện khi cuDNN benchmark cố thử 1 thuật toán convolution
# không được driver/cuDNN hiện tại hỗ trợ. Tắt benchmark autotune tránh được
# việc thử các plan không tương thích; deterministic cũng giúp kết quả ổn định
# hơn để so sánh giữa các lần chạy.
# ------------------------------------------------------------------
torch.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore", message="Plan failed with a cudnnException")


def evaluate(model, loader, device, dataset, desc="Evaluating"):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for image, params, target in tqdm(loader, desc=desc, leave=False):
            image, params = image.to(device), params.to(device)
            pred = model(image, params).cpu().numpy()
            preds.append(pred)
            targets.append(target.numpy())
    preds = dataset.denormalize_j(np.concatenate(preds))
    targets = dataset.denormalize_j(np.concatenate(targets))
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)
    return mae, rmse, r2


def split_by_phantom(df: pd.DataFrame, seed: int, train_frac=0.8, val_frac=0.1):
    """
    QUAN TRỌNG: chia train/val/test theo PHANTOM (không theo từng dòng dữ liệu),
    để đảm bảo 1 phantom chỉ xuất hiện trong đúng 1 tập -- tránh data leakage
    (ảnh cùng phantom rất giống nhau, nếu lẫn giữa train/test sẽ đánh giá sai).
    """
    phantoms = sorted(int(p) for p in df["phantom"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(phantoms)

    n = len(phantoms)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_phantoms = set(phantoms[:n_train])
    val_phantoms = set(phantoms[n_train:n_train + n_val])
    test_phantoms = set(phantoms[n_train + n_val:])

    print(f"Phantom split: train={sorted(train_phantoms)[:5]}...({len(train_phantoms)}), "
          f"val={sorted(val_phantoms)}, test={sorted(test_phantoms)}")

    train_df = df[df["phantom"].isin(train_phantoms)]
    val_df = df[df["phantom"].isin(val_phantoms)]
    test_df = df[df["phantom"].isin(test_phantoms)]
    return train_df, val_df, test_df, train_phantoms, val_phantoms, test_phantoms


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="dataset/metadata.csv")
    parser.add_argument("--dataset_root", default="dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default=get_default_device())
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", default="training/checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device: {args.device}")
    full_df = pd.read_csv(args.metadata)

    train_df, val_df, test_df, train_p, val_p, test_p = split_by_phantom(full_df, seed=args.seed)
    print(f"Rows: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Thống kê chuẩn hóa J CHỈ tính trên train set, dùng chung cho val/test (tránh leakage thống kê)
    all_j_train = pd.concat([train_df["J0"], train_df["Jc"]])
    j_mean, j_std = all_j_train.mean(), all_j_train.std()
    print(f"J normalization stats (from train only): mean={j_mean:.4f}, std={j_std:.4f}")

    train_ds = MRIParamDataset(train_df, args.dataset_root, j_mean=j_mean, j_std=j_std)
    val_ds = MRIParamDataset(val_df, args.dataset_root, j_mean=j_mean, j_std=j_std)
    test_ds = MRIParamDataset(test_df, args.dataset_root, j_mean=j_mean, j_std=j_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=(args.device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(args.device == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(args.device == "cuda"))

    model = MRIParamNet(pretrained=True).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    # Lưu lại toàn bộ lịch sử huấn luyện (train_loss, val_loss, lr, thời gian
    # mỗi epoch) -- cần thiết để vẽ đồ thị loss curve và phát hiện overfitting
    # trong báo cáo (mục 3.2.4/3.2.5 "Model Evaluation").
    # ------------------------------------------------------------------
    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": [], "epoch_time_sec": []}

    best_val_loss = float("inf")
    import time as _time

    for epoch in range(1, args.epochs + 1):
        epoch_start = _time.time()

        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        for image, params, target in pbar:
            image, params, target = image.to(args.device), params.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            pred = model(image, params)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * image.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for image, params, target in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]", leave=False):
                image, params, target = image.to(args.device), params.to(args.device), target.to(args.device)
                pred = model(image, params)
                val_loss += criterion(pred, target).item() * image.size(0)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        epoch_time = _time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        history["epoch_time_sec"].append(epoch_time)

        print(f"Epoch {epoch}/{args.epochs} - train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"lr={current_lr:.2e} time={epoch_time:.1f}s")

        # Lưu history sau MỖI epoch (không chỉ lúc kết thúc) để không mất dữ liệu nếu bị ngắt giữa chừng
        pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "training_history.csv"), index=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "train_loss": train_loss,
                "j_mean": j_mean,
                "j_std": j_std,
                "train_phantoms": sorted(train_p),
                "val_phantoms": sorted(val_p),
                "test_phantoms": sorted(test_p),
                "args": vars(args),
            }, os.path.join(args.out_dir, "best_model.pt"))
            print(f"  -> saved new best model (val_loss={val_loss:.4f})")

    # ------------------------------------------------------------------
    # Vẽ loss curve (train vs val theo epoch) -- hình bắt buộc cho phần đánh giá
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.plot(history["epoch"], history["train_loss"], label="Train loss", marker="o", markersize=3)
        ax.plot(history["epoch"], history["val_loss"], label="Validation loss", marker="o", markersize=3)
        best_epoch = history["epoch"][int(np.argmin(history["val_loss"]))]
        ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.6,
                    label=f"Best epoch ({best_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss (normalized J)")
        ax.set_title("Training / Validation Loss Curve")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "loss_curve.png"), dpi=130)
        plt.close(fig)
        print(f"Loss curve saved to {os.path.join(args.out_dir, 'loss_curve.png')}")
    except Exception as e:
        print(f"WARNING: could not plot loss curve ({e})")

    # Final test evaluation using best checkpoint
    checkpoint = torch.load(os.path.join(args.out_dir, "best_model.pt"), map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    mae, rmse, r2 = evaluate(model, test_loader, args.device, test_ds, desc="Final test eval")
    print(f"\nTest set (phantom-level held-out, {len(test_p)} phantoms): "
          f"MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

    # ------------------------------------------------------------------
    # Lưu đầy đủ metrics quan trọng vào 1 file JSON duy nhất (dễ đọc lại,
    # dễ đưa vào báo cáo) thay vì chỉ txt rời rạc như trước.
    # ------------------------------------------------------------------
    summary = {
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_loss": float(checkpoint["val_loss"]),
        "best_train_loss": float(checkpoint["train_loss"]),
        "test_MAE": float(mae),
        "test_RMSE": float(rmse),
        "test_R2": float(r2),
        "j_mean": float(j_mean),
        "j_std": float(j_std),
        "n_train_rows": len(train_df),
        "n_val_rows": len(val_df),
        "n_test_rows": len(test_df),
        "train_phantoms": sorted(train_p),
        "val_phantoms": sorted(val_p),
        "test_phantoms": sorted(test_p),
        "hyperparameters": vars(args),
        "total_epochs_run": args.epochs,
    }
    with open(os.path.join(args.out_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Giữ lại test_metrics.txt (tương thích ngược với script cũ)
    with open(os.path.join(args.out_dir, "test_metrics.txt"), "w") as f:
        f.write(f"MAE={mae:.4f}\nRMSE={rmse:.4f}\nR2={r2:.4f}\n")
        f.write(f"test_phantoms={sorted(test_p)}\n")

    print(f"\nĐã lưu: {args.out_dir}/best_model.pt, training_history.csv, "
          f"loss_curve.png, training_summary.json, test_metrics.txt")


if __name__ == "__main__":
    main()
