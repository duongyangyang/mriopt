"""
PyTorch Dataset (Anchor-based design)
=========================================
Mỗi mẫu: (ảnh ở TR0/TE0, TR0, TE0, J0, TRc, TEc) -> label = Jc
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

TR_MIN, TR_MAX = 200.0, 4000.0
TE_MIN, TE_MAX = 10.0, 200.0


class MRIParamDataset(Dataset):
    def __init__(self, df: pd.DataFrame, dataset_root: str, j_mean: float = None, j_std: float = None):
        """
        df       : DataFrame con (đã lọc theo phantom cho train/val/test)
        j_mean/j_std : thống kê chuẩn hóa J -- PHẢI tính trên toàn bộ metadata (train set)
                        và truyền giống nhau cho val/test để tránh leakage thống kê.
        """
        self.df = df.reset_index(drop=True)
        self.root = dataset_root

        # J0 và Jc dùng CHUNG 1 thống kê chuẩn hóa vì cùng phân phối giá trị J
        if j_mean is None or j_std is None:
            all_j = pd.concat([self.df["J0"], self.df["Jc"]])
            self.j_mean = all_j.mean()
            self.j_std = all_j.std()
        else:
            self.j_mean = j_mean
            self.j_std = j_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, row["image_path"])
        image = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)

        tr0_norm = (row["TR0"] - TR_MIN) / (TR_MAX - TR_MIN)
        te0_norm = (row["TE0"] - TE_MIN) / (TE_MAX - TE_MIN)
        j0_norm = (row["J0"] - self.j_mean) / (self.j_std + 1e-8)
        trc_norm = (row["TRc"] - TR_MIN) / (TR_MAX - TR_MIN)
        tec_norm = (row["TEc"] - TE_MIN) / (TE_MAX - TE_MIN)

        params = torch.tensor([tr0_norm, te0_norm, j0_norm, trc_norm, tec_norm], dtype=torch.float32)

        jc_norm = (row["Jc"] - self.j_mean) / (self.j_std + 1e-8)
        target = torch.tensor(jc_norm, dtype=torch.float32)

        return image, params, target

    def denormalize_j(self, j_norm):
        return j_norm * self.j_std + self.j_mean


class MRIParamListDataset(Dataset):
    """
    Listwise (anchor-grouped) variant of MRIParamDataset for ranking-aware
    training. Mỗi item = 1 anchor (ảnh ở TR0/TE0) kèm DANH SÁCH candidate
    (TRc, TEc, Jc) của anchor đó, pad về `list_size`.

    Returns per item: (image (1,H,W), params (L,5), target (L,), mask (L,))
        params[:, :] = [TR0n, TE0n, J0n, TRcn, TEcn]  (chuẩn hóa)
        target      = Jc chuẩn hóa
        mask        = 1 ở candidate thật, 0 ở ô pad
    """
    def __init__(self, df: pd.DataFrame, dataset_root: str, list_size: int = 32,
                 j_mean: float = None, j_std: float = None):
        self.root = dataset_root
        self.list_size = list_size
        # group theo image_path: mỗi group = 1 anchor + các candidate của nó
        self.groups = [g for _, g in df.groupby("image_path", sort=False)]

        if j_mean is None or j_std is None:
            all_j = pd.concat([df["J0"], df["Jc"]])
            self.j_mean = all_j.mean()
            self.j_std = all_j.std()
        else:
            self.j_mean = j_mean
            self.j_std = j_std

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        g = self.groups[idx]
        # ảnh anchor dùng chung cho cả list -- load 1 lần
        img_path = os.path.join(self.root, g["image_path"].iloc[0])
        image = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)

        L = self.list_size
        params = torch.zeros(L, 5, dtype=torch.float32)
        target = torch.zeros(L, dtype=torch.float32)
        mask = torch.zeros(L, dtype=torch.float32)

        rows = g.head(L)
        for k, row in enumerate(rows.itertuples()):
            tr0_norm = (row.TR0 - TR_MIN) / (TR_MAX - TR_MIN)
            te0_norm = (row.TE0 - TE_MIN) / (TE_MAX - TE_MIN)
            j0_norm = (row.J0 - self.j_mean) / (self.j_std + 1e-8)
            trc_norm = (row.TRc - TR_MIN) / (TR_MAX - TR_MIN)
            tec_norm = (row.TEc - TE_MIN) / (TE_MAX - TE_MIN)
            params[k] = torch.tensor([tr0_norm, te0_norm, j0_norm, trc_norm, tec_norm],
                                      dtype=torch.float32)
            target[k] = (row.Jc - self.j_mean) / (self.j_std + 1e-8)
            mask[k] = 1.0
        return image, params, target, mask

    def denormalize_j(self, j_norm):
        return j_norm * self.j_std + self.j_mean
