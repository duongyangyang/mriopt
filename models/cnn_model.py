"""
CNN Model (Anchor-based design)
==================================
Input:  ảnh MRI đã chụp (ở TR0/TE0) + (TR0, TE0, J0, TR_c, TE_c)
Output: J_c dự đoán (giá trị mục tiêu tại điểm candidate)

f(image, TR0, TE0, J0, TR_c, TE_c) -> J_c
"""
import torch
import torch.nn as nn
import torchvision.models as models


class MRIParamNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                    stride=old_conv.stride, padding=old_conv.padding, bias=False)
        if pretrained:
            backbone.conv1.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)

        backbone.fc = nn.Identity()
        self.backbone = backbone  # outputs 512-d feature vector

        # Param branch: [TR0_norm, TE0_norm, J0_norm, TRc_norm, TEc_norm] -> 5 scalars
        self.param_mlp = nn.Sequential(
            nn.Linear(5, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(512 + 32, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, image, params):
        """
        image  : (B, 1, H, W) -- ảnh chụp ở (TR0, TE0), normalized [0,1]
        params : (B, 5) -- [TR0_norm, TE0_norm, J0_norm, TRc_norm, TEc_norm]
        """
        feat_img = self.backbone(image)          # (B, 512)
        feat_param = self.param_mlp(params)        # (B, 32)
        combined = torch.cat([feat_img, feat_param], dim=1)
        return self.head(combined).squeeze(-1)    # (B,) -> J_c dự đoán
