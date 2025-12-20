# model/neck/bev_neck.py
from __future__ import annotations

from typing import Tuple, Optional, Sequence, Union, List
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


__all__ = ["FPNLSS_Torch", "LSSFPN3D_Torch"]


# -------------------------
# 2D Neck: FPNLSS_Torch
# -------------------------
class FPNLSS_Torch(nn.Module):
    """
    Pure PyTorch replacement for mm FPN_LSS (2D).

    Input:
        feats: list/tuple of multi-scale 2D feature maps
            feats[i]: [B, C_i, H_i, W_i]

    Args:
        in_channels: 單一數值，直接指定輸入 channel 數
        in_channels_list: 如果用 multi-scale 版本，會從這裡依照 input_feature_index
                          把兩個 scale concat，因此 in_channels = C_low + C_high
        out_channels: 最終輸出 channel
        scale_factor: 對高層特徵的 upsample 倍數
        input_feature_index: 要拿哪兩層來做 FPN，例如 (0, 2)
        extra_upsample: 最後再多一次 upsample（例如從 H/2 → H）
        lateral: 如果不為 None，對低層特徵先做 1x1 conv 調整 channel
        norm_type: "bn" / "gn" / "none"
    """

    def __init__(
        self,
        in_channels: Optional[int] = None,
        in_channels_list: Optional[Sequence[int]] = None,
        out_channels: int = 256,
        scale_factor: int = 4,
        input_feature_index: Tuple[int, int] = (0, 2),
        extra_upsample: Optional[int] = 2,
        lateral: Optional[int] = None,
        norm_type: str = "bn",  # "bn" / "gn" / "none"
        align_corners: bool = True,
    ):
        super().__init__()
        self.input_feature_index = tuple(input_feature_index)
        self.scale_factor = int(scale_factor)
        self.extra_upsample = extra_upsample
        self.use_lateral = lateral is not None
        self.align_corners = align_corners
        self.norm_type = norm_type.lower()

        # resolve in_channels
        if in_channels is None:
            assert in_channels_list is not None, "Need in_channels or in_channels_list"
            i0, i1 = self.input_feature_index
            in_channels = int(in_channels_list[i0] + in_channels_list[i1])
        self.in_channels = int(in_channels)

        def norm_layer_2d(c: int) -> nn.Module:
            n = self.norm_type
            if n == "bn":
                return nn.BatchNorm2d(c)
            if n == "gn":
                g = 32
                if c % g != 0:
                    for gg in range(min(g, c), 0, -1):
                        if c % gg == 0:
                            g = gg
                            break
                return nn.GroupNorm(g, c)
            if n == "none":
                return nn.Identity()
            raise ValueError(f"Unsupported norm_type={self.norm_type}")

        # 上採樣，把高層特徵拉成跟低層同空間解析度
        self.up = nn.Upsample(
            scale_factor=self.scale_factor,
            mode="bilinear",
            align_corners=self.align_corners,
        )

        # optional lateral conv (低層先做 1x1 調整)
        if self.use_lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(lateral, lateral, kernel_size=1, padding=0, bias=False),
                norm_layer_2d(lateral),
                nn.ReLU(inplace=True),
            )

        channels_factor = 2 if (self.extra_upsample is not None) else 1
        mid_channels = out_channels * channels_factor

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm_layer_2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm_layer_2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        if self.extra_upsample is not None:
            self.up2 = nn.Sequential(
                nn.Upsample(
                    scale_factor=int(self.extra_upsample),
                    mode="bilinear",
                    align_corners=self.align_corners,
                ),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                norm_layer_2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=True),
            )
        else:
            self.up2 = None

    def forward(self, feats: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """
        feats: list/tuple
          feats[i]: [B, C_i, H_i, W_i]
        """
        i0, i1 = self.input_feature_index
        x_low = feats[i0]   # 比較高解析度
        x_high = feats[i1]  # 比較低解析度

        if self.use_lateral:
            x_low = self.lateral_conv(x_low)

        # 將低解析度特徵上採樣到與 x_low 一樣大小
        x_high = self.up(x_high)

        # concat + conv
        x = torch.cat([x_low, x_high], dim=1)
        x = self.conv(x)

        if self.up2 is not None:
            x = self.up2(x)
        return x


# -------------------------
# 3D Neck: LSSFPN3D_Torch
# -------------------------
class LSSFPN3D_Torch(nn.Module):
    """
    Pure PyTorch replacement for Phigent/mm LSSFPN3D.

    Input feats (typical):
        feats = [
          x_8  : (B,  C,   Dz,   Dy,   Dx),
          x_16 : (B, 2C, Dz/2, Dy/2, Dx/2),
          x_32 : (B, 4C, Dz/4, Dy/4, Dx/4),
        ]

    Forward:
        x_16 upsample x2, x_32 upsample x4 -> concat -> 1x1 Conv3d + Norm + ReLU
    """

    def __init__(
        self,
        in_channels: int,      # should be C + 2C + 4C = 7C
        out_channels: int,     # usually C
        with_cp: bool = False,
        norm_type: str = "bn",      # "bn" / "gn" / "none"
        align_corners: bool = True,
    ):
        super().__init__()
        self.with_cp = bool(with_cp)
        self.align_corners = bool(align_corners)
        self.norm_type = norm_type.lower()

        self.up1 = nn.Upsample(
            scale_factor=2,
            mode="trilinear",
            align_corners=self.align_corners,
        )
        self.up2 = nn.Upsample(
            scale_factor=4,
            mode="trilinear",
            align_corners=self.align_corners,
        )

        def norm_layer_3d(c: int) -> nn.Module:
            n = self.norm_type
            if n == "bn":
                return nn.BatchNorm3d(c)
            if n == "gn":
                g = 32
                if c % g != 0:
                    for gg in range(min(g, c), 0, -1):
                        if c % gg == 0:
                            g = gg
                            break
                return nn.GroupNorm(g, c)
            if n == "none":
                return nn.Identity()
            raise ValueError(f"Unsupported norm_type={self.norm_type}")

        self.conv = nn.Sequential(
            nn.Conv3d(int(in_channels), int(out_channels), kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer_3d(int(out_channels)),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _ckpt(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        # 兼容不同 torch 版本的 use_reentrant
        try:
            return checkpoint.checkpoint(module, x, use_reentrant=False)
        except TypeError:
            return checkpoint.checkpoint(module, x)

    def forward(self, feats: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        assert isinstance(feats, (list, tuple)) and len(feats) == 3, \
            "LSSFPN3D_Torch expects feats=[x_8,x_16,x_32]"
        x_8, x_16, x_32 = feats

        x_16 = self.up1(x_16)
        x_32 = self.up2(x_32)

        x = torch.cat([x_8, x_16, x_32], dim=1)  # (B, 7C, Dz, Dy, Dx)

        if self.with_cp and x.requires_grad:
            x = self._ckpt(self.conv, x)
        else:
            x = self.conv(x)
        return x
