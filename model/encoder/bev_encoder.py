# model/bev_encoder.py
from __future__ import annotations

from typing import List, Optional, Sequence, Dict, Any

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


__all__ = ["CustomResNet_NoMM", "CustomResNet3D_NoMM"]


# -------------------------
# Norm builders (no-mm)
# -------------------------
def _get_norm_type(norm_cfg: Optional[Dict[str, Any]]) -> str:
    if norm_cfg is None:
        return "BN"
    t = norm_cfg.get("type", "BN")
    return str(t)


def build_norm2d(num_features: int, norm_cfg: Optional[Dict[str, Any]] = None) -> nn.Module:
    t = _get_norm_type(norm_cfg)
    if t == "BN":
        eps = float(norm_cfg.get("eps", 1e-5)) if norm_cfg else 1e-5
        momentum = float(norm_cfg.get("momentum", 0.1)) if norm_cfg else 0.1
        return nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
    if t == "SyncBN":
        eps = float(norm_cfg.get("eps", 1e-5)) if norm_cfg else 1e-5
        momentum = float(norm_cfg.get("momentum", 0.1)) if norm_cfg else 0.1
        return nn.SyncBatchNorm(num_features, eps=eps, momentum=momentum)
    if t == "GN":
        num_groups = int(norm_cfg.get("num_groups", 32)) if norm_cfg else 32
        if num_features % num_groups != 0:
            for g in range(min(num_groups, num_features), 0, -1):
                if num_features % g == 0:
                    num_groups = g
                    break
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
    raise ValueError(f"Unsupported norm type: {t}")


def build_norm3d(num_features: int, norm_cfg: Optional[Dict[str, Any]] = None) -> nn.Module:
    t = _get_norm_type(norm_cfg)
    if t == "BN":
        eps = float(norm_cfg.get("eps", 1e-5)) if norm_cfg else 1e-5
        momentum = float(norm_cfg.get("momentum", 0.1)) if norm_cfg else 0.1
        return nn.BatchNorm3d(num_features, eps=eps, momentum=momentum)
    if t == "SyncBN":
        # SyncBN 對 3D 不太實用，直接用 GN 比較穩
        num_groups = int(norm_cfg.get("num_groups", 32)) if norm_cfg else 32
        if num_features % num_groups != 0:
            for g in range(min(num_groups, num_features), 0, -1):
                if num_features % g == 0:
                    num_groups = g
                    break
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
    if t == "GN":
        num_groups = int(norm_cfg.get("num_groups", 32)) if norm_cfg else 32
        if num_features % num_groups != 0:
            for g in range(min(num_groups, num_features), 0, -1):
                if num_features % g == 0:
                    num_groups = g
                    break
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
    raise ValueError(f"Unsupported norm type for 3D: {t}")


# -------------------------
# Blocks (2D)
# -------------------------
def conv3x3_2d(in_c: int, out_c: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1_2d(in_c: int, out_c: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0, bias=False)


class BasicBlock2D_NoMM(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.conv1 = conv3x3_2d(inplanes, planes, stride=stride)
        self.bn1 = build_norm2d(planes, norm_cfg)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_2d(planes, planes, stride=1)
        self.bn2 = build_norm2d(planes, norm_cfg)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class Bottleneck2D_NoMM(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        outplanes = planes * self.expansion

        self.conv1 = conv1x1_2d(inplanes, planes, stride=1)
        self.bn1 = build_norm2d(planes, norm_cfg)

        self.conv2 = conv3x3_2d(planes, planes, stride=stride)
        self.bn2 = build_norm2d(planes, norm_cfg)

        self.conv3 = conv1x1_2d(planes, outplanes, stride=1)
        self.bn3 = build_norm2d(outplanes, norm_cfg)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


# -------------------------
# CustomResNet (2D) no-mm
# -------------------------
class CustomResNet_NoMM(nn.Module):
    """
    Drop-in replacement for Phigent CustomResNet, without mmcv/mmdet.

    Args:
        numC_input, num_layer, num_channels, stride,
        backbone_output_ids, norm_cfg, with_cp, block_type
    """

    def __init__(
        self,
        numC_input: int,
        num_layer: Sequence[int] = (2, 2, 2),
        num_channels: Optional[Sequence[int]] = None,
        stride: Sequence[int] = (2, 2, 2),
        backbone_output_ids: Optional[Sequence[int]] = None,
        norm_cfg: Optional[Dict[str, Any]] = None,
        with_cp: bool = False,
        block_type: str = "Basic",
        downsample_with_norm: bool = False,
    ):
        super().__init__()
        assert len(num_layer) == len(stride), "num_layer 與 stride 長度需相同"

        if num_channels is None:
            num_channels = [numC_input * (2 ** (i + 1)) for i in range(len(num_layer))]

        if backbone_output_ids is None:
            backbone_output_ids = list(range(len(num_layer)))
        self.backbone_output_ids = set(list(backbone_output_ids))

        self.with_cp = bool(with_cp)
        self.norm_cfg = norm_cfg
        self.downsample_with_norm = bool(downsample_with_norm)

        layers: List[nn.Module] = []
        curr_numC = int(numC_input)

        block_type = str(block_type)
        if block_type == "BottleNeck":
            Block = Bottleneck2D_NoMM
            for i in range(len(num_layer)):
                out_c = int(num_channels[i])
                st = int(stride[i])
                planes = out_c // 4

                down = nn.Conv2d(curr_numC, out_c, kernel_size=3, stride=st, padding=1, bias=False)
                if self.downsample_with_norm:
                    down = nn.Sequential(down, build_norm2d(out_c, norm_cfg))

                stage = [Block(inplanes=curr_numC, planes=planes, stride=st, downsample=down, norm_cfg=norm_cfg)]
                curr_numC = out_c
                for _ in range(int(num_layer[i]) - 1):
                    stage.append(Block(inplanes=curr_numC, planes=planes, stride=1, downsample=None, norm_cfg=norm_cfg))
                layers.append(nn.Sequential(*stage))

        elif block_type == "Basic":
            Block = BasicBlock2D_NoMM
            for i in range(len(num_layer)):
                out_c = int(num_channels[i])
                st = int(stride[i])

                down = nn.Conv2d(curr_numC, out_c, kernel_size=3, stride=st, padding=1, bias=False)
                if self.downsample_with_norm:
                    down = nn.Sequential(down, build_norm2d(out_c, norm_cfg))

                stage = [Block(inplanes=curr_numC, planes=out_c, stride=st, downsample=down, norm_cfg=norm_cfg)]
                curr_numC = out_c
                for _ in range(int(num_layer[i]) - 1):
                    stage.append(Block(inplanes=curr_numC, planes=out_c, stride=1, downsample=None, norm_cfg=norm_cfg))
                layers.append(nn.Sequential(*stage))
        else:
            raise ValueError("block_type must be 'Basic' or 'BottleNeck'")

        self.layers = nn.ModuleList(layers)

        # ⭐ 給 FPN 用：每個 stage 的輸出 channel
        self.out_channels_list = list(num_channels)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp and x_tmp.requires_grad:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats


# -------------------------
# Blocks (3D)
# -------------------------
def conv3x3_3d(in_c: int, out_c: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock3D_NoMM(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.conv1 = conv3x3_3d(channels_in, channels_out, stride=stride)
        self.bn1 = build_norm3d(channels_out, norm_cfg)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3_3d(channels_out, channels_out, stride=1)
        self.bn2 = build_norm3d(channels_out, norm_cfg)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class CustomResNet3D_NoMM(nn.Module):
    """
    Drop-in replacement for Phigent CustomResNet3D, without mmcv/mmdet.
    """

    def __init__(
        self,
        numC_input: int,
        num_layer: Sequence[int] = (2, 2, 2),
        num_channels: Optional[Sequence[int]] = None,
        stride: Sequence[int] = (2, 2, 2),
        backbone_output_ids: Optional[Sequence[int]] = None,
        with_cp: bool = False,
        norm_cfg: Optional[Dict[str, Any]] = None,
        downsample_with_norm: bool = False,
    ):
        super().__init__()
        assert len(num_layer) == len(stride), "num_layer 與 stride 長度需相同"

        if num_channels is None:
            num_channels = [numC_input * (2 ** (i + 1)) for i in range(len(num_layer))]

        if backbone_output_ids is None:
            backbone_output_ids = list(range(len(num_layer)))
        self.backbone_output_ids = set(list(backbone_output_ids))

        self.with_cp = bool(with_cp)
        self.norm_cfg = norm_cfg
        self.downsample_with_norm = bool(downsample_with_norm)

        layers: List[nn.Module] = []
        curr_c = int(numC_input)

        for i in range(len(num_layer)):
            out_c = int(num_channels[i])
            st = int(stride[i])

            down = nn.Conv3d(curr_c, out_c, kernel_size=3, stride=st, padding=1, bias=False)
            if self.downsample_with_norm:
                down = nn.Sequential(down, build_norm3d(out_c, norm_cfg))

            stage = [BasicBlock3D_NoMM(curr_c, out_c, stride=st, downsample=down, norm_cfg=norm_cfg)]
            curr_c = out_c
            for _ in range(int(num_layer[i]) - 1):
                stage.append(BasicBlock3D_NoMM(curr_c, curr_c, stride=1, downsample=None, norm_cfg=norm_cfg))
            layers.append(nn.Sequential(*stage))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp and x_tmp.requires_grad:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats
