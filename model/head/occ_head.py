# model/head/occ_head.py
from __future__ import annotations

from typing import Optional, Dict, Any, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..loss.semkitti_loss import sem_scal_loss, geo_scal_loss
from ..loss.lovasz_softmax import lovasz_softmax


# -------------------------
# class frequency (same as MM file)
# -------------------------
nusc_class_frequencies = np.array([
    944004,
    1897170,
    152386,
    2391677,
    16957802,
    724139,
    189027,
    2074468,
    413451,
    2384460,
    5916653,
    175883646,
    4275424,
    51393615,
    61411620,
    105975596,
    116424404,
    1892500630
], dtype=np.float64)


# -------------------------
# Helpers
# -------------------------
def _flatten_logits_and_labels(
    pred: torch.Tensor,
    label: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    pred:
      - (N, C) OR (B, C, d1, d2, ...)
    label:
      - (N,)  OR (B, d1, d2, ...)
    sample_weight:
      - (N,)  OR (B, d1, d2, ...)
    Return:
      pred_flat: (M, C)
      label_flat: (M,)
      w_flat: (M,) or None
    """
    if pred.dim() == 2:
        pred_flat = pred
        label_flat = label.reshape(-1)
        w_flat = sample_weight.reshape(-1) if sample_weight is not None else None
        return pred_flat, label_flat, w_flat

    # (B,C,...) -> (B,...,C) -> (M,C)
    assert pred.dim() >= 3, f"pred dim must be >=2, got {pred.shape}"
    C = pred.shape[1]
    spatial = pred.shape[2:]
    assert tuple(label.shape) == (pred.shape[0], *spatial), \
        f"label must be (B, {spatial}), got {label.shape} vs pred {pred.shape}"

    pred_flat = pred.permute(0, *range(2, pred.dim()), 1).contiguous().view(-1, C)
    label_flat = label.contiguous().view(-1)

    if sample_weight is not None:
        assert tuple(sample_weight.shape) == (pred.shape[0], *spatial), \
            f"sample_weight must be (B, {spatial}), got {sample_weight.shape}"
        w_flat = sample_weight.contiguous().view(-1)
    else:
        w_flat = None

    return pred_flat, label_flat, w_flat


def _apply_mask_to_ignore(
    voxel_semantics: torch.Tensor,
    mask_camera: Optional[torch.Tensor],
    ignore_index: int,
    use_mask: bool,
) -> torch.Tensor:
    """
    把不可見區域設成 ignore_index（給 CE / lovasz / sem/geo 去跳過）
    """
    if (not use_mask) or (mask_camera is None):
        return voxel_semantics
    vs = voxel_semantics
    mc = mask_camera.bool()
    # 避免 in-place 影響外部
    vs = vs.clone()
    vs[~mc] = int(ignore_index)
    return vs


# -------------------------
# MM-like CrossEntropyLoss (supports avg_factor + sample_weight + class_weight)
# -------------------------
class CrossEntropyLossMM(nn.Module):
    """
    Minimal MM-aligned CrossEntropyLoss that supports:
      - pred: (N, C) OR (B, C, ...)
      - label: (N,) OR (B, ...)
      - sample_weight: element-wise weight (same shape as label), optional
      - avg_factor: MM-style denominator override
      - ignore_index: passed to F.cross_entropy
      - class_weight: class-wise weights (C,)
      - loss_weight: multiply final loss
    """
    def __init__(
        self,
        use_sigmoid: bool = False,  # kept for config compatibility
        ignore_index: int = 255,
        reduction: str = "mean",    # "mean"/"sum"/"none" (MM-style)
        loss_weight: float = 1.0,
        class_weight: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        super().__init__()
        self.use_sigmoid = bool(use_sigmoid)
        self.ignore_index = int(ignore_index)
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction
        self.loss_weight = float(loss_weight)

        if class_weight is not None:
            cw = torch.as_tensor(class_weight, dtype=torch.float32)
            self.register_buffer("class_weight", cw)
        else:
            self.class_weight = None

        if self.use_sigmoid:
            raise NotImplementedError("This project uses softmax CE (use_sigmoid=False).")

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
        avg_factor: Optional[Union[int, float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        pred_flat, label_flat, w_flat = _flatten_logits_and_labels(pred, label, sample_weight)
        label_flat = label_flat.long()

        # per-element CE
        loss = F.cross_entropy(
            pred_flat,
            label_flat,
            weight=(self.class_weight.to(pred_flat) if self.class_weight is not None else None),
            ignore_index=self.ignore_index,
            reduction="none",
        )  # (M,)

        # apply sample-wise weight (e.g., mask_camera)
        if w_flat is not None:
            w_flat = w_flat.to(loss).float()
            loss = loss * w_flat

        # MM avg_factor
        if avg_factor is None:
            if w_flat is not None:
                denom = w_flat.sum()
            else:
                denom = (label_flat != self.ignore_index).to(loss).sum()
        else:
            denom = avg_factor
            if not torch.is_tensor(denom):
                denom = loss.new_tensor(float(denom))
            else:
                denom = denom.to(loss)

        denom = torch.clamp(denom, min=1.0)

        if self.reduction == "none":
            out = loss
        elif self.reduction == "sum":
            out = loss.sum()
        else:
            out = loss.sum() / denom

        return out * self.loss_weight


def build_loss(cfg: Optional[Dict[str, Any]]) -> nn.Module:
    """
    Tiny replacement for mmdet3d build_loss().

    Example:
      loss_occ=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        ignore_index=255,
        loss_weight=1.0,
        reduction='mean',
        class_weight=... (optional)
      )
    """
    if cfg is None:
        return CrossEntropyLossMM(use_sigmoid=False, ignore_index=255, loss_weight=1.0)

    cfg = dict(cfg)
    loss_type = cfg.pop("type", "CrossEntropyLoss")

    if loss_type in ("CrossEntropyLoss", "CE"):
        return CrossEntropyLossMM(**cfg)

    raise KeyError(f"Unsupported loss type: {loss_type}")


# -------------------------
# Simple Conv + optional Act (MM ConvModule-like minimal)
# -------------------------
class SimpleConvAct2d(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1, bias: bool = True, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class SimpleConvAct3d(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1, bias: bool = True, act: bool = True):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


# -------------------------
# BEVOCCHead3D
# -------------------------
class BEVOccHead3D(nn.Module):
    def __init__(
        self,
        in_dim: int = 32,
        out_dim: int = 32,
        use_mask: bool = True,
        num_classes: int = 18,
        use_predicter: bool = True,
        class_balance: bool = False,
        loss_occ: Optional[Dict[str, Any]] = None,
        final_conv_act: Optional[bool] = None,
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.num_classes = int(num_classes)
        self.use_mask = bool(use_mask)
        self.use_predicter = bool(use_predicter)
        self.class_balance = bool(class_balance)

        # 如果 final_conv 直接輸出 logits（use_predicter=False），建議不要 ReLU
        if final_conv_act is None:
            final_conv_act = bool(self.use_predicter)

        out_channels = self.out_dim if self.use_predicter else self.num_classes
        self.final_conv = SimpleConvAct3d(in_dim, out_channels, k=3, s=1, p=1, bias=True, act=bool(final_conv_act))

        if self.use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, self.num_classes),
            )

        # class balance
        if self.class_balance:
            class_weights = (1.0 / np.log(nusc_class_frequencies[: self.num_classes] + 0.001)).astype(np.float32)
            cw = torch.from_numpy(class_weights)
            self.register_buffer("cls_weights", cw)
            if loss_occ is None:
                loss_occ = dict(type="CrossEntropyLoss", use_sigmoid=False, ignore_index=255, loss_weight=1.0)
            loss_occ = dict(loss_occ)
            loss_occ["class_weight"] = cw
        else:
            self.cls_weights = None

        self.loss_occ = build_loss(loss_occ)

    def forward(self, img_feats: torch.Tensor) -> torch.Tensor:
        """
        img_feats: (B, C, Dz, Dy, Dx)
        return:    (B, Dx, Dy, Dz, n_cls)
        """
        occ_feat = self.final_conv(img_feats)                      # (B, C', Dz, Dy, Dx)
        occ_pred = occ_feat.permute(0, 4, 3, 2, 1).contiguous()    # (B, Dx, Dy, Dz, C')
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)                    # -> last dim = n_cls
        return occ_pred

    def loss(self, occ_pred: torch.Tensor, voxel_semantics: torch.Tensor, mask_camera: Optional[torch.Tensor]):
        """
        occ_pred:        (B, Dx, Dy, Dz, n_cls)
        voxel_semantics: (B, Dx, Dy, Dz)
        mask_camera:     (B, Dx, Dy, Dz) 0/1 or bool
        """
        voxel_semantics = voxel_semantics.long()
        loss = {}

        voxel_flat = voxel_semantics.reshape(-1)
        preds_flat = occ_pred.reshape(-1, self.num_classes)

        if self.use_mask:
            assert mask_camera is not None, "use_mask=True but mask_camera is None"
            mask_flat = mask_camera.reshape(-1).to(torch.float32)

            if self.class_balance:
                valid_voxels = voxel_flat[mask_flat.bool()]
                num_total_samples = preds_flat.new_tensor(0.0)
                for i in range(self.num_classes):
                    num_total_samples += (valid_voxels == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = mask_flat.sum()

            loss_occ = self.loss_occ(preds_flat, voxel_flat, sample_weight=mask_flat, avg_factor=num_total_samples)
        else:
            if self.class_balance:
                num_total_samples = preds_flat.new_tensor(0.0)
                for i in range(self.num_classes):
                    num_total_samples += (voxel_flat == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = preds_flat.new_tensor(float(voxel_flat.numel()))

            loss_occ = self.loss_occ(preds_flat, voxel_flat, sample_weight=None, avg_factor=num_total_samples)

        loss["loss_occ"] = loss_occ
        return loss

    @torch.no_grad()
    def get_occ(self, occ_pred: torch.Tensor, img_metas=None):
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        return list(occ_res.cpu().numpy().astype(np.uint8))


# -------------------------
# BEVOCCHead2D
# -------------------------
class BEVOccHead2D(nn.Module):
    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 256,
        Dz: int = 16,
        use_mask: bool = True,
        num_classes: int = 18,
        use_predicter: bool = True,
        class_balance: bool = False,
        loss_occ: Optional[Dict[str, Any]] = None,
        final_conv_act: Optional[bool] = None,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.Dz = int(Dz)
        self.use_mask = bool(use_mask)
        self.num_classes = int(num_classes)
        self.use_predicter = bool(use_predicter)
        self.class_balance = bool(class_balance)

        if final_conv_act is None:
            final_conv_act = bool(self.use_predicter)

        out_channels = self.out_dim if self.use_predicter else (self.num_classes * self.Dz)
        self.final_conv = SimpleConvAct2d(self.in_dim, out_channels, k=3, s=1, p=1, bias=True, act=bool(final_conv_act))

        if self.use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, self.num_classes * self.Dz),
            )

        if self.class_balance:
            class_weights = (1.0 / np.log(nusc_class_frequencies[: self.num_classes] + 0.001)).astype(np.float32)
            cw = torch.from_numpy(class_weights)
            self.register_buffer("cls_weights", cw)
            if loss_occ is None:
                loss_occ = dict(type="CrossEntropyLoss", use_sigmoid=False, ignore_index=255, loss_weight=1.0)
            loss_occ = dict(loss_occ)
            loss_occ["class_weight"] = cw
        else:
            self.cls_weights = None

        self.loss_occ = build_loss(loss_occ)

    def forward(self, img_feats: torch.Tensor) -> torch.Tensor:
        """
        img_feats: (B, C, Dy, Dx)
        return:    (B, Dx, Dy, Dz, n_cls)
        """
        occ_feat = self.final_conv(img_feats)                  # (B, C', Dy, Dx)
        occ_pred = occ_feat.permute(0, 3, 2, 1).contiguous()   # (B, Dx, Dy, C')
        bs, Dx, Dy = occ_pred.shape[:3]

        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)               # (B, Dx, Dy, Dz*n_cls)
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)
        else:
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)

        return occ_pred

    def loss(self, occ_pred: torch.Tensor, voxel_semantics: torch.Tensor, mask_camera: Optional[torch.Tensor]):
        """
        單一 CE（這就是為什麼常常比 V2 弱很多）
        """
        voxel_semantics = voxel_semantics.long()
        loss = {}

        voxel_flat = voxel_semantics.reshape(-1)
        preds_flat = occ_pred.reshape(-1, self.num_classes)

        if self.use_mask:
            assert mask_camera is not None, "use_mask=True but mask_camera is None"
            mask_flat = mask_camera.reshape(-1).to(torch.float32)

            if self.class_balance:
                valid_voxels = voxel_flat[mask_flat.bool()]
                num_total_samples = preds_flat.new_tensor(0.0)
                for i in range(self.num_classes):
                    num_total_samples += (valid_voxels == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = mask_flat.sum()

            loss_occ = self.loss_occ(preds_flat, voxel_flat, sample_weight=mask_flat, avg_factor=num_total_samples)
        else:
            if self.class_balance:
                num_total_samples = preds_flat.new_tensor(0.0)
                for i in range(self.num_classes):
                    num_total_samples += (voxel_flat == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = preds_flat.new_tensor(float(voxel_flat.numel()))

            loss_occ = self.loss_occ(preds_flat, voxel_flat, sample_weight=None, avg_factor=num_total_samples)

        loss["loss_occ"] = loss_occ
        return loss

    @torch.no_grad()
    def get_occ(self, occ_pred: torch.Tensor, img_metas=None):
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        return list(occ_res.cpu().numpy().astype(np.uint8))

    @torch.no_grad()
    def get_occ_gpu(self, occ_pred: torch.Tensor, img_metas=None):
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1).int()
        return list(occ_res)


# -------------------------
# BEVOccHead2D_V2 (4-loss version)
# -------------------------
class BEVOccHead2D_V2(BEVOccHead2D):
    """
    V2: 4 個 loss（跟你貼的 MM 版本一致）：
      - loss_occ (CE, *100)
      - loss_voxel_sem_scal
      - loss_voxel_geo_scal
      - loss_voxel_lovasz

    重要：這裡補上 mask_camera -> ignore_index，避免不可見 voxel 汙染 supervision
    """
    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 256,
        Dz: int = 16,
        use_mask: bool = True,
        num_classes: int = 18,
        use_predicter: bool = True,
        class_balance: bool = True,                # ✅ V2 建議預設 True
        loss_occ: Optional[Dict[str, Any]] = None,
        empty_idx: int = 17,
        final_conv_act: Optional[bool] = None,
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            Dz=Dz,
            use_mask=use_mask,
            num_classes=num_classes,
            use_predicter=use_predicter,
            class_balance=class_balance,
            loss_occ=loss_occ,
            final_conv_act=final_conv_act,
        )
        self.empty_idx = int(empty_idx)

        # 確保 cls_weights 一定存在（就算你不小心把 class_balance=False）
        if not hasattr(self, "cls_weights") or (self.cls_weights is None):
            class_weights = (1.0 / np.log(nusc_class_frequencies[: self.num_classes] + 0.001)).astype(np.float32)
            cw = torch.from_numpy(class_weights)
            self.register_buffer("cls_weights", cw)

        # 確保 ignore_index 取得到
        self.ignore_index = int(getattr(self.loss_occ, "ignore_index", 255))

    def loss(self, occ_pred: torch.Tensor, voxel_semantics: torch.Tensor, mask_camera: Optional[torch.Tensor]):
        loss = {}
        voxel_semantics = voxel_semantics.long()  # (B, Dx, Dy, Dz)

        # ✅ mask -> ignore
        voxel_semantics = _apply_mask_to_ignore(
            voxel_semantics=voxel_semantics,
            mask_camera=mask_camera,
            ignore_index=self.ignore_index,
            use_mask=self.use_mask,
        )

        # (B, Dx, Dy, Dz, C) -> (B, C, Dx, Dy, Dz)
        preds = occ_pred.permute(0, 4, 1, 2, 3).contiguous()

        # --- CE (class-weight) * 100 ---
        # 直接用 F.cross_entropy，明確控制 class weights + ignore_index
        loss_ce = F.cross_entropy(
            preds,
            voxel_semantics,
            weight=self.cls_weights.to(preds),
            ignore_index=self.ignore_index,
            reduction="mean",
        ) * 100.0
        loss["loss_occ"] = loss_ce

        # --- extra losses (假設你的實作會跳過 ignore_index=255；多數占用 repo 都會) ---
        loss["loss_voxel_sem_scal"] = sem_scal_loss(preds, voxel_semantics)
        loss["loss_voxel_geo_scal"] = geo_scal_loss(preds, voxel_semantics, non_empty_idx=self.empty_idx)
        loss["loss_voxel_lovasz"] = lovasz_softmax(
            torch.softmax(preds, dim=1),
            voxel_semantics,
            ignore=self.ignore_index,
        )


        return loss