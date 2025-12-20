import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_one_hot(target: torch.Tensor, num_classes: int, dtype: torch.dtype) -> torch.Tensor:
    # target: (N,) int -> (N,C) float
    oh = F.one_hot(target.long(), num_classes=num_classes)
    return oh.to(dtype=dtype)


def sigmoid_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = "mean",
    avg_factor=None,
):
    """
    Pure PyTorch sigmoid focal loss.

    pred:   (N, C) logits
    target: (N,) class index  OR  (N, C) one-hot
    weight: broadcastable to (N, C) or (N,) or (C,)
    """
    assert pred.dim() == 2, f"pred must be (N,C), got {pred.shape}"

    if target.dim() == 1:
        target_oh = _to_one_hot(target, pred.size(1), pred.dtype)
    else:
        target_oh = target.to(dtype=pred.dtype)

    bce = F.binary_cross_entropy_with_logits(pred, target_oh, reduction="none")  # (N,C)
    prob = torch.sigmoid(pred)
    p_t = prob * target_oh + (1.0 - prob) * (1.0 - target_oh)
    alpha_factor = alpha * target_oh + (1.0 - alpha) * (1.0 - target_oh)
    loss = bce * alpha_factor * (1.0 - p_t).pow(gamma)

    if weight is not None:
        # allow weight shape: (C,), (N,), (N,C)
        if weight.shape != loss.shape:
            if weight.numel() == pred.size(1):
                weight = weight.view(1, -1).expand_as(loss)
            elif weight.numel() == pred.size(0):
                weight = weight.view(-1, 1).expand_as(loss)
            else:
                weight = weight.view_as(loss)
        loss = loss * weight

    loss = loss.sum(dim=1)  # (N,)

    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Invalid reduction: {reduction}")


def focal_loss_with_prob(
    prob: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = "mean",
    avg_factor=None,
):
    """
    Focal loss where input is probability (already sigmoid-ed), not logits.

    prob:   (N, C) in [0,1]
    target: (N,) class index OR (N,C) one-hot
    """
    assert prob.dim() == 2, f"prob must be (N,C), got {prob.shape}"

    if target.dim() == 1:
        target_oh = _to_one_hot(target, prob.size(1), prob.dtype)
    else:
        target_oh = target.to(dtype=prob.dtype)

    bce = F.binary_cross_entropy(prob, target_oh, reduction="none")  # (N,C)
    p_t = prob * target_oh + (1.0 - prob) * (1.0 - target_oh)
    alpha_factor = alpha * target_oh + (1.0 - alpha) * (1.0 - target_oh)
    loss = bce * alpha_factor * (1.0 - p_t).pow(gamma)

    if weight is not None:
        if weight.shape != loss.shape:
            if weight.numel() == prob.size(1):
                weight = weight.view(1, -1).expand_as(loss)
            elif weight.numel() == prob.size(0):
                weight = weight.view(-1, 1).expand_as(loss)
            else:
                weight = weight.view_as(loss)
        loss = loss * weight

    loss = loss.sum(dim=1)  # (N,)

    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Invalid reduction: {reduction}")


class CustomFocalLoss(nn.Module):
    """
    原本是 mmdet registry + mmcv op 版本。這裡改成純 PyTorch。
    也把 self.c 的 cuda() 改成 register_buffer + 動態依 target H/W 生成。
    """

    def __init__(
        self,
        use_sigmoid: bool = True,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
        loss_weight: float = 100.0,
        activated: bool = False,
    ):
        super().__init__()
        assert use_sigmoid is True, "Only sigmoid focal loss is supported."
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

        # cache for radial weighting map c (H,W)
        self.register_buffer("_c_map", torch.empty(0), persistent=False)
        self._c_hw = None  # tuple(H,W)

    def _get_c(self, H: int, W: int, device, dtype):
        if self._c_hw != (H, W) or self._c_map.numel() == 0 or self._c_map.device != device:
            yy = torch.arange(H, device=device, dtype=torch.float32) - (H / 2.0)
            xx = torch.arange(W, device=device, dtype=torch.float32) - (W / 2.0)
            grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
            c = torch.stack([grid_y, grid_x], dim=-1)  # (H,W,2)
            c = torch.norm(c, p=2, dim=-1)  # (H,W)
            c_max = c.max().clamp(min=1e-12)
            c = (c / c_max + 1.0)  # (H,W) in [1,2]
            self._c_map = c
            self._c_hw = (H, W)
        return self._c_map.to(dtype=dtype)

    def forward(
        self,
        pred: torch.Tensor,          # (B, C, H, W, D)
        target: torch.Tensor,        # (B, H, W, D) int labels
        weight: torch.Tensor | None = None,  # usually (C,) class weights
        avg_factor=None,
        ignore_index: int = 255,
        reduction_override: str | None = None,
    ):
        assert pred.dim() == 5, f"pred should be (B,C,H,W,D), got {pred.shape}"
        assert target.dim() == 4, f"target should be (B,H,W,D), got {target.shape}"

        B, C, H, W, D = pred.shape
        device = pred.device
        dtype = pred.dtype

        visible = (target != ignore_index).reshape(-1)
        if visible.sum() == 0:
            return pred.sum() * 0.0

        visible_idx = visible.nonzero(as_tuple=False).squeeze(1)  # (Nvis,)

        # radial weight: (Nvis,)
        c_map = self._get_c(H, W, device=device, dtype=torch.float32)  # keep stable float32
        c_flat = c_map[None, :, :, None].repeat(B, 1, 1, D).reshape(-1)  # (B*H*W*D,)
        c_vis = c_flat[visible_idx]  # (Nvis,)

        # reshape logits & labels
        pred_flat = pred.permute(0, 2, 3, 4, 1).reshape(-1, C)[visible_idx]  # (Nvis,C)
        tgt_flat = target.reshape(-1)[visible_idx].long()                    # (Nvis,)

        # build weight mask to (Nvis,C)
        weight_mask = None
        if weight is not None:
            if weight.numel() != C:
                raise ValueError(f"weight should be (C,), got {tuple(weight.shape)} but C={C}")
            weight_mask = (weight.view(1, C) * c_vis.view(-1, 1)).to(device=device, dtype=pred_flat.dtype)
        else:
            # 只用 radial 權重也可以
            weight_mask = c_vis.view(-1, 1).to(device=device, dtype=pred_flat.dtype)

        reduction = reduction_override if reduction_override else self.reduction

        if self.activated:
            # pred_flat is probability in [0,1]
            loss = focal_loss_with_prob(
                pred_flat,
                tgt_flat,
                weight=weight_mask,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor,
            )
        else:
            # pred_flat is logits
            loss = sigmoid_focal_loss(
                pred_flat,
                tgt_flat,
                weight=weight_mask,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor,
            )

        return self.loss_weight * loss

