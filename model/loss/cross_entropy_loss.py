import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


def reduce_loss(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Invalid reduction: {reduction}")


def weight_reduce_loss(
    loss: torch.Tensor,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
    avg_factor: float | None = None,
) -> torch.Tensor:
    # loss: element-wise loss
    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        return reduce_loss(loss, reduction)

    # avg_factor is only meaningful for mean-like reduction
    if reduction == "mean":
        eps = 1e-12
        return loss.sum() / max(float(avg_factor), eps)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Invalid reduction: {reduction}")


def cross_entropy(
    pred: torch.Tensor,
    label: torch.Tensor,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
    avg_factor: float | None = None,
    class_weight: torch.Tensor | None = None,
    ignore_index: int | None = -100,
    avg_non_ignore: bool = False,
) -> torch.Tensor:
    ignore_index = -100 if ignore_index is None else ignore_index

    # element-wise loss
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction="none",
        ignore_index=ignore_index,
    )

    if (avg_factor is None) and avg_non_ignore and reduction == "mean":
        avg_factor = label.numel() - (label == ignore_index).sum().item()

    if weight is not None:
        weight = weight.float()

    return weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)


def _expand_onehot_labels(labels, label_weights, label_channels, ignore_index):
    # labels: (N,)
    bin_labels = labels.new_zeros((labels.size(0), label_channels))
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask & (labels < label_channels), as_tuple=False).squeeze(1)
    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    valid_mask_f = valid_mask.view(-1, 1).expand(labels.size(0), label_channels).float()
    if label_weights is None:
        bin_label_weights = valid_mask_f
    else:
        bin_label_weights = label_weights.view(-1, 1).repeat(1, label_channels)
        bin_label_weights = bin_label_weights * valid_mask_f

    return bin_labels, bin_label_weights, valid_mask_f


def binary_cross_entropy(
    pred: torch.Tensor,
    label: torch.Tensor,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
    avg_factor: float | None = None,
    class_weight: torch.Tensor | None = None,
    ignore_index: int | None = -100,
    avg_non_ignore: bool = False,
) -> torch.Tensor:
    ignore_index = -100 if ignore_index is None else ignore_index

    if pred.dim() != label.dim():
        label, weight, valid_mask = _expand_onehot_labels(label, weight, pred.size(-1), ignore_index)
    else:
        valid_mask = ((label >= 0) & (label != ignore_index)).float()
        if weight is not None:
            weight = weight * valid_mask
        else:
            weight = valid_mask

    if (avg_factor is None) and avg_non_ignore and reduction == "mean":
        avg_factor = valid_mask.sum().item()

    weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction="none"
    )
    return weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)


def mask_cross_entropy(
    pred: torch.Tensor,
    target: torch.Tensor,
    label: torch.Tensor,
    reduction: str = "mean",
    avg_factor: float | None = None,
    class_weight: torch.Tensor | None = None,
    ignore_index=None,
    **kwargs,
) -> torch.Tensor:
    assert ignore_index is None, "BCE loss does not support ignore_index"
    assert reduction == "mean" and avg_factor is None
    num_rois = pred.size(0)
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(pred_slice, target, weight=class_weight, reduction="mean")[None]


class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        use_sigmoid: bool = False,
        use_mask: bool = False,
        reduction: str = "mean",
        class_weight=None,
        ignore_index=None,
        loss_weight: float = 1.0,
        avg_non_ignore: bool = False,
    ):
        super().__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore

        if ((ignore_index is not None) and (not self.avg_non_ignore) and self.reduction == "mean"):
            warnings.warn(
                "Default avg_non_ignore=False. If you want to ignore labels and average only over non-ignored "
                "targets (same as PyTorch official cross_entropy), set avg_non_ignore=True."
            )

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def extra_repr(self) -> str:
        return f"avg_non_ignore={self.avg_non_ignore}"

    def forward(
        self,
        cls_score: torch.Tensor,
        label: torch.Tensor,
        weight: torch.Tensor | None = None,
        avg_factor: float | None = None,
        reduction_override: str | None = None,
        ignore_index: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if ignore_index is None:
            ignore_index = self.ignore_index

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight, device=cls_score.device)
        else:
            class_weight = None

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index,
            avg_non_ignore=self.avg_non_ignore,
            **kwargs,
        )
        return loss_cls

