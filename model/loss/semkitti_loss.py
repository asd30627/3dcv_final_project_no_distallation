import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast

semantic_kitti_class_frequencies = np.array(
    [
        5.41773033e09, 1.57835390e07, 1.25136000e05, 1.18809000e05, 6.46799000e05,
        8.21951000e05, 2.62978000e05, 2.83696000e05, 2.04750000e05, 6.16887030e07,
        4.50296100e06, 4.48836500e07, 2.26992300e06, 5.68402180e07, 1.57196520e07,
        1.58442623e08, 2.06162300e06, 3.69705220e07, 1.15198800e06, 3.34146000e05,
    ]
)

kitti_class_names = [
    "empty", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person",
    "bicyclist", "motorcyclist", "road", "parking", "sidewalk", "other-ground",
    "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign",
]


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Stable logit(x). Works for scalar tensor safely (no python while-loop)."""
    x = x.to(torch.float32).clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def KL_sep(p: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """KL divergence on nonzeros classes"""
    nonzeros = target != 0
    nonzero_p = p[nonzeros]
    kl_term = F.kl_div(torch.log(nonzero_p), target[nonzeros], reduction="sum")
    return kl_term


def geo_scal_loss(pred: torch.Tensor, ssc_target: torch.Tensor, ignore_index: int = 255, non_empty_idx: int = 0):
    # pred: (B, C, Dx, Dy, Dz), target: (B, Dx, Dy, Dz)
    pred = F.softmax(pred, dim=1)
    empty_probs = pred[:, non_empty_idx]
    nonempty_probs = 1.0 - empty_probs

    mask = ssc_target != ignore_index
    nonempty_target = (ssc_target != non_empty_idx)[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    eps = 1e-5
    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / (nonempty_probs.sum() + eps)
    recall = intersection / (nonempty_target.sum() + eps)
    spec = ((1.0 - nonempty_target) * empty_probs).sum() / ((1.0 - nonempty_target).sum() + eps)

    with autocast(False):
        return (
            F.binary_cross_entropy_with_logits(inverse_sigmoid(precision), torch.ones_like(precision))
            + F.binary_cross_entropy_with_logits(inverse_sigmoid(recall), torch.ones_like(recall))
            + F.binary_cross_entropy_with_logits(inverse_sigmoid(spec), torch.ones_like(spec))
        )


def sem_scal_loss(pred_: torch.Tensor, ssc_target: torch.Tensor, ignore_index: int = 255):
    # pred_: (B, C, Dx, Dy, Dz), target: (B, Dx, Dy, Dz)
    with autocast(False):
        pred = F.softmax(pred_, dim=1)
        mask = ssc_target != ignore_index
        n_classes = pred.shape[1]

        loss = pred_.new_zeros(())
        count = 0.0

        # 你原本寫 n_classes-1，我保留（常見用法是不算最後一類）
        for i in range(0, n_classes - 1):
            p = pred[:, i]  # (B, Dx, Dy, Dz)
            p = p[mask]
            target = ssc_target[mask]

            completion_target = torch.zeros_like(target)
            completion_target[target == i] = 1

            if completion_target.sum() <= 0:
                continue

            count += 1.0
            nominator = torch.sum(p * completion_target)

            loss_class = pred_.new_zeros(())
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p) + 1e-5)
                loss_class += F.binary_cross_entropy_with_logits(inverse_sigmoid(precision), torch.ones_like(precision))

            recall = nominator / (torch.sum(completion_target) + 1e-5)
            loss_class += F.binary_cross_entropy_with_logits(inverse_sigmoid(recall), torch.ones_like(recall))

            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (torch.sum(1 - completion_target) + 1e-5)
                loss_class += F.binary_cross_entropy_with_logits(inverse_sigmoid(specificity), torch.ones_like(specificity))

            loss = loss + loss_class

        if count == 0.0:
            return pred_.sum() * 0.0

        l = loss / count
        return l


def CE_ssc_loss(pred: torch.Tensor, target: torch.Tensor, class_weights=None, ignore_index: int = 255):
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index, reduction="mean")
    with autocast(False):
        return criterion(pred, target.long())


def vel_loss(pred: torch.Tensor, gt: torch.Tensor):
    with autocast(False):
        return F.l1_loss(pred, gt)

