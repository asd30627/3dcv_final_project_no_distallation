# -*- coding:utf-8 -*-
# author: Xinge
"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

from __future__ import print_function, division

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse

from torch.cuda.amp import autocast


def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1.0, ignore=None, per_image=True):
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    return 100 * mean(ious)


def iou(preds, labels, C, EMPTY=1.0, ignore=None, per_image=False):
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou_per = []
        for i in range(C):
            if i != ignore:
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou_per.append(EMPTY)
                else:
                    iou_per.append(float(intersection) / float(union))
        ious.append(iou_per)
    ious = [mean(v) for v in zip(*ious)]
    return 100 * np.array(ious)


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    if per_image:
        return mean(
            lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
            for log, lab in zip(logits, labels)
        )
    return lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))


def lovasz_hinge_flat(logits, labels):
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = (1.0 - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), Variable(grad))


def flatten_binary_scores(scores, labels, ignore=None):
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    return scores[valid], labels[valid]


class StableBCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    return StableBCELoss()(logits, Variable(labels.float()))


def lovasz_softmax(probas, labels, classes="present", per_image=False, ignore=None):
    if per_image:
        return mean(
            lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
            for prob, lab in zip(probas, labels)
        )
    with autocast(False):
        return lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)


def lovasz_softmax_flat(probas, labels, classes="present"):
    if probas.numel() == 0:
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if (classes == "present") and fg.sum() == 0:
            continue
        class_pred = probas[:, 0] if C == 1 else probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    if probas.dim() == 2:
        if ignore is not None:
            valid = labels != ignore
            probas = probas[valid]
            labels = labels[valid]
        return probas, labels

    if probas.dim() == 3:
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    elif probas.dim() == 5:
        B, C, L, H, W = probas.size()
        probas = probas.contiguous().view(B, C, L, H * W)

    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)

    if ignore is None:
        return probas, labels

    valid = labels != ignore
    idx = valid.nonzero(as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return probas[:0], labels[:0]
    return probas[idx], labels[valid]


def xloss(logits, labels, ignore=None):
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


def jaccard_loss(probas, labels, ignore=None, smooth=100, bk_class=None):
    vprobas, vlabels = flatten_probas(probas, labels, ignore)
    true_1_hot = torch.eye(vprobas.shape[1], device=vprobas.device)[vlabels]

    if bk_class is not None:
        one_hot_assignment = torch.ones_like(vlabels, dtype=torch.float32, device=vprobas.device)
        one_hot_assignment[vlabels == bk_class] = 0.0
        one_hot_assignment = one_hot_assignment.unsqueeze(1)
        true_1_hot = true_1_hot * one_hot_assignment

    intersection = torch.sum(vprobas * true_1_hot)
    cardinality = torch.sum(vprobas + true_1_hot)
    loss = (intersection + smooth) / (cardinality - intersection + smooth)
    return (1.0 - loss) * smooth


def hinge_jaccard_loss(probas, labels, ignore=None, classes="present", hinge=0.1, smooth=100):
    vprobas, vlabels = flatten_probas(probas, labels, ignore)
    C = vprobas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
    for c in class_to_sum:
        if (vlabels == c).any():
            c_sample_ind = vlabels == c
            cprobas = vprobas[c_sample_ind, :]
            non_c_ind = np.array([a for a in class_to_sum if a != c])
            class_pred = cprobas[:, c]
            max_non_class_pred = torch.max(cprobas[:, non_c_ind], dim=1)[0]
            TP = torch.sum(torch.clamp(class_pred - max_non_class_pred, max=hinge) + 1.0) + smooth
            FN = torch.sum(torch.clamp(max_non_class_pred - class_pred, min=-hinge) + hinge)

            if (~c_sample_ind).sum() == 0:
                FP = 0.0
            else:
                nonc_probas = vprobas[~c_sample_ind, :]
                class_pred = nonc_probas[:, c]
                max_non_class_pred = torch.max(nonc_probas[:, non_c_ind], dim=1)[0]
                FP = torch.sum(torch.clamp(class_pred - max_non_class_pred, max=hinge) + 1.0)

            losses.append(1.0 - TP / (TP + FP + FN))

    if len(losses) == 0:
        return vprobas.sum() * 0.0
    return mean(losses)


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    return acc if n == 1 else acc / n

