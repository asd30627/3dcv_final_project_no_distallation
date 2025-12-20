from __future__ import annotations

"""
eval_occ_metrics_gfstyle_for_student.py

目的
- 沿用你貼的 GaussianFormer/FlashOcc-style 評估邏輯：
  * confusion matrix
  * exclude free in mean
  * camera mask (occ_cam_mask)
  * 強力 sanity check：missing/unexpected、gt/pred 分佈、mask 覆蓋率、H/W swap test
- 但把 pred 取得方式改成「跟你 vis_student.py 一樣」：
  * 自動辨識 logits shape -> (B,H,W,D) semantics
  * 避免評估時 class_dim/permute 對不齊導致 mIoU 假爛

額外
- 可選：RayIoU@{1,2,4}m（可重現版本，取每條 ray 第一個撞到 occupied voxel 的 (class, depth)）
  * 注意：若你有官方/資料集提供的 rays 取樣方式，可再把 dirs 換掉

用法（先跑 mIoU）
python eval_occ_metrics_gfstyle_for_student.py \
  --py-config <cfg.py> \
  --ckpt <latest.pth> \
  --use-ema \
  --free-id 17 \
  --ignore-id 255 \
  --max-batches 200 \
  --debug-first-batches 2

用法（加 RayIoU）
python eval_occ_metrics_gfstyle_for_student.py \
  --py-config <cfg.py> \
  --ckpt <latest.pth> \
  --use-ema \
  --ray \
  --ray-thrs 1,2,4
"""

import argparse
import itertools
import json
import numpy as np
import torch
from tqdm import tqdm
from mmengine import Config

from model.student import OccStudent
from dataset import get_dataloader


# ----------------------------
# Batch -> device（沿用你貼的版本，支援 np.object/list/tuple）
# ----------------------------
def to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, np.ndarray):
        if batch.dtype == object:
            batch = np.stack([np.asarray(x, dtype=np.float32) for x in list(batch)], axis=0)
        return torch.from_numpy(batch).to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(x, device) for x in batch)
    return batch


# ----------------------------
# confusion matrix（沿用你貼的版本）
# ----------------------------
@torch.no_grad()
def update_confusion_matrix(cm: torch.Tensor, pred: torch.Tensor, gt: torch.Tensor, K: int):
    idx = gt * K + pred
    binc = torch.bincount(idx, minlength=K * K)
    cm += binc.reshape(K, K)
    return cm


def compute_iou_from_cm(cm: torch.Tensor, eps: float = 1e-6):
    cm = cm.to(torch.float64)
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    iou = tp / (tp + fp + fn + eps)
    return iou, tp, fp, fn


def _topk_hist(x: torch.Tensor, K: int, topk: int = 10):
    if x.numel() == 0:
        return [], torch.zeros((K,), dtype=torch.int64)
    h = torch.bincount(x, minlength=K).to(torch.int64)
    vals, idxs = torch.topk(h, k=min(topk, K))
    out = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        if v > 0:
            out.append((i, v))
    return out, h


# ----------------------------
# build model（盡量對齊你 vis_student.py 的 hardcode）
# ----------------------------
def build_student_from_cfg(cfg: Config):
    bev_h = int(getattr(cfg, "bev_h", 200)) if hasattr(cfg, "bev_h") else 200
    bev_w = int(getattr(cfg, "bev_w", 200)) if hasattr(cfg, "bev_w") else 200
    depth_bins = int(getattr(cfg, "depth_bins", 16)) if hasattr(cfg, "depth_bins") else 16
    num_classes = int(getattr(cfg, "num_classes", 18)) if hasattr(cfg, "num_classes") else 18

    pc_range = None
    if hasattr(cfg, "pc_range"):
        pc_range = cfg.pc_range
    elif hasattr(cfg, "train_dataset_config") and isinstance(cfg.train_dataset_config, dict):
        pc_range = cfg.train_dataset_config.get("pc_range", None)
    if pc_range is None:
        pc_range = (-50.0, -50.0, -5.0, 50.0, 50.0, 3.0)

    input_size = (480, 640)
    if hasattr(cfg, "input_size"):
        input_size = tuple(cfg.input_size)

    model = OccStudent(
        bev_h=bev_h,
        bev_w=bev_w,
        depth_bins=depth_bins,
        num_classes=num_classes,
        backbone_pretrained=False,
        backbone_frozen_stages=1,
        input_size=input_size,
        numC_Trans=128,
        pc_range=pc_range,
    )
    return model


# ----------------------------
# ckpt load（沿用你貼的版本：ema_state/model_state/state_dict）
# ----------------------------
def load_ckpt_into_model(model: torch.nn.Module, ckpt_path: str, use_ema: bool = False):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if use_ema and "ema_state" in ckpt:
            state = ckpt["ema_state"]
            used = "ema_state"
        elif "model_state" in ckpt:
            state = ckpt["model_state"]
            used = "model_state"
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
            used = "state_dict"
        else:
            state = ckpt
            used = "raw_dict"
    else:
        state = ckpt
        used = "raw"
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected, used


# ----------------------------
# logits -> semantics (B,H,W,D)（改成你 vis_student.py 的推斷思路）
# ----------------------------
def logits_to_semantics_bhwd(logits: torch.Tensor, K: int, Dz: int) -> torch.Tensor:
    """
    支援常見 shape：
      1) (B, K, Dz, H, W)
      2) (B, Dz, K, H, W)
      3) (B, K*Dz, H, W)
      4) (B, Dz, H, W, K)
      5) (B, H, W, Dz, K)
    回傳 (B,H,W,Dz) int64
    """
    x = logits.detach()

    # already semantics?
    if x.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8):
        # accept (B,H,W,D) or (B,D,H,W)
        if x.ndim == 4:
            if x.shape[-1] == Dz:
                return x.to(torch.int64)
            if x.shape[1] == Dz:
                return x.permute(0, 2, 3, 1).contiguous().to(torch.int64)
        raise ValueError(f"Integer output but unrecognized shape: {tuple(x.shape)}")

    if x.ndim == 5:
        # (B,K,D,H,W)
        if x.shape[1] == K and x.shape[2] == Dz:
            x = x.permute(0, 3, 4, 2, 1)  # (B,H,W,D,K)
        # (B,D,K,H,W)
        elif x.shape[1] == Dz and x.shape[2] == K:
            x = x.permute(0, 3, 4, 1, 2)  # (B,H,W,D,K)
        # (B,D,H,W,K)
        elif x.shape[1] == Dz and x.shape[-1] == K:
            x = x.permute(0, 2, 3, 1, 4)  # (B,H,W,D,K)
        # (B,H,W,D,K)
        elif x.shape[-1] == K and x.shape[-2] == Dz:
            pass
        else:
            raise ValueError(f"Unrecognized 5D logits shape: {tuple(x.shape)} (K={K}, Dz={Dz})")

        sem = torch.argmax(x, dim=-1).to(torch.int64)  # (B,H,W,D)
        return sem

    if x.ndim == 4:
        B, CD, H, W = x.shape
        if CD == K * Dz:
            x = x.view(B, K, Dz, H, W).permute(0, 3, 4, 2, 1)  # (B,H,W,D,K)
            sem = torch.argmax(x, dim=-1).to(torch.int64)
            return sem
        raise ValueError(f"Unrecognized 4D logits shape: {tuple(x.shape)} (expect K*Dz={K*Dz})")

    raise ValueError(f"Unsupported logits dim={x.ndim} shape={tuple(x.shape)}")


def ensure_gt_bhwd(gt: torch.Tensor, Dz: int) -> torch.Tensor:
    """
    GT 常見：
      (B,H,W,D) 或 (B,D,H,W)
    """
    x = gt
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.ndim != 4:
        raise ValueError(f"GT must be 4D, got {tuple(x.shape)}")
    if x.shape[-1] == Dz:
        return x.to(torch.int64)
    if x.shape[1] == Dz:
        return x.permute(0, 2, 3, 1).contiguous().to(torch.int64)
    raise ValueError(f"Cannot infer GT layout: {tuple(x.shape)} (Dz={Dz})")


# ----------------------------
# RayIoU: simple reproducible implementation
# ----------------------------
def make_ray_dirs(pc_range, n_az: int, n_el: int, origin_z: float):
    x_min, y_min, z_min, x_max, y_max, z_max = [float(v) for v in pc_range]
    rmax_xy = float(np.sqrt(max(abs(x_min), abs(x_max))**2 + max(abs(y_min), abs(y_max))**2))
    el_min = np.arctan2((z_min - origin_z), rmax_xy + 1e-6)
    el_max = np.arctan2((z_max - origin_z), rmax_xy + 1e-6)

    az = torch.linspace(-np.pi, np.pi, steps=n_az, dtype=torch.float32)
    el = torch.linspace(el_min, el_max, steps=n_el, dtype=torch.float32)

    az_grid, el_grid = torch.meshgrid(az, el, indexing="ij")
    azv = az_grid.reshape(-1)
    elv = el_grid.reshape(-1)

    cos_el = torch.cos(elv)
    dx = cos_el * torch.cos(azv)
    dy = cos_el * torch.sin(azv)
    dz = torch.sin(elv)
    return torch.stack([dx, dy, dz], dim=1)  # (R,3)


@torch.no_grad()
def ray_cast_first_hit(sem_bhwd: torch.Tensor,
                       pc_range,
                       free_id: int,
                       origin=(0.0, 0.0, 0.0),
                       dirs: torch.Tensor | None = None,
                       step: float = 0.25,
                       chunk_rays: int = 4096):
    """
    sem_bhwd: (B,H,W,D) int64 on GPU
    return hit_cls, hit_depth:
      hit_cls: (B,R) int64
      hit_depth: (B,R) float32
    """
    if dirs is None:
        raise ValueError("dirs must be provided")

    x_min, y_min, z_min, x_max, y_max, z_max = [float(v) for v in pc_range]
    B, H, W, D = sem_bhwd.shape

    vx = (x_max - x_min) / H
    vy = (y_max - y_min) / W
    vz = (z_max - z_min) / D

    max_t = float(np.linalg.norm([
        max(abs(x_min), abs(x_max)),
        max(abs(y_min), abs(y_max)),
        max(abs(z_min - origin[2]), abs(z_max - origin[2]))
    ])) + 1.0

    device = sem_bhwd.device
    t = torch.arange(0.0, max_t, step=step, dtype=torch.float32, device=device)  # (S,)
    dirs = dirs.to(device)
    R = dirs.shape[0]

    ox, oy, oz = [float(v) for v in origin]
    origin_t = torch.tensor([ox, oy, oz], dtype=torch.float32, device=device)

    hit_cls = torch.full((B, R), free_id, dtype=torch.int64, device=device)
    hit_dep = torch.full((B, R), max_t + 1e6, dtype=torch.float32, device=device)

    for b in range(B):
        sem = sem_bhwd[b]  # (H,W,D)
        for s in range(0, R, chunk_rays):
            e = min(R, s + chunk_rays)
            d = dirs[s:e]  # (r,3)
            r = d.shape[0]

            pts = origin_t[None, None, :] + d[:, None, :] * t[None, :, None]  # (r,S,3)
            px, py, pz = pts[..., 0], pts[..., 1], pts[..., 2]

            ix = torch.floor((px - x_min) / vx).to(torch.int64)
            iy = torch.floor((py - y_min) / vy).to(torch.int64)
            iz = torch.floor((pz - z_min) / vz).to(torch.int64)

            inside = (ix >= 0) & (ix < H) & (iy >= 0) & (iy < W) & (iz >= 0) & (iz < D)

            vals = torch.full((r, t.shape[0]), free_id, dtype=torch.int64, device=device)
            if inside.any():
                ii = ix[inside]
                jj = iy[inside]
                kk = iz[inside]
                vals[inside] = sem[ii, jj, kk]

            occ = (vals != free_id) & inside
            hit_any = occ.any(dim=1)
            first = occ.to(torch.int8).argmax(dim=1)
            depth = t[first]
            cls = vals[torch.arange(r, device=device), first]

            cls = torch.where(hit_any, cls, torch.full_like(cls, free_id))
            depth = torch.where(hit_any, depth, torch.full_like(depth, max_t + 1e6))

            hit_cls[b, s:e] = cls
            hit_dep[b, s:e] = depth

    return hit_cls, hit_dep


def update_ray_stats(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor,
                     pred_cls: torch.Tensor, pred_dep: torch.Tensor,
                     gt_cls: torch.Tensor, gt_dep: torch.Tensor,
                     thr: float, K: int):
    """
    pred/gt: (B,R)
    """
    match = (pred_cls == gt_cls) & (torch.abs(pred_dep - gt_dep) <= thr)
    for c in range(K):
        gt_is = (gt_cls == c)
        pr_is = (pred_cls == c)
        tp_c = (match & gt_is).sum().item()
        fp_c = pr_is.sum().item() - tp_c
        fn_c = gt_is.sum().item() - tp_c
        tp[c] += int(tp_c)
        fp[c] += int(fp_c)
        fn[c] += int(fn_c)


def compute_ray_iou(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, exclude_ids: set[int] | None = None):
    tp = tp.float()
    fp = fp.float()
    fn = fn.float()
    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom, torch.zeros_like(denom))
    valid = denom > 0
    if exclude_ids:
        for k in exclude_ids:
            if 0 <= k < iou.numel():
                valid[k] = False
    return iou, (iou[valid].mean().item() if valid.any() else 0.0)


# ----------------------------
# eval loop
# ----------------------------
@torch.no_grad()
def eval_metrics_debug(
    model,
    val_loader,
    device,
    K: int,
    Dz: int,
    free_id: int,
    ignore_ids: tuple[int, ...],
    gt_key: str,
    mask_key: str | None,
    exclude_free_in_mean: bool,
    also_occupied_only_miou: bool,
    max_batches: int | None,
    debug_first_batches: int,
    amp: bool,
    # ray
    ray: bool,
    ray_thrs: list[float],
    ray_n_az: int,
    ray_n_el: int,
    ray_step: float,
    ray_origin_z: float,
    ray_chunk: int,
):
    model.eval()

    cm_all = torch.zeros((K, K), dtype=torch.int64)
    cm_occ_only = torch.zeros((K, K), dtype=torch.int64) if also_occupied_only_miou else None

    occ_tp = 0
    occ_fp = 0
    occ_fn = 0

    pc_range = getattr(model, "pc_range", (-50.0, -50.0, -5.0, 50.0, 50.0, 3.0))

    # ray stats
    exclude_ids = {free_id} if exclude_free_in_mean else set()
    dirs = None
    ray_stats = None
    if ray:
        dirs = make_ray_dirs(pc_range, n_az=ray_n_az, n_el=ray_n_el, origin_z=ray_origin_z)
        ray_stats = {thr: {"tp": torch.zeros((K,), dtype=torch.int64),
                           "fp": torch.zeros((K,), dtype=torch.int64),
                           "fn": torch.zeros((K,), dtype=torch.int64)} for thr in ray_thrs}

    pbar = tqdm(val_loader, desc="Eval(GF-style for Student)", leave=True)

    for bi, batch in enumerate(pbar):
        if max_batches is not None and bi >= max_batches:
            break

        if bi == 0 and debug_first_batches > 0:
            print("[DBG] batch keys:", sorted(list(batch.keys())))

        batch = to_device(batch, device)

        # forward
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(batch)
        logits = out["logits"] if (isinstance(out, dict) and "logits" in out) else out

        # pred semantics: (B,H,W,D)
        pred = logits_to_semantics_bhwd(logits, K=K, Dz=Dz)

        # gt: (B,H,W,D)
        if gt_key not in batch:
            raise KeyError(f"Cannot find gt_key='{gt_key}' in batch. Keys={sorted(list(batch.keys()))}")
        gt = ensure_gt_bhwd(batch[gt_key], Dz=Dz).to(device)

        # mask
        mk = None
        if mask_key is not None and mask_key in batch and batch[mask_key] is not None:
            mk = batch[mask_key]
            if not torch.is_tensor(mk):
                mk = torch.as_tensor(mk)
            # accept (B,H,W,D) or (B,D,H,W)
            if mk.ndim == 4 and mk.shape[-1] == Dz:
                mk = mk.bool()
            elif mk.ndim == 4 and mk.shape[1] == Dz:
                mk = mk.permute(0, 2, 3, 1).contiguous().bool()
            else:
                mk = None

        # debug prints
        if bi < debug_first_batches:
            print("\n[DBG] ------------------------------")
            print(f"[DBG] batch={bi}")
            print(f"[DBG] logits shape={tuple(logits.shape)} dtype={logits.dtype}")
            print(f"[DBG] pred  shape={tuple(pred.shape)} dtype={pred.dtype}")
            print(f"[DBG] gt    shape={tuple(gt.shape)} dtype={gt.dtype} unique(min,max)={(int(gt.min().item()), int(gt.max().item()))}")
            if mk is not None:
                print(f"[DBG] mask  key={mask_key} shape={tuple(mk.shape)} dtype={mk.dtype} cov={float(mk.float().mean().item()):.4f}")
            else:
                print(f"[DBG] mask  disabled/missing: {mask_key}")

        # flatten with mask
        if mk is not None:
            pred_f = pred[mk].reshape(-1)
            gt_f = gt[mk].reshape(-1)
        else:
            pred_f = pred.reshape(-1)
            gt_f = gt.reshape(-1)

        valid = (gt_f >= 0) & (gt_f < K)
        for ig in ignore_ids:
            valid = valid & (gt_f != ig)

        pred_f = pred_f[valid].to(torch.int64).detach().cpu()
        gt_f = gt_f[valid].to(torch.int64).detach().cpu()

        if bi < debug_first_batches:
            top_gt, _ = _topk_hist(gt_f, K, topk=10)
            top_pr, _ = _topk_hist(pred_f, K, topk=10)
            free_ratio_gt = float((gt_f == free_id).float().mean().item()) if gt_f.numel() else 0.0
            free_ratio_pr = float((pred_f == free_id).float().mean().item()) if pred_f.numel() else 0.0
            print(f"[DBG] valid_voxels={gt_f.numel()}  free_id={free_id}")
            print(f"[DBG] gt  free_ratio={free_ratio_gt:.4f}  top={top_gt}")
            print(f"[DBG] pred free_ratio={free_ratio_pr:.4f}  top={top_pr}")

        update_confusion_matrix(cm_all, pred_f, gt_f, K)

        if cm_occ_only is not None:
            occ_mask = (gt_f != free_id)
            if occ_mask.any():
                update_confusion_matrix(cm_occ_only, pred_f[occ_mask], gt_f[occ_mask], K)

        gt_occ = (gt_f != free_id)
        pred_occ = (pred_f != free_id)
        occ_tp += int((gt_occ & pred_occ).sum().item())
        occ_fp += int((~gt_occ & pred_occ).sum().item())
        occ_fn += int((gt_occ & ~pred_occ).sum().item())

        # running miou
        iou_all, *_ = compute_iou_from_cm(cm_all)
        if exclude_free_in_mean:
            used = [i for i in range(K) if i != free_id]
            miou_run = float(iou_all[used].mean().cpu()) if len(used) > 0 else float("nan")
        else:
            miou_run = float(iou_all.mean().cpu())
        pbar.set_postfix({"mIoU": f"{miou_run:.4f}", "mIoU%": f"{miou_run*100:.2f}"})

        # axis sanity: try H/W swap + (optional) best perm on batch0
        if bi == 0 and debug_first_batches > 0:
            pred0 = pred[0]
            gt0 = gt[0]
            # H/W swap test
            if pred0.shape[0] == pred0.shape[1]:
                pr_hw = pred0.transpose(0, 1).contiguous()
                gt_hw = gt0
                if mk is not None:
                    m0 = mk[0]
                    pr2 = pr_hw[m0].reshape(-1).cpu()
                    gt2 = gt_hw[m0].reshape(-1).cpu()
                else:
                    pr2 = pr_hw.reshape(-1).cpu()
                    gt2 = gt_hw.reshape(-1).cpu()
                valid2 = (gt2 >= 0) & (gt2 < K)
                for ig in ignore_ids:
                    valid2 = valid2 & (gt2 != ig)
                pr2 = pr2[valid2].to(torch.int64)
                gt2 = gt2[valid2].to(torch.int64)
                cm2 = torch.zeros((K, K), dtype=torch.int64)
                update_confusion_matrix(cm2, pr2, gt2, K)
                iou2, *_ = compute_iou_from_cm(cm2)
                if exclude_free_in_mean:
                    used = [i for i in range(K) if i != free_id]
                    miou2 = float(iou2[used].mean().cpu()) if len(used) > 0 else float("nan")
                else:
                    miou2 = float(iou2.mean().cpu())
                print(f"[DBG] H/W swap test (batch0): mIoU={miou2:.4f} ({miou2*100:.2f}%)  vs running={miou_run:.4f} ({miou_run*100:.2f}%)")
                print("[DBG] 若 swap 後暴增，代表 pred/gt 的 H/W 對齊錯了。")

            # best-permutation test (only when perm keeps same shape)
            best = (miou_run, (0, 1, 2))
            for perm in itertools.permutations([0, 1, 2], 3):
                prp = pred0.permute(*perm).contiguous()
                if prp.shape != gt0.shape:
                    continue
                if mk is not None:
                    m0 = mk[0]
                    prp_f = prp[m0].reshape(-1).cpu()
                    gt_f2 = gt0[m0].reshape(-1).cpu()
                else:
                    prp_f = prp.reshape(-1).cpu()
                    gt_f2 = gt0.reshape(-1).cpu()
                validp = (gt_f2 >= 0) & (gt_f2 < K)
                for ig in ignore_ids:
                    validp = validp & (gt_f2 != ig)
                prp_f = prp_f[validp].to(torch.int64)
                gt_f2 = gt_f2[validp].to(torch.int64)
                cm_p = torch.zeros((K, K), dtype=torch.int64)
                update_confusion_matrix(cm_p, prp_f, gt_f2, K)
                iou_p, *_ = compute_iou_from_cm(cm_p)
                if exclude_free_in_mean:
                    used = [i for i in range(K) if i != free_id]
                    miou_p = float(iou_p[used].mean().cpu()) if len(used) > 0 else float("nan")
                else:
                    miou_p = float(iou_p.mean().cpu())
                if miou_p > best[0]:
                    best = (miou_p, perm)
            print(f"[DBG] best perm (batch0, shape-matched only): perm={best[1]}  mIoU={best[0]:.4f} ({best[0]*100:.2f}%)")

        # RayIoU
        if ray:
            pred_cls, pred_dep = ray_cast_first_hit(
                pred, pc_range=pc_range, free_id=free_id,
                origin=(0.0, 0.0, float(ray_origin_z)),
                dirs=dirs, step=float(ray_step), chunk_rays=int(ray_chunk)
            )
            gt_cls, gt_dep = ray_cast_first_hit(
                gt, pc_range=pc_range, free_id=free_id,
                origin=(0.0, 0.0, float(ray_origin_z)),
                dirs=dirs, step=float(ray_step), chunk_rays=int(ray_chunk)
            )
            for thr in ray_thrs:
                st = ray_stats[thr]
                update_ray_stats(st["tp"], st["fp"], st["fn"],
                                 pred_cls, pred_dep, gt_cls, gt_dep,
                                 thr=float(thr), K=K)

    # final metrics
    iou_all, tp_all, fp_all, fn_all = compute_iou_from_cm(cm_all)
    if exclude_free_in_mean:
        used_classes = [i for i in range(K) if i != free_id]
        miou = float(iou_all[used_classes].mean().cpu()) if len(used_classes) > 0 else float("nan")
    else:
        used_classes = list(range(K))
        miou = float(iou_all.mean().cpu())

    miou_occ_only = None
    if cm_occ_only is not None:
        iou2, *_ = compute_iou_from_cm(cm_occ_only)
        if exclude_free_in_mean:
            used2 = [i for i in range(K) if i != free_id]
            miou_occ_only = float(iou2[used2].mean().cpu()) if len(used2) > 0 else float("nan")
        else:
            miou_occ_only = float(iou2.mean().cpu())

    denom = occ_tp + occ_fp + occ_fn
    occ_iou = float(occ_tp / denom) if denom > 0 else float("nan")

    out = {
        "semantic_miou": miou,
        "semantic_iou_per_class": iou_all.cpu().numpy().tolist(),
        "semantic_used_classes": used_classes,
        "tp": tp_all.cpu().numpy().tolist(),
        "fp": fp_all.cpu().numpy().tolist(),
        "fn": fn_all.cpu().numpy().tolist(),
        "occ_iou": occ_iou,
        "occ_tp": int(occ_tp),
        "occ_fp": int(occ_fp),
        "occ_fn": int(occ_fn),
        "free_id": int(free_id),
        "ignore_ids": list(ignore_ids),
        "gt_key": gt_key,
        "mask_key": mask_key,
        "exclude_free_in_mean": bool(exclude_free_in_mean),
        "ray": bool(ray),
    }
    if miou_occ_only is not None:
        out["semantic_miou_occupied_only"] = miou_occ_only

    if ray:
        ray_out = {}
        for thr in ray_thrs:
            st = ray_stats[thr]
            iou_c, riou = compute_ray_iou(st["tp"], st["fp"], st["fn"], exclude_ids=exclude_ids)
            ray_out[str(thr)] = {"ray_iou": riou, "per_class_ray_iou": iou_c.tolist()}
        ray_out["avg"] = float(np.mean([ray_out[str(t)]["ray_iou"] for t in ray_thrs])) if len(ray_thrs) > 0 else 0.0
        out["rayIoU"] = ray_out

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py-config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--use-ema", action="store_true")

    ap.add_argument("--num-classes", type=int, default=18)
    ap.add_argument("--depth-bins", type=int, default=None)
    ap.add_argument("--free-id", type=int, default=None, help="default: K-1")
    ap.add_argument("--ignore-id", type=int, default=255, help="set -1 to disable")
    ap.add_argument("--ignore-id2", type=int, default=-1)

    ap.add_argument("--gt-key", default="occ_label")
    ap.add_argument("--mask-key", default="occ_cam_mask", help="set 'none' to disable")
    ap.add_argument("--exclude-free-in-mean", type=int, default=1)
    ap.add_argument("--also-occupied-only-miou", type=int, default=1)

    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--debug-first-batches", type=int, default=1)
    ap.add_argument("--amp", action="store_true")

    # RayIoU
    ap.add_argument("--ray", action="store_true")
    ap.add_argument("--ray-thrs", type=str, default="1,2,4")
    ap.add_argument("--ray-n-az", type=int, default=512)
    ap.add_argument("--ray-n-el", type=int, default=16)
    ap.add_argument("--ray-step", type=float, default=0.25)
    ap.add_argument("--ray-origin-z", type=float, default=0.0)
    ap.add_argument("--ray-chunk", type=int, default=4096)

    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    cfg = Config.fromfile(args.py_config)
    _, val_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=False,
        iter_resume=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_student_from_cfg(cfg).to(device)

    missing, unexpected, used = load_ckpt_into_model(model, args.ckpt, use_ema=args.use_ema)
    print(f"[CKPT] loaded: {args.ckpt}")
    print(f"[CKPT] state used: {used}")
    print(f"[CKPT] missing keys: {len(missing)}  unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print(f"[CKPT] missing keys (first 30):\n  " + "\n  ".join(missing[:30]))
    if len(unexpected) > 0:
        print(f"[CKPT] unexpected keys (first 30):\n  " + "\n  ".join(unexpected[:30]))
    if len(missing) > 50:
        print("[CKPT][WARN] missing keys 很多 -> 你可能根本沒把權重正確載進來（mIoU 會像 random）。")

    K = int(args.num_classes)
    Dz = int(args.depth_bins) if args.depth_bins is not None else int(getattr(cfg, "depth_bins", 16))
    free_id = int(args.free_id) if args.free_id is not None else (K - 1)

    ignore_ids = []
    if args.ignore_id != -1:
        ignore_ids.append(int(args.ignore_id))
    if args.ignore_id2 != -1:
        ignore_ids.append(int(args.ignore_id2))
    ignore_ids = tuple(ignore_ids)

    mask_key = args.mask_key
    if isinstance(mask_key, str) and mask_key.lower() in ("none", "null", "no"):
        mask_key = None

    ray_thrs = [float(x) for x in args.ray_thrs.split(",") if x.strip() != ""]

    metrics = eval_metrics_debug(
        model=model,
        val_loader=val_loader,
        device=device,
        K=K,
        Dz=Dz,
        free_id=free_id,
        ignore_ids=ignore_ids,
        gt_key=args.gt_key,
        mask_key=mask_key,
        exclude_free_in_mean=bool(args.exclude_free_in_mean),
        also_occupied_only_miou=bool(args.also_occupied_only_miou),
        max_batches=args.max_batches,
        debug_first_batches=args.debug_first_batches,
        amp=args.amp,
        ray=args.ray,
        ray_thrs=ray_thrs,
        ray_n_az=args.ray_n_az,
        ray_n_el=args.ray_n_el,
        ray_step=args.ray_step,
        ray_origin_z=args.ray_origin_z,
        ray_chunk=args.ray_chunk,
    )

    print("\n========================")
    print("[RESULT] Semantic voxel mIoU")
    print(f"  mIoU (exclude_free_in_mean={metrics['exclude_free_in_mean']}, free_id={metrics['free_id']})")
    print(f"    = {metrics['semantic_miou']:.6f}  ({metrics['semantic_miou']*100:.2f}%)")
    if "semantic_miou_occupied_only" in metrics:
        v = metrics["semantic_miou_occupied_only"]
        print(f"  mIoU (occupied-only sanity) = {v:.6f} ({v*100:.2f}%)")
    print("[RESULT] Occupancy (binary) IoU")
    print(f"  occ_IoU = {metrics['occ_iou']:.6f} ({metrics['occ_iou']*100:.2f}%)  "
          f"(tp={metrics['occ_tp']}, fp={metrics['occ_fp']}, fn={metrics['occ_fn']})")
    if metrics.get("ray", False):
        print("[RESULT] RayIoU")
        for k, v in metrics["rayIoU"].items():
            if k == "avg":
                continue
            print(f"  RayIoU@{k}m = {v['ray_iou']:.6f} ({v['ray_iou']*100:.2f}%)")
        print(f"  RayIoU@avg = {metrics['rayIoU']['avg']:.6f} ({metrics['rayIoU']['avg']*100:.2f}%)")
    print("========================\n")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] {args.out_json}")


if __name__ == "__main__":
    main()
