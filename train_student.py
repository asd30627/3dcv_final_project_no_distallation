from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CV_NUM_THREADS"] = "1" # 考慮一下有沒有要加? 配合 Line 37: cv2.setNumThreads(0)


import argparse
import os.path as osp
import time
import random
import numpy as np
import logging
from datetime import timedelta
import copy
import math
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import cv2

from mmengine import Config
from model.student import OccStudent
from dataset import get_dataloader

# =========================================================
# 多線程控制：OpenCV + PyTorch
# =========================================================

# # 關掉 OpenCV 自動多線程（很重要，不然每個 worker 再開一堆 thread）
cv2.setNumThreads(0)

def compute_total_loss(student, teacher, batch, amp_enabled, kd_cfg):
    # 1) student forward (要回傳 distill 用的 feature/logits)
    out_s = student(batch, return_feats=True)
    s_logits = out_s["logits"]
    s_bev = out_s.get("bev_feat", None)

    # 2) teacher forward (no_grad)
    with torch.no_grad():
        out_t = teacher(batch, return_feats=True)
        t_logits = out_t["logits"]
        t_bev = out_t.get("bev_feat", None)

    # 3) supervised loss
    loss_dict = student.head.loss(
        occ_pred=s_logits,
        voxel_semantics=batch["occ_label"],
        mask_camera=batch["occ_cam_mask"],
    )
    occ_loss = loss_dict["loss_occ"]

    depth_loss = getattr(student, "last_depth_loss", None)
    if depth_loss is None:
        depth_loss = occ_loss.new_tensor(0.0)
    elif not torch.is_tensor(depth_loss):
        depth_loss = occ_loss.new_tensor(float(depth_loss))

    # 4) kd losses
    lambda_feat = kd_cfg.get("lambda_feat", 0.0)
    lambda_logit = kd_cfg.get("lambda_logit", 0.0)
    T = kd_cfg.get("T", 2.0)

    loss_kd_feat = occ_loss.new_tensor(0.0)
    if (s_bev is not None) and (t_bev is not None) and lambda_feat > 0:
        loss_kd_feat = torch.mean((s_bev - t_bev) ** 2)

    loss_kd_logit = occ_loss.new_tensor(0.0)
    if lambda_logit > 0:
        s = torch.log_softmax(s_logits / T, dim=1)
        t = torch.softmax(t_logits / T, dim=1)
        loss_kd_logit = torch.nn.functional.kl_div(s, t, reduction="batchmean") * (T * T)

    total = occ_loss + depth_loss + lambda_feat * loss_kd_feat + lambda_logit * loss_kd_logit

    # 你也可以回傳一個 dict 方便 log
    return total, {
        "occ": occ_loss.detach(),
        "depth": depth_loss.detach(),
        "kd_feat": loss_kd_feat.detach(),
        "kd_logit": loss_kd_logit.detach(),
    }


# ---------------------------------------------------------
# 工具函數區塊
# ---------------------------------------------------------
def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True  # 固定輸入大小通常可以加速


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.ema_model.state_dict()


def format_seconds(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def get_logger(log_file):
    logger = logging.getLogger("FlashOCCTrain")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def adjust_learning_rate(optimizer, current_iter, warmup_iters, max_iters, base_lr):
    if current_iter < warmup_iters:
        alpha = float(current_iter) / max(warmup_iters, 1)
        lr = base_lr * (0.001 + (1 - 0.001) * alpha)
    else:
        progress = (current_iter - warmup_iters) / max((max_iters - warmup_iters), 1)
        lr = base_lr * 0.5 * (1. + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def to_device(batch, device):
    """把 batch 裡所有 tensor 搬到 GPU。"""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)

    new_batch = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            new_batch[k] = v.to(device, non_blocking=True)
        else:
            new_batch[k] = v
    return new_batch


# =========================================================================
# 🚀 主程式
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--py-config', default='config/nuscenes_gs25600_solid.py')
    parser.add_argument('--work-dir', default=None, help='path to save logs and ckpts')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', default=None, help='resume from checkpoint path')
    parser.add_argument('--max-epochs', type=int, default=24, help='force max epochs if not in config')
    parser.add_argument('--amp', action='store_true', help='Enable Automatic Mixed Precision (AMP)')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size in config')

    # 🔧 新增：控制 PyTorch CPU 線程數
    # parser.add_argument(
    #     '--torch_threads',
    #     type=int,
    #     default=2,
    #     help='Max PyTorch CPU threads per process (torch.set_num_threads)'
    # )

    args = parser.parse_args()

    # # =====================================================
    # # 設定 PyTorch 線程數（這才是真的有用的地方）
    # # =====================================================
    # if args.torch_threads is not None and args.torch_threads > 0:
    #     torch.set_num_threads(args.torch_threads)
    #     torch.set_num_interop_threads(max(1, args.torch_threads // 2))
    #     print(f"[THREAD] torch_threads = {args.torch_threads}, "
    #           f"interop_threads = {max(1, args.torch_threads // 2)}")
    # else:
    #     print("[THREAD] torch_threads not set or <=0, using PyTorch default.")

    # 1. 讀取 Config
    cfg = Config.fromfile(args.py_config)

    
    # 覆寫 Batch Size
    if args.batch_size is not None and cfg.get('train_loader'):
        cfg.train_loader['batch_size'] = args.batch_size
        print(f"🚀 [Command Override] Batch size set to {args.batch_size}")

    set_random_seed(args.seed, deterministic=False)

    # Work Dir
    if args.work_dir:
        work_dir = args.work_dir
    else:
        work_dir = cfg.get('work_dir') or 'work_dirs/flash_occ_torch'

    os.makedirs(work_dir, exist_ok=True)

    logger = get_logger(osp.join(work_dir, 'train.log'))
    writer = SummaryWriter(work_dir)

    logger.info(f"🚀 Start training with config: {args.py_config}")
    logger.info(f"📂 Work dir: {work_dir}")

    # 顯示最終的 Batch Size
    current_bs = cfg.train_loader.get('batch_size', 'Unknown') if cfg.get('train_loader') else 'Unknown'
    logger.info(f"📦 Batch Size: {current_bs}")
    logger.info(f"⚡ AMP Enabled: {args.amp}")

    # 2. Dataloader
    logger.info("🛠 Building dataloaders...")
    train_loader, val_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=False,
        iter_resume=False,
    )

    # 3. Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🤖 Building OccStudent Model on {device}...")

    model = OccStudent(
        bev_h=200,
        bev_w=200,
        depth_bins=16,
        num_classes=18,
        backbone_pretrained=True,
        backbone_frozen_stages=1,
        input_size=(480, 640),
        numC_Trans=128,
        pc_range=(-50.0, -50.0, -5.0, 50.0, 50.0, 3.0),
    ).to(device)

    # 4. Optimizer
    optimizer_cfg = cfg.get('optimizer', {})
    if isinstance(optimizer_cfg, dict):
        base_lr = optimizer_cfg.get('lr', 1e-4)
        weight_decay = optimizer_cfg.get('weight_decay', 1e-2)
    else:
        base_lr = 1e-4
        weight_decay = 1e-2

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    # 5. EMA
    ema = ModelEMA(model, decay=0.999)

    # 6. Resume
    start_epoch = 1
    global_iter = 0
    if args.resume and os.path.exists(args.resume):
        logger.info(f"🔄 Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        if 'ema_state' in ckpt:
            ema.ema_model.load_state_dict(ckpt['ema_state'])
        start_epoch = ckpt['epoch'] + 1
        global_iter = ckpt['global_iter']

    # 7. Training Setup
    runner_cfg = cfg.get('runner', {})
    if isinstance(runner_cfg, dict):
        max_epochs = runner_cfg.get('max_epochs', args.max_epochs)
    else:
        max_epochs = args.max_epochs

    logger.info(f"📅 Total Epochs: {max_epochs}")

    scaler = GradScaler(enabled=args.amp)

    total_iters = max_epochs * len(train_loader)
    warmup_iters = 200

    start_time = time.time()

    # -----------------------------------------------------
    # 🧠 Training Loop
    # -----------------------------------------------------
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        epoch_start_time = time.time()

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{max_epochs}",
            leave=False
        )

        running_loss = 0.0

        for batch_idx, batch in pbar:
            global_iter += 1

            lr = adjust_learning_rate(optimizer, global_iter, warmup_iters, total_iters, base_lr)
            optimizer.zero_grad()

            # 整個 batch 先搬到 GPU
            batch = to_device(batch, device)

            with autocast(enabled=args.amp):
                occ_pred = model(batch)

                # ✅ 1) 從 model 取出 VT 算好的 depth_loss
                depth_loss = getattr(model, "last_depth_loss", None)

                occ_label = batch["occ_label"]
                occ_cam_mask = batch["occ_cam_mask"]

                loss_dict = model.head.loss(
                    occ_pred=occ_pred,
                    voxel_semantics=occ_label,
                    mask_camera=occ_cam_mask,
                )
                occ_loss = loss_dict["loss_occ"]

                # ✅ 2) depth_loss 保險處理（None / float 都變成 tensor）
                if depth_loss is None:
                    depth_loss = occ_loss.new_tensor(0.0)
                elif not torch.is_tensor(depth_loss):
                    depth_loss = occ_loss.new_tensor(float(depth_loss))

                loss = occ_loss + depth_loss



            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            ema.update(model)

            running_loss += loss.item()
            current_loss = loss.item()

            if global_iter % 50 == 0:
                writer.add_scalar('Train/Loss', current_loss, global_iter)
                writer.add_scalar('Train/LR', lr, global_iter)
                if global_iter % 200 == 0:
                    logger.info(f"Iter {global_iter} | Loss: {current_loss:.4f} | LR: {lr:.6f}")

            pbar.set_postfix({
                'loss': f"{current_loss:.4f}",
                'lr': f"{lr:.6f}"
            })

        avg_loss = running_loss / max(len(train_loader), 1)
        epoch_duration = time.time() - epoch_start_time
        logger.info(f"[TRAIN] Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}. Time: {format_seconds(epoch_duration)}")

        # -------------------------------------------------
        # 🔍 Validation
        # -------------------------------------------------
        logger.info(f"🔍 Validating Epoch {epoch}...")
        val_loss = run_validation(model, val_loader, device)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        logger.info(f"[VAL] Epoch {epoch} Val Loss: {val_loss:.4f}")

        # -------------------------------------------------
        # 💾 Save Checkpoint
        # -------------------------------------------------
        ckpt_path = osp.join(work_dir, f"epoch_{epoch}.pth")
        save_dict = {
            "epoch": epoch,
            "global_iter": global_iter,
            "model_state": model.state_dict(),
            "ema_state": ema.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
        }
        torch.save(save_dict, ckpt_path)
        torch.save(save_dict, osp.join(work_dir, "latest.pth"))
        logger.info(f"💾 Saved checkpoint to {ckpt_path}")

    total_time = time.time() - start_time
    logger.info(f"✅ Training Finished. Total time: {format_seconds(total_time)}")
    writer.close()


@torch.no_grad()
def run_validation(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    count = 0

    for batch in tqdm(val_loader, desc="Validation", leave=False):
        batch = to_device(batch, device)

        occ_pred = model(batch)
        occ_label = batch["occ_label"]
        occ_cam_mask = batch["occ_cam_mask"]

        loss_dict = model.head.loss(
            occ_pred=occ_pred,
            voxel_semantics=occ_label,
            mask_camera=occ_cam_mask,
        )

        occ_loss = loss_dict["loss_occ"]
        depth_loss = getattr(model, "last_depth_loss", None)
        if depth_loss is None:
            depth_loss = occ_loss.new_tensor(0.0)
        elif not torch.is_tensor(depth_loss):
            depth_loss = occ_loss.new_tensor(float(depth_loss))

        loss = occ_loss + depth_loss



        total_loss += loss.item()
        count += 1


    return total_loss / max(count, 1)


if __name__ == "__main__":
    main()
