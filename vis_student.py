# vis_student.py
from __future__ import annotations

import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm

from mmengine import Config

# 你的 repo 內部
from dataset import get_dataloader
from model.student import OccStudent


# ----------------------------
# Class names & colormap (沿用你的版本)
# ----------------------------
occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free'
]

color_map = np.array([
    [0, 0, 0, 255],          # others
    [255, 120, 50, 255],     # barrier
    [255, 192, 203, 255],    # bicycle
    [255, 255, 0, 255],      # bus
    [0, 150, 245, 255],      # car
    [0, 255, 255, 255],      # construction_vehicle
    [200, 180, 0, 255],      # motorcycle
    [255, 0, 0, 255],        # pedestrian
    [255, 240, 150, 255],    # traffic_cone
    [135, 60, 0, 255],       # trailer
    [160, 32, 240, 255],     # truck
    [255, 0, 255, 255],      # driveable_surface
    [175, 0, 75, 255],       # other_flat
    [75, 0, 75, 255],        # sidewalk
    [150, 240, 80, 255],     # terrain
    [230, 230, 250, 255],    # manmade
    [0, 175, 0, 255],        # vegetation
    [255, 255, 255, 255],    # free
], dtype=np.uint8)

# pano color（你學生 repo 目前沒有 instance head，先留著接口）
def generate_rgb_color(number: int):
    red = (number % 256)
    green = ((number // 256) % 256)
    blue = ((number // 65536) % 256)
    return [red, green, blue]

pano_color_map = np.array([generate_rgb_color(number)
                           for number in np.random.randint(0, 65536 * 256, 256)], dtype=np.uint8)

inst_class_ids = [2, 3, 4, 5, 6, 7, 9, 10]


def occ2img(semantics: np.ndarray, is_pano: bool = False, panoptics: np.ndarray | None = None,
            out_size: int = 800) -> np.ndarray:
    """
    semantics: (H, W, D) int
    """
    H, W, D = semantics.shape
    free_id = len(occ_class_names) - 1

    # “柱狀”投影：沿 D 方向，取最上層非 free 覆蓋到 2D（和你原本一樣）
    semantics_2d = np.ones([H, W], dtype=np.int32) * free_id
    for i in range(D):
        si = semantics[..., i]
        non_free = (si != free_id)
        semantics_2d[non_free] = si[non_free]

    viz = color_map[semantics_2d][..., :3]

    # numpy 2.x / 1.24+：不要用 np.bool
    inst_mask = np.zeros_like(semantics_2d, dtype=bool)
    for ind in inst_class_ids:
        inst_mask[semantics_2d == ind] = True

    if is_pano and (panoptics is not None):
        panoptics_2d = np.zeros([H, W], dtype=np.int32)
        for i in range(D):
            pi = panoptics[..., i]
            si = semantics[..., i]
            non_free = (si != free_id)
            panoptics_2d[non_free] = pi[non_free]
        viz_pano = pano_color_map[panoptics_2d % len(pano_color_map)]
        viz[inst_mask, :] = viz_pano[inst_mask, :]

    viz = cv2.resize(viz, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return viz


# ----------------------------
# Batch -> GPU（沿用你 train 的方式）
# ----------------------------
def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    new_batch = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            new_batch[k] = v.to(device, non_blocking=True)
        else:
            new_batch[k] = v
    return new_batch


# ----------------------------
# logits -> (H,W,D) semantics
# ----------------------------
def logits_to_semantics(logits: torch.Tensor, num_classes: int, Dz: int) -> np.ndarray:
    """
    盡量“自動推斷” head 輸出的排列方式，轉成 (H,W,D) 的 int semantics。
    常見形狀（任一）：
      1) (B, C, Dz, H, W)
      2) (B, Dz, C, H, W)
      3) (B, C*Dz, H, W)
      4) (B, Dz, H, W, C)
      5) (B, H, W, Dz, C)
    回傳第一個 batch 的 (H, W, Dz)
    """
    assert torch.is_tensor(logits), f"logits must be tensor, got {type(logits)}"
    x = logits.detach()

    if x.dim() == 5:
        # Try to detect which dim is class
        # case (B,C,D,H,W)
        if x.shape[1] == num_classes and x.shape[2] == Dz:
            # (B,C,D,H,W) -> (B,H,W,D,C)
            x = x.permute(0, 3, 4, 2, 1)
        # case (B,D,C,H,W)
        elif x.shape[1] == Dz and x.shape[2] == num_classes:
            x = x.permute(0, 3, 4, 1, 2)  # (B,H,W,D,C)
        # case (B,D,H,W,C)
        elif x.shape[1] == Dz and x.shape[-1] == num_classes:
            x = x.permute(0, 2, 3, 1, 4)  # (B,H,W,D,C)
        # case (B,H,W,D,C)
        elif x.shape[-1] == num_classes and x.shape[-2] == Dz:
            # already (B,?, ?, D, C) but need ensure (B,H,W,D,C)
            # assume (B,H,W,D,C)
            pass
        else:
            raise ValueError(f"Unrecognized 5D logits shape: {tuple(x.shape)} "
                             f"(num_classes={num_classes}, Dz={Dz})")

        sem = torch.argmax(x, dim=-1)  # (B,H,W,D)
        sem = sem[0].cpu().numpy().astype(np.int32)
        return sem

    if x.dim() == 4:
        # (B, C*Dz, H, W)
        B, CD, H, W = x.shape
        if CD == num_classes * Dz:
            x = x.view(B, num_classes, Dz, H, W).permute(0, 3, 4, 2, 1)  # (B,H,W,D,C)
            sem = torch.argmax(x, dim=-1)[0].cpu().numpy().astype(np.int32)
            return sem
        # (B, Dz, H, W) 這種不可能是 logits（除非已 argmax），但保險
        if CD == Dz:
            # treat as already semantic id? (not likely)
            sem = x[0].cpu().numpy().astype(np.int32)
            sem = sem.transpose(1, 2, 0)  # (H,W,D)
            return sem
        raise ValueError(f"Unrecognized 4D logits shape: {tuple(x.shape)} "
                         f"(expect C*Dz={num_classes*Dz})")

    raise ValueError(f"Unsupported logits dim={x.dim()} shape={tuple(x.shape)}")

def save_surround_view(batch, out_path):
    """
    batch["img"]: (B,N,3,H,W), normalized
    """
    imgs = batch["img"][0].detach().cpu().numpy()  # (N,3,H,W)

    mean = np.array([123.675, 116.28, 103.53]).reshape(1,3,1,1)
    std = np.array([58.395, 57.12, 57.375]).reshape(1,3,1,1)

    imgs = imgs * std + mean
    imgs = imgs.astype(np.uint8)
    imgs = imgs.transpose(0,2,3,1)  # (N,H,W,3)

    # NuScenes: 3 上 + 3 下
    up = np.concatenate([imgs[0], imgs[1], imgs[2]], axis=1)
    down = np.concatenate([imgs[3], imgs[4], imgs[5]], axis=1)
    out = np.concatenate([up, down], axis=0)

    cv2.imwrite(out_path, out[..., ::-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--py-config', required=True)
    parser.add_argument('--weights', required=True, help='e.g., work_dirs/.../latest.pth')
    parser.add_argument('--viz-dir', required=True)
    parser.add_argument('--max-samples', type=int, default=50)
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.viz_dir, exist_ok=True)

    cfg = Config.fromfile(args.py_config)

    # build dataloader (和 train_student.py 一樣)
    train_loader, val_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=False,
        iter_resume=False,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build model (和 train_student.py 一樣)
    model = OccStudent(
        bev_h=200,
        bev_w=200,
        depth_bins=cfg.get('depth_bins', 16),
        num_classes=cfg.get('num_classes', 18),
        backbone_pretrained=False,  # 推論不需要下載 pretrained，避免你環境卡住；要也可改 True
        backbone_frozen_stages=1,
        input_size=(480, 640),
        numC_Trans=128,
        pc_range=(-50.0, -50.0, -5.0, 50.0, 50.0, 3.0),
    ).to(device)
    model.eval()

    # load checkpoint
    ckpt = torch.load(args.weights, map_location='cpu')
    if 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'], strict=True)
    else:
        # 允許你直接存 state_dict 的情況
        model.load_state_dict(ckpt, strict=True)

    Dz = cfg.get('depth_bins', 16)
    C = cfg.get('num_classes', 18)

    # inference
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader), total=min(len(val_loader), args.max_samples)):
            if i >= args.max_samples:
                break

            batch = to_device(batch, device)

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(batch)

            # logits -> semantics (H,W,D)
            sem = logits_to_semantics(logits, num_classes=C, Dz=Dz)

            # save
            sem_img = occ2img(semantics=sem, is_pano=False)
            out_path = osp.join(args.viz_dir, f"{i:04d}-sem.jpg")
            cv2.imwrite(out_path, sem_img[..., ::-1])  # RGB->BGR

    print(f"✅ Done. Saved to: {args.viz_dir}")


if __name__ == '__main__':
    main()
