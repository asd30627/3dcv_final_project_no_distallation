from __future__ import annotations

import os
import os.path as osp
import argparse
import glob
import numpy as np
import cv2
import torch
from tqdm import tqdm

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

inst_class_ids = [2, 3, 4, 5, 6, 7, 9, 10]


def occ2img(semantics: np.ndarray, out_size: int = 800) -> np.ndarray:
    """
    semantics: (H, W, D) int
    “柱狀”投影：沿 D 方向，取最上層非 free 覆蓋到 2D（和你原本一樣）
    """
    H, W, D = semantics.shape
    free_id = len(occ_class_names) - 1

    semantics_2d = np.ones([H, W], dtype=np.int32) * free_id
    for i in range(D):
        si = semantics[..., i]
        non_free = (si != free_id)
        semantics_2d[non_free] = si[non_free]

    viz = color_map[semantics_2d][..., :3]  # RGB
    viz = cv2.resize(viz, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return viz


def to_device(batch: dict, device: torch.device) -> dict:
    new_batch = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            new_batch[k] = v.to(device, non_blocking=True)
        else:
            new_batch[k] = v
    return new_batch


def logits_to_semantics(logits: torch.Tensor, num_classes: int, Dz: int) -> np.ndarray:
    """
    盡量自動推斷 head 輸出的排列方式，轉成 (H,W,D) 的 int semantics。
    支援常見形狀：
      1) (B, C, Dz, H, W)
      2) (B, Dz, C, H, W)
      3) (B, C*Dz, H, W)
      4) (B, Dz, H, W, C)
      5) (B, H, W, Dz, C)
    """
    x = logits.detach()

    if x.dim() == 5:
        # (B,C,D,H,W)
        if x.shape[1] == num_classes and x.shape[2] == Dz:
            x = x.permute(0, 3, 4, 2, 1)  # (B,H,W,D,C)
        # (B,D,C,H,W)
        elif x.shape[1] == Dz and x.shape[2] == num_classes:
            x = x.permute(0, 3, 4, 1, 2)  # (B,H,W,D,C)
        # (B,D,H,W,C)
        elif x.shape[1] == Dz and x.shape[-1] == num_classes:
            x = x.permute(0, 2, 3, 1, 4)  # (B,H,W,D,C)
        # (B,H,W,D,C)
        elif x.shape[-1] == num_classes and x.shape[-2] == Dz:
            pass
        else:
            raise ValueError(f"Unrecognized 5D logits shape: {tuple(x.shape)} "
                             f"(num_classes={num_classes}, Dz={Dz})")

        sem = torch.argmax(x, dim=-1)  # (B,H,W,D)
        return sem[0].cpu().numpy().astype(np.int32)

    if x.dim() == 4:
        B, CD, H, W = x.shape
        if CD == num_classes * Dz:
            x = x.view(B, num_classes, Dz, H, W).permute(0, 3, 4, 2, 1)  # (B,H,W,D,C)
            sem = torch.argmax(x, dim=-1)
            return sem[0].cpu().numpy().astype(np.int32)

        # 保險：如果你不小心餵到已 argmax 的 (B,D,H,W) 類型
        if CD == Dz:
            sem = x[0].cpu().numpy().astype(np.int32)  # (D,H,W)
            return sem.transpose(1, 2, 0)  # (H,W,D)

        raise ValueError(f"Unrecognized 4D logits shape: {tuple(x.shape)} "
                         f"(expect C*Dz={num_classes*Dz})")

    raise ValueError(f"Unsupported logits dim={x.dim()} shape={tuple(x.shape)}")


def save_surround_view(batch: dict, out_path: str):
    """
    batch["img"]: (B,N,3,H,W), normalized
    你原本的反正規化+拼圖，輸出 PNG
    """
    if "img" not in batch:
        return

    imgs = batch["img"][0].detach().cpu().numpy()  # (N,3,H,W)

    mean = np.array([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1)
    std = np.array([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)

    imgs = imgs * std + mean
    imgs = imgs.astype(np.uint8)
    imgs = imgs.transpose(0, 2, 3, 1)  # (N,H,W,3) RGB

    # NuScenes: 3 上 + 3 下
    up = np.concatenate([imgs[0], imgs[1], imgs[2]], axis=1)
    down = np.concatenate([imgs[3], imgs[4], imgs[5]], axis=1)
    out = np.concatenate([up, down], axis=0)

    cv2.imwrite(out_path, out[..., ::-1])  # RGB->BGR


def load_pt_list(pt_dir: str):
    pts = sorted(glob.glob(osp.join(pt_dir, "golden_*.pt")))
    if len(pts) == 0:
        one = osp.join(pt_dir, "golden_batch.pt")
        if osp.exists(one):
            pts = [one]
        else:
            raise FileNotFoundError(f"No golden_*.pt or golden_batch.pt found in: {pt_dir}")
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-dir", required=True, help="包含 golden_*.pt 或 golden_batch.pt 的資料夾")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--amp", action="store_true")

    # model cfg（跟你訓練一致的那些）
    ap.add_argument("--depth-bins", type=int, default=16)
    ap.add_argument("--num-classes", type=int, default=18)
    ap.add_argument("--bev-h", type=int, default=200)
    ap.add_argument("--bev-w", type=int, default=200)
    ap.add_argument("--out-size", type=int, default=800)

    # output 控制
    ap.add_argument("--save-surround", action="store_true", help="輸出 6 視角拼圖 png")
    ap.add_argument("--save-sem-npy", action="store_true", help="另外存 semantics.npy")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pt_list = load_pt_list(args.pt_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OccStudent(
        bev_h=args.bev_h,
        bev_w=args.bev_w,
        depth_bins=args.depth_bins,
        num_classes=args.num_classes,
        backbone_pretrained=False,
        backbone_frozen_stages=1,
        input_size=(480, 640),
        numC_Trans=128,
        pc_range=(-50.0, -50.0, -5.0, 50.0, 50.0, 3.0),
    ).to(device).eval()

    ckpt = torch.load(args.weights, map_location="cpu")
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)

    Dz = args.depth_bins
    C = args.num_classes

    with torch.no_grad():
        for i, pt_path in tqdm(enumerate(pt_list[:args.max_samples]),
                               total=min(len(pt_list), args.max_samples)):
            batch = torch.load(pt_path, map_location="cpu")
            if not isinstance(batch, dict):
                raise TypeError(f"{pt_path} is not a dict batch")

            # （可選）輸出 surround view
            if args.save_surround:
                save_surround_view(batch, osp.join(args.out_dir, f"{i:04d}-surround.png"))

            batch_gpu = to_device(batch, device)

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(batch_gpu)

            sem = logits_to_semantics(logits, num_classes=C, Dz=Dz)

            # （可選）存 npy
            if args.save_sem_npy:
                np.save(osp.join(args.out_dir, f"{i:04d}-semantics.npy"), sem)

            # 存 sem png（和你原本 vis 一樣）
            sem_img = occ2img(sem, out_size=args.out_size)  # RGB
            cv2.imwrite(osp.join(args.out_dir, f"{i:04d}-sem.png"), sem_img[..., ::-1])  # RGB->BGR

    print(f"✅ Done. Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
