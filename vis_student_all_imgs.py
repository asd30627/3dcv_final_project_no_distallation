# vis_student_all_imgs.py
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
# Class names & colormap
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


def generate_rgb_color(number: int):
    red = (number % 256)
    green = ((number // 256) % 256)
    blue = ((number // 65536) % 256)
    return [red, green, blue]


pano_color_map = np.array(
    [generate_rgb_color(number) for number in np.random.randint(0, 65536 * 256, 256)],
    dtype=np.uint8
)

inst_class_ids = [2, 3, 4, 5, 6, 7, 9, 10]


def occ2img(
    semantics: np.ndarray,
    is_pano: bool = False,
    panoptics: np.ndarray | None = None,
    out_size: int = 800
) -> np.ndarray:
    """
    semantics: (H, W, D) int
    """
    assert semantics.ndim == 3, f"semantics must be (H,W,D), got {semantics.shape}"
    H, W, D = semantics.shape
    free_id = len(occ_class_names) - 1

    # 柱狀投影：沿 D 方向，最後覆蓋的非 free 會留在 2D 上
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


def logits_to_semantics(logits: torch.Tensor, num_classes: int, Dz: int) -> np.ndarray:
    """
    自動推斷 logits layout，回傳 (H,W,D) int32
    """
    assert torch.is_tensor(logits), f"logits must be tensor, got {type(logits)}"
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
            raise ValueError(
                f"Unrecognized 5D logits shape: {tuple(x.shape)} (C={num_classes}, Dz={Dz})"
            )

        sem = torch.argmax(x, dim=-1)  # (B,H,W,D)
        return sem[0].cpu().numpy().astype(np.int32)

    if x.dim() == 4:
        # (B, C*Dz, H, W)
        B, CD, H, W = x.shape
        if CD == num_classes * Dz:
            x = x.view(B, num_classes, Dz, H, W).permute(0, 3, 4, 2, 1)  # (B,H,W,D,C)
            sem = torch.argmax(x, dim=-1)[0].cpu().numpy().astype(np.int32)
            return sem
        raise ValueError(
            f"Unrecognized 4D logits shape: {tuple(x.shape)} (expect C*Dz={num_classes*Dz})"
        )

    raise ValueError(f"Unsupported logits dim={x.dim()} shape={tuple(x.shape)}")


def save_surround_view(batch, out_path):
    """
    batch["img"]: (B,N,3,H,W), normalized
    """
    if "img" not in batch:
        return False

    imgs = batch["img"][0].detach().cpu().numpy()  # (N,3,H,W)
    if imgs.shape[0] < 6:
        return False

    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 3, 1, 1)

    imgs = imgs * std + mean
    imgs = np.clip(imgs, 0, 255).astype(np.uint8)
    imgs = imgs.transpose(0, 2, 3, 1)  # (N,H,W,3)

    up = np.concatenate([imgs[0], imgs[1], imgs[2]], axis=1)
    down = np.concatenate([imgs[3], imgs[4], imgs[5]], axis=1)
    out = np.concatenate([up, down], axis=0)

    cv2.imwrite(out_path, out[..., ::-1])
    return True


def save_bev_mask(mask_hw: np.ndarray, out_path: str, out_size: int = 800):
    """
    mask_hw: (H,W) bool/0-1
    """
    m = mask_hw.astype(np.uint8) * 255
    m = cv2.resize(m, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    m3 = np.stack([m, m, m], axis=-1)
    cv2.imwrite(out_path, m3)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--py-config', required=True)
    parser.add_argument('--weights', required=True, help='e.g., work_dirs/.../latest.pth or epoch_*.pth')
    parser.add_argument('--viz-dir', required=True)
    parser.add_argument('--max-samples', type=int, default=100)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--start-idx', type=int, default=0, help='skip first k samples')
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
        backbone_pretrained=False,
        backbone_frozen_stages=1,
        input_size=(480, 640),
        numC_Trans=128,
        pc_range=(-50.0, -50.0, -5.0, 50.0, 50.0, 3.0),
    ).to(device)
    model.eval()

    # load checkpoint
    ckpt = torch.load(args.weights, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'], strict=True)
        used_epoch = ckpt.get("epoch", None)
    else:
        model.load_state_dict(ckpt, strict=True)
        used_epoch = None

    Dz = cfg.get('depth_bins', 16)
    C = cfg.get('num_classes', 18)

    if used_epoch is not None:
        print(f"[VIS] Using checkpoint epoch = {used_epoch}")
    print(f"[VIS] Dz={Dz}, num_classes={C}")
    print(f"[VIS] Saving to: {args.viz_dir}")

    # 你要觀察「變小塊」的關鍵 idx
    dbg_idxs = set([75, 79, 80, 81, 82])


    # inference
    saved = 0
    with torch.no_grad():
        pbar_total = args.start_idx + args.max_samples
        for i, batch in tqdm(enumerate(val_loader), total=pbar_total):
            if i < args.start_idx:
                continue
            if saved >= args.max_samples:
                break

            batch = to_device(batch, device)

            # ---------- forward ----------
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(batch)

            # ---------- pred -> semantics ----------
            sem_pred = logits_to_semantics(logits, num_classes=C, Dz=Dz)

            # ---------- save pred ----------
            idx = i
            pred_path = osp.join(args.viz_dir, f"{idx:04d}-sem_pred.jpg")
            cv2.imwrite(pred_path, occ2img(sem_pred)[..., ::-1])

            # ---------- save pred any-occupied (避免 occ2img 沿 D 覆蓋造成視覺誤判) ----------
            free_id = C - 1
            pred_anyocc = (sem_pred != free_id).any(axis=-1).astype(np.uint8) * 255  # (H,W)
            pred_anyocc = cv2.resize(pred_anyocc, (800, 800), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(osp.join(args.viz_dir, f"{idx:04d}-pred_anyocc.jpg"), pred_anyocc)

            # ---------- DBG2：看是不是突然變成幾乎全 free ----------
            if idx in dbg_idxs:
                free_ratio = float((sem_pred == free_id).mean())
                uniq, cnt = np.unique(sem_pred.reshape(-1), return_counts=True)
                top = sorted(zip(cnt.tolist(), uniq.tolist()), reverse=True)[:8]
                print(f"[DBG2] idx={idx:04d} free_ratio={free_ratio*100:.2f}% top8(count,class)={top}")


            # ---------- save GT semantics ----------
            if "occ_label" in batch and batch["occ_label"] is not None:
                gt = batch["occ_label"][0].detach().cpu().numpy().astype(np.int32)  # (H,W,D)
                gt_path = osp.join(args.viz_dir, f"{idx:04d}-sem_gt.jpg")
                cv2.imwrite(gt_path, occ2img(gt)[..., ::-1])

                gt_anyocc = (gt != free_id).any(axis=-1).astype(np.uint8) * 255
                gt_anyocc = cv2.resize(gt_anyocc, (800, 800), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(osp.join(args.viz_dir, f"{idx:04d}-gt_anyocc.jpg"), gt_anyocc)


            # ---------- save surround-view RGB ----------
            rgb_path = osp.join(args.viz_dir, f"{idx:04d}-rgb.jpg")
            save_surround_view(batch, rgb_path)

            # ---------- save BEV cam mask (非常關鍵：抓 0080 之後縮小) ----------
            if "occ_cam_mask" in batch and batch["occ_cam_mask"] is not None:
                m = batch["occ_cam_mask"][0].detach().cpu().numpy()  # (H,W,D) or (H,W) or (H,W,1)
                if m.ndim == 3:
                    # (H,W,D) -> any-visible mask
                    m2 = (m.sum(axis=-1) > 0)
                elif m.ndim == 2:
                    m2 = (m > 0)
                else:
                    m2 = (m.squeeze() > 0)

                mask_path = osp.join(args.viz_dir, f"{idx:04d}-cam_mask.jpg")
                save_bev_mask(m2, mask_path)

                # debug: coverage ratio
                cov = float(m2.mean())
                # 若 cov 很低（例如 < 1%），你就會看到 pred 只剩一小塊
                # if (saved == 0) or (saved % 30 == 0) or (idx in [75, 79, 80, 81, 82]):
                #     print(f"[DBG] idx={idx:04d} cam_mask_coverage={cov*100:.2f}% logits_shape={tuple(logits.shape)}")
                if idx in dbg_idxs:
                    print(f"[DBG] idx={idx:04d} cam_mask_coverage={cov*100:.2f}% logits_shape={tuple(logits.shape)}")



            # ---------- (optional) gt_depth 存成灰階 ----------
            if "gt_depth" in batch:
                gd0 = batch["gt_depth"]

                # 情況 A：整個就是 None
                if gd0 is None:
                    pass
                else:
                    # 情況 B：可能是 list/tuple（常見），或 np.object_ array
                    # 先拿到第一個 batch 的內容
                    if isinstance(gd0, (list, tuple)):
                        gd0 = gd0[0] if len(gd0) > 0 else None
                    elif isinstance(gd0, np.ndarray) and gd0.dtype == object:
                        gd0 = gd0[0] if gd0.size > 0 else None
                    else:
                        # tensor: (B,N,Hf,Wf) -> 取 batch0
                        if torch.is_tensor(gd0) and gd0.dim() >= 4:
                            gd0 = gd0[0]

                    # 情況 C：取完 batch0 仍然是 None（你現在炸的就是這種）
                    if gd0 is None:
                        pass
                    else:
                        # gd0 現在預期是 tensor 或 numpy，shape (N,Hf,Wf)
                        if torch.is_tensor(gd0):
                            gd = gd0.detach().cpu().numpy()
                        else:
                            gd = np.asarray(gd0)

                        if gd.ndim == 3 and gd.shape[0] >= 1:
                            d0 = gd[0]  # cam0
                            d0 = np.nan_to_num(d0, nan=0.0, posinf=0.0, neginf=0.0)

                            # normalize for visualization
                            pos = d0[d0 > 0]
                            vmax = np.percentile(pos, 95) if pos.size > 0 else 1.0
                            d0n = np.clip(d0 / max(vmax, 1e-6), 0, 1)
                            d0u = (d0n * 255).astype(np.uint8)
                            d0u = cv2.resize(d0u, (800, 800), interpolation=cv2.INTER_NEAREST)
                            cv2.imwrite(osp.join(args.viz_dir, f"{idx:04d}-gt_depth_cam0.jpg"), d0u)


            saved += 1

    print(f"✅ Done. Saved {saved} samples to: {args.viz_dir}")


if __name__ == '__main__':
    main()

