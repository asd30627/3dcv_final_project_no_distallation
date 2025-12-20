# debug_dataloader_student.py
import os
import os.path as osp
import argparse
import numpy as np
import torch
from mmengine import Config
from mmengine.runner import set_random_seed

# ⭐ 用 Student，而不是只用 backbone
from model.student import OccStudent


def summarize(x, max_list=8, max_sample=10):
    def rec(v):
        if torch.is_tensor(v):
            flat = v.flatten()
            sample_len = min(flat.numel(), max_sample)
            sample = flat[:sample_len].detach().cpu().tolist()
            return dict(
                type="torch.Tensor",
                shape=list(v.shape),
                dtype=str(v.dtype),
                device=str(v.device),
                sample_len=sample_len,
                sample=sample,
            )
        if isinstance(v, np.ndarray):
            flat = v.reshape(-1)
            sample_len = min(flat.size, max_sample)
            sample = flat[:sample_len].tolist()
            return dict(
                type="np.ndarray",
                shape=list(v.shape),
                sample_len=sample_len,
                sample=sample,
            )
        if isinstance(v, dict):
            out = {"type": "dict", "keys": list(v.keys())}
            for k in list(v.keys())[:max_list]:
                out[k] = rec(v[k])
            if len(v) > max_list:
                out["_truncated_keys"] = len(v) - max_list
            return out
        if isinstance(v, (list, tuple)):
            out = {"type": type(v).__name__, "len": len(v)}
            for i in range(min(len(v), max_list)):
                out[str(i)] = rec(v[i])
            if len(v) > max_list:
                out["_truncated_items"] = len(v) - max_list
            return out
        return {"type": type(v).__name__, "repr": repr(v)[:200]}
    return rec(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--py-config', default='config/nuscenes_gs25600_solid.py')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_random_seed(args.seed)
    cfg = Config.fromfile(args.py_config)

    # 1) dataloader —— 這裡「一定要」跟 train_student.py 寫法一致
    from dataset import get_dataloader
    train_loader, val_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=False,
        iter_resume=False,
        # 如果你在 train_student.py 多給了 train_wrapper_config / val_wrapper_config
        # 就照那邊一模一樣搬過來
    )

    print("[DLDBG] train_loader len:", len(train_loader))

    # 2) 建 OccStudent（參數請跟 train_student.py 一樣，這裡給你一個典型寫法）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 如果你在 model.student 裡有定 bev_h, bev_w, depth_bins, num_classes 常數，就可以 from 那邊 import
    # 這邊假設就是 200,200,16,18；如果不一樣請改掉
    student = OccStudent(
        bev_h=200,
        bev_w=200,
        depth_bins=16,
        num_classes=18,
        backbone_pretrained=True,
        backbone_frozen_stages=1,
        input_size=(864, 1600),  # 或跟你 dataloader 給的一樣；跟 train_student.py 對齊
        numC_Trans=128,
        pc_range=(-50.0, -50.0, -5.0, 50.0, 50.0, 3.0),
        vt_depth_min=1.0,
        vt_depth_max=45.0,
        vt_normalize=True,
        vt_depthnet_mid=256,
        vt_depthnet_with_cp=False,
        vt_depthnet_use_aspp=True,
        vt_hw_chunk=None,
        bev_encoder_channels=(128, 256, 512),
        bev_out_channels=256,
    ).to(device)
    student.eval()
    print("[STDBG] OccStudent ready on", device)

    # 3) 拿一個 batch，整條 pipeline 跑過去
    for i, batch in enumerate(train_loader):
        print("==================================================")
        print("[DLDBG] batch idx:", i)
        print("[DLDBG] batch keys:", list(batch.keys()))

        imgs = batch['img'].to(device)   # [B, N_cam, 3, H, W]
        print("[DLDBG] imgs shape:", imgs.shape)

        # 也看一下 GT、mask 形狀
        if 'occ_label' in batch:
            print("[GTDBG] occ_label shape:", batch['occ_label'].shape)
        if 'mask_camera' in batch:
            print("[GTDBG] mask_camera shape:", batch['mask_camera'].shape)

        # ---- forward through full student model ----
        with torch.no_grad():
            logits = student(batch)   # 期望: [B, Dx, Dy, Dz, num_classes]

        print("[STDBG] logits summary:")
        print(summarize(logits))

        # 4) 順便試算一次 head loss，看介面有沒有 mismatch
        if ('occ_label' in batch):
            voxel_sem = batch['occ_label'].to(device)          # (B, Dx, Dy, Dz)
            mask_cam = batch.get('mask_camera', None)
            if mask_cam is not None:
                mask_cam = mask_cam.to(device)

            # student.head 是 BEVOccHead2D_V2
            loss_dict = student.head.loss(logits, voxel_sem, mask_cam)
            print("[LOSSDBG] loss keys:", list(loss_dict.keys()))
            for k, v in loss_dict.items():
                print(f"  {k}: {float(v):.4f}")

        # 我們只看第一個 batch 就 break
        break


if __name__ == "__main__":
    main()
