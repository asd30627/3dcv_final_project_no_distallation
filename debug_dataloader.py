import os, os.path as osp
import argparse
import numpy as np
import torch
from mmengine import Config
from mmengine.runner import set_random_seed

from model.backbone.backbone import MultiViewResNetBackbone  # ⭐ 新增這行

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

    # 1) 只用 mm 的 dataloader
    from dataset import get_dataloader
    train_loader, val_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=False,
        iter_resume=False
    )

    print("[DLDBG] train_loader len:", len(train_loader))

    # 2) 建純 PyTorch backbone（不用 mm 的 model）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = MultiViewResNetBackbone(
        out_channels=256,
        pretrained=True,
        frozen_stages=1,
        input_size=None,   # 目前 dataloader 已經給 864x1600，不縮放
    ).to(device)
    backbone.eval()
    print("[BBDBG] backbone ready on", device)

    # 3) 取一個 batch，丟進 backbone
    for i, batch in enumerate(train_loader):
        print("[DLDBG] batch idx:", i)
        print("[DLDBG] batch keys:", batch.keys())

        imgs = batch['img'].to(device)   # [B, 6, 3, 864, 1600]
        print("[DLDBG] imgs shape:", imgs.shape)

        with torch.no_grad():
            feats = backbone(imgs)       # [B, 6, C_out, H_feat, W_feat]

        print("[BBDBG] backbone feats summary:")
        print(summarize(feats))

        # 也順便看一下 occ_label，之後要算 loss 會用到
        print("[GTDBG] occ_label summary:")
        print(summarize(batch['occ_label']))

        break

if __name__ == "__main__":
    main()
