# train_student_overfit1b.py
import argparse
import torch
from mmengine import Config
from mmengine.runner import set_random_seed

from model.student import OccStudent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--py-config', default='config/nuscenes_gs25600_solid.py')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-iters', type=int, default=200)
    args = parser.parse_args()

    set_random_seed(args.seed)
    cfg = Config.fromfile(args.py_config)

    from dataset import get_dataloader
    train_loader, val_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=False,
        iter_resume=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OccStudent(
        bev_h=200,
        bev_w=200,
        depth_bins=16,
        num_classes=18,
        backbone_pretrained=True,
        backbone_frozen_stages=1,
        input_size=(900, 1600),
        numC_Trans=128,
        pc_range=(-50.0, -50.0, -5.0, 50.0, 50.0, 3.0),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)

    print("✅ OccStudent init done. Start 1-batch overfit loop ...")

    model.train()
    first_batch = None
    it = 0

    while it < args.max_iters:
        for batch in train_loader:
            # 只取第一個 batch，之後都重複用它
            if first_batch is None:
                first_batch = {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in batch.items()
                }
                print("[OVERFIT] captured first batch keys:", first_batch.keys())

            batch = first_batch
            it += 1
            if it > args.max_iters:
                break

            optimizer.zero_grad()

            # forward
            logits = model(batch)  # [B, Dx, Dy, Dz, num_classes]

            occ_label = batch["occ_label"]      # [B, 200, 200, 16]
            occ_cam_mask = batch["occ_cam_mask"]  # [B, 200, 200, 16]

            loss_dict = model.head.loss(
                occ_pred=logits,
                voxel_semantics=occ_label,
                mask_camera=occ_cam_mask,
            )

            loss_total = (
                loss_dict["loss_occ"]
                + loss_dict["loss_voxel_sem_scal"]
                + loss_dict["loss_voxel_geo_scal"]
                + loss_dict["loss_voxel_lovasz"]
            )

            loss_total.backward()
            optimizer.step()

            print(
                f"[OVERFIT-1B] iter={it} "
                f"loss_total={loss_total.item():.4f} | "
                f"occ={loss_dict['loss_occ'].item():.4f}, "
                f"sem={loss_dict['loss_voxel_sem_scal'].item():.4f}, "
                f"geo={loss_dict['loss_voxel_geo_scal'].item():.4f}, "
                f"lovasz={loss_dict['loss_voxel_lovasz'].item():.4f}"
            )

        # 保險：避免 train_loader 沒有資料還在 while 裡卡死
        if len(train_loader) == 0:
            break


if __name__ == "__main__":
    main()
