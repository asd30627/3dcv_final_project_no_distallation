from __future__ import annotations
import os
import time
import numpy as np
import cv2
import torch

from model.student import OccStudent

# ====== 你訓練用的 normalization ======
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)  # RGB
STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)

def build_projection_mat(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    生成 4x4 的 projection_mat，讓它跟你 log 的檢查一致：projection_mat ≈ K@[R|t]
    K: (3,3)
    R: (3,3)
    t: (3,)  (camera frame? 或 lidar frame? 這要跟你訓練一致)
    回傳: (4,4)
    """
    Rt = np.eye(4, dtype=np.float32)
    Rt[:3, :3] = R.astype(np.float32)
    Rt[:3,  3] = t.astype(np.float32)

    K4 = np.eye(4, dtype=np.float32)
    K4[:3, :3] = K.astype(np.float32)

    P = K4 @ Rt  # 4x4
    return P

def preprocess_imgs(imgs_bgr: list[np.ndarray], H=480, W=640) -> torch.Tensor:
    """
    imgs_bgr: list of 6 images (H,W,3) BGR uint8
    return: img tensor (1,6,3,H,W) float32 normalized (RGB)
    """
    assert len(imgs_bgr) == 6, "Need 6 camera images"
    imgs = []
    for im_bgr in imgs_bgr:
        im_bgr = cv2.resize(im_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        im_rgb = im_bgr[..., ::-1].astype(np.float32)  # BGR->RGB
        im_rgb = (im_rgb - MEAN) / STD                 # normalize
        im_chw = np.transpose(im_rgb, (2, 0, 1))       # (3,H,W)
        imgs.append(im_chw)
    imgs = np.stack(imgs, axis=0)                      # (6,3,H,W)
    imgs = torch.from_numpy(imgs).unsqueeze(0)         # (1,6,3,H,W)
    return imgs.float()

def occ2img(semantics: np.ndarray, out_size: int = 800) -> np.ndarray:
    occ_class_names = [
        'others','barrier','bicycle','bus','car','construction_vehicle','motorcycle','pedestrian',
        'traffic_cone','trailer','truck','driveable_surface','other_flat','sidewalk','terrain',
        'manmade','vegetation','free'
    ]
    color_map = np.array([
        [0,0,0,255],[255,120,50,255],[255,192,203,255],[255,255,0,255],[0,150,245,255],
        [0,255,255,255],[200,180,0,255],[255,0,0,255],[255,240,150,255],[135,60,0,255],
        [160,32,240,255],[255,0,255,255],[175,0,75,255],[75,0,75,255],[150,240,80,255],
        [230,230,250,255],[0,175,0,255],[255,255,255,255]
    ], dtype=np.uint8)

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

def logits_to_semantics(logits: torch.Tensor, num_classes: int, Dz: int) -> np.ndarray:
    x = logits.detach()
    if x.dim() == 5:
        if x.shape[1] == num_classes and x.shape[2] == Dz:          # (B,C,D,H,W)
            x = x.permute(0, 3, 4, 2, 1)                             # (B,H,W,D,C)
        elif x.shape[1] == Dz and x.shape[2] == num_classes:         # (B,D,C,H,W)
            x = x.permute(0, 3, 4, 1, 2)
        elif x.shape[1] == Dz and x.shape[-1] == num_classes:        # (B,D,H,W,C)
            x = x.permute(0, 2, 3, 1, 4)
        elif x.shape[-1] == num_classes and x.shape[-2] == Dz:       # (B,H,W,D,C)
            pass
        else:
            raise ValueError(f"Unrecognized 5D logits shape: {tuple(x.shape)}")
        sem = torch.argmax(x, dim=-1)[0].cpu().numpy().astype(np.int32)  # (H,W,D)
        return sem

    if x.dim() == 4:
        B, CD, H, W = x.shape
        if CD == num_classes * Dz:
            x = x.view(B, num_classes, Dz, H, W).permute(0, 3, 4, 2, 1)
            sem = torch.argmax(x, dim=-1)[0].cpu().numpy().astype(np.int32)
            return sem
        raise ValueError(f"Unrecognized 4D logits shape: {tuple(x.shape)}")
    raise ValueError(f"Unsupported logits dim={x.dim()}")

def main():
    weights = "./work_dirs/occ_student_nobase/epoch_10.pth"
    out_dir = "./realtime_out"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OccStudent(
        bev_h=200, bev_w=200,
        depth_bins=16, num_classes=18,
        backbone_pretrained=False,
        backbone_frozen_stages=1,
        input_size=(480, 640),
        numC_Trans=128,
        pc_range=(-50.0, -50.0, -5.0, 50.0, 50.0, 3.0),
    ).to(device).eval()

    ckpt = torch.load(weights, map_location="cpu")
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt, strict=True)

    # ====== TODO: 你要把這裡改成「即時抓六路相機」 ======
    # 這裡示範用六張圖檔
    cam_paths = [
        "cam0.png","cam1.png","cam2.png","cam3.png","cam4.png","cam5.png"
    ]

    # ====== TODO: 你要填你的相機標定 ======
    # K_list: 6 個 (3,3)
    # R_list: 6 個 (3,3)
    # t_list: 6 個 (3,)
    # 這三個必須跟你訓練/val 時 NuScenesAdaptor 定義一致（同座標系）
    K_list = [np.eye(3, dtype=np.float32) for _ in range(6)]
    R_list = [np.eye(3, dtype=np.float32) for _ in range(6)]
    t_list = [np.zeros(3, dtype=np.float32) for _ in range(6)]

    img_aug_matrix = np.eye(4, dtype=np.float32)   # 即時先 identity
    bda_mat = np.eye(3, dtype=np.float32)          # 即時先 identity

    Dz, C = 16, 18
    idx = 0

    while True:
        imgs_bgr = []
        for p in cam_paths:
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is None:
                raise FileNotFoundError(p)
            imgs_bgr.append(im)

        img = preprocess_imgs(imgs_bgr)  # (1,6,3,480,640)

        # projection_mat (1,6,4,4)
        P = []
        for i in range(6):
            P.append(build_projection_mat(K_list[i], R_list[i], t_list[i]))
        projection_mat = np.stack(P, axis=0)  # (6,4,4)

        batch = {
            "img": img.to(device),
            "projection_mat": torch.from_numpy(projection_mat).unsqueeze(0).to(device),
            "img_aug_matrix": torch.from_numpy(np.stack([img_aug_matrix]*6, axis=0)).unsqueeze(0).to(device),
            "bda_mat": torch.from_numpy(bda_mat).unsqueeze(0).to(device),
            "K": torch.from_numpy(np.stack(K_list, axis=0)).unsqueeze(0).to(device),
            "R": torch.from_numpy(np.stack(R_list, axis=0)).unsqueeze(0).to(device),
            "t": torch.from_numpy(np.stack(t_list, axis=0)).unsqueeze(0).to(device),
        }

        with torch.no_grad():
            logits = model(batch)
        sem = logits_to_semantics(logits, num_classes=C, Dz=Dz)
        sem_img = occ2img(sem, out_size=800)  # RGB

        out_path = osp.join(out_dir, f"{idx:06d}-sem.png")
        cv2.imwrite(out_path, sem_img[..., ::-1])  # RGB->BGR
        print("saved:", out_path)

        idx += 1
        time.sleep(0.05)  # 模擬 20Hz，真實就用你的相機 fps

if __name__ == "__main__":
    main()
