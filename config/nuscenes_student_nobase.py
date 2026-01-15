# config/nuscenes_student_nobase.py
# ============================================================
# ✅ NO _base_ VERSION (single-source-of-truth config)
# ============================================================

# -----------------------
# basic / runtime
# -----------------------
print_freq = 50
work_dir = "work_dirs/occ_student_nobase"
load_from = None

runner = dict(max_epochs=12)
max_epochs = 12  # 保留一份給你看，也可不用

# -----------------------
# optimizer
# -----------------------
optimizer = dict(
    lr=1e-4,
    weight_decay=0.01,
)

grad_max_norm = 35

# -----------------------
# data paths
# -----------------------
data_root = "/mnt/xs1000/data/nuscenes/"
anno_root = "/mnt/xs1000/data/nuscenes_cam/"
occ_path  = "/mnt/xs1000/data/surroundocc/samples"


# 你的模型輸入 (W,H) = (640,480)
input_shape = (640, 480)
batch_size = 1

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

# 給 Depth 生成用 (PointToMultiViewDepth)
grid_config = {
    "x": [-50, 50, 0.5],
    "y": [-50, 50, 0.5],
    "z": [-5.0, 3.0, 8.0],
    "depth": [1.0, 45.0, 1.0],
}

# -----------------------
# image augmentation (給 dataset 產生 aug_configs 用)
# 注意：你 ResizeCropFlipImage 是靠 results["aug_configs"]，
# 通常是 dataset 依 data_aug_conf 生成後塞進 results。
# 這裡先用「固定 resize=1, rot=0」讓你先把幾何對齊搞穩。
# -----------------------
data_aug_conf = {
    "resize_lim": (1.0, 1.0),       # 固定不 resize
    "final_dim": (480, 640),        # (H, W)
    "bot_pct_lim": (0.0, 0.0),      # 不裁底部
    "rot_lim": (0.0, 0.0),          # 固定不旋轉 (影像端)
    "H": 480,
    "W": 640,
    "rand_flip": True,              # 影像 flip 由 dataset 決定
}

# ============================================================
# pipelines
# ============================================================
train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),

    # 1) 讀原始 NuScenes .bin 點雲
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=3,
        pc_range=[-50, -50, -5, 50, 50, 3],
        num_pts=100000,
    ),

    # 2) ✅ BDA (BEV aug) —— 你要測試 bda_mat 是否真的進 batch
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0.0, 0.0], 
        scale_ratio_range=[0.95, 1.05],
        flip_dx_ratio=0.5,
        flip_dy_ratio=0.5
    ),

    # 3) 讀 occupancy GT（會吃到 flip_dx / flip_dy）
    dict(type="LoadOccupancySurroundOcc", occ_path=occ_path, semantic=True, use_ego=False),

    # 4) 圖像端 augmentation（這段會輸出 img_aug_matrix）
    dict(type="ResizeCropFlipImage"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),

    # 5) ✅ 由點雲投影產生 gt_depth
    dict(type="PointToMultiViewDepth", grid_config=grid_config, downsample=16),

    # 6) 打包成模型需要的 key（projection_mat / img_aug_matrix / bda_mat）
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadOccupancySurroundOcc", occ_path=occ_path, semantic=True, use_ego=False),

    # 影像端：驗證先固定，不做 photometric
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),

    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]

# ============================================================
# dataset configs
# ============================================================
train_dataset_config = dict(
    type="NuScenesDataset",
    data_root=data_root,
    imageset=anno_root + "nuscenes_infos_train_sweeps_occ.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=train_pipeline,
    phase="train",
)

val_dataset_config = dict(
    type="NuScenesDataset",
    data_root=data_root,
    imageset=anno_root + "nuscenes_infos_val_sweeps_occ.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=test_pipeline,
    phase="val",
)

# ============================================================
# dataloaders
# ============================================================
train_loader = dict(
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    drop_last=True,
)

val_loader = dict(
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

# ============================================================
# (Optional) model-side constants you might want to keep in cfg
# ============================================================
pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
depth_bins = 16
num_classes = 18
