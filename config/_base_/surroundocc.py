# ================== data ========================
data_root = "data/nuscenes/"
anno_root = "data/nuscenes_cam/"
occ_path = "data/surroundocc/samples"
input_shape = (640, 480)
batch_size = 1

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

# 定義 grid config (給 Depth 生成用)
grid_config = {
    'x': [-50, 50, 0.5],    # 範圍 -50~50，解析度 0.5 (總長 100m / 0.5 = 200 grid)
    'y': [-50, 50, 0.5],    # 範圍 -50~50，解析度 0.5 (總長 100m / 0.5 = 200 grid)
    'z': [-5.0, 3.0, 8.0],  # 範圍 -5~3 (Vertical)
    'depth': [1.0, 45.0, 1.0], # Depth range 1m ~ 45m
}

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    
    # ❌ 刪除這個 (這是讀 .npy 的)
    # dict(type='LoadPseudoPointFromFile', ...),

    # ✅ 改用這個 (讀原始 .bin 的)
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,  # NuScenes 原始資料是 5 維
        use_dim=3,   # 我們只需要 x,y,z 來生成 Depth (如果需要 intensity 可以改 4)
        pc_range=[-50, -50, -5, 50, 50, 3], # 配合你的 grid_config
        num_pts=100000 # 取 10 萬點
    ),

    # 2. BDA 增強 (旋轉/縮放)
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0.0, 0.0], 
        scale_ratio_range=[0.95, 1.05],
        flip_dx_ratio=0.5,
        flip_dy_ratio=0.5
    ),

    # 3. 載入 GT (會吃到 flip)
    dict(type="LoadOccupancySurroundOcc", occ_path=occ_path, semantic=True, use_ego=False),
    
    # 4. 圖像增強
    dict(type="ResizeCropFlipImage"), # 記得 config 裡要有 aug_configs 來源
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),

    # 5. ✅ [新增] 生成 GT Depth
    # 注意: downsample 要跟 ViewTransformer 一樣 (例如 16)
    dict(type="PointToMultiViewDepth", grid_config=grid_config, downsample=16),

    # 6. 打包 (bda_mat 會在這裡被轉 tensor)
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadOccupancySurroundOcc", occ_path=occ_path, semantic=True, use_ego=False),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]

data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 480,
    "W": 640,
    "rand_flip": True,
}

train_dataset_config = dict(
    type='NuScenesDataset',
    data_root=data_root,
    imageset=anno_root + "nuscenes_infos_train_sweeps_occ.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=train_pipeline,
    phase='train'
)

val_dataset_config = dict(
    type='NuScenesDataset',
    data_root=data_root,
    imageset=anno_root + "nuscenes_infos_val_sweeps_occ.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=test_pipeline,
    phase='val'
)

# train_loader = dict(
#     batch_size=batch_size,
#     num_workers=8,
#     shuffle=True
# )

# val_loader = dict(
#     batch_size=batch_size,
#     num_workers=4,
# )


# 修改 config 中的 train_loader 和 val_loader
train_loader = dict(
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,          # 8 可先保留，不夠再調 12/16（看 CPU 核心數）
    pin_memory=True,        # ✅ 讓 H2D copy 能 non_blocking / 更快
    persistent_workers=True,# ✅ 不要每個 epoch 重生 worker
    prefetch_factor=4,      # ✅ 提前準備 batch（2~8 都可試）
    drop_last=True,         # ✅ 避免最後一個小 batch 造成速度/顯存抖動
)

val_loader = dict(
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,          # 驗證不需要太多 worker
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
