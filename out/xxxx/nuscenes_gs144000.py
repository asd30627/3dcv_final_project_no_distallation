anno_root = 'data/nuscenes_cam/'
batch_size = 1
data_aug_conf = dict(
    H=900,
    W=1600,
    bot_pct_lim=(
        0.0,
        0.0,
    ),
    final_dim=(
        864,
        1600,
    ),
    rand_flip=True,
    resize_lim=(
        1.0,
        1.0,
    ),
    rot_lim=(
        0.0,
        0.0,
    ))
data_root = 'data/nuscenes/'
drop_out = 0.1
embed_dims = 128
grad_max_norm = 35
img_norm_cfg = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
include_opa = False
input_shape = (
    1600,
    864,
)
load_from = '/home/ivlab3/GaussianFormer/ckpts/r101_dcn_fcos3d_pretrain.pth'
loss = dict(
    loss_cfgs=[
        dict(
            balance_cls_weight=True,
            empty_label=17,
            lovasz_ignore=17,
            manual_class_weight=[
                1.01552756,
                1.06897009,
                1.30013094,
                1.07253735,
                0.94637502,
                1.10087012,
                1.26960524,
                1.06258364,
                1.189019,
                1.06217292,
                1.00595144,
                0.85706115,
                1.03923299,
                0.90867526,
                0.8936431,
                0.85486129,
                0.8527829,
                0.5,
            ],
            multi_loss_weights=dict(
                loss_voxel_ce_weight=10.0, loss_voxel_lovasz_weight=1.0),
            num_classes=18,
            type='OccupancyLoss',
            use_dice_loss=False,
            use_focal_loss=False,
            use_lovasz_loss=True,
            use_sem_geo_scal_loss=False,
            weight=1.0),
    ],
    type='MultiLoss')
loss_input_convertion = dict(
    occ_mask='occ_mask',
    pred_occ='pred_occ',
    sampled_label='sampled_label',
    sampled_xyz='sampled_xyz')
max_epochs = 12
model = dict(
    encoder=dict(
        anchor_encoder=dict(
            embed_dims=128,
            include_opa=False,
            semantic_dim=18,
            semantics=True,
            type='SparseGaussian3DEncoder'),
        deformable_model=dict(
            attn_drop=0.15,
            embed_dims=128,
            kps_generator=dict(
                embed_dims=128,
                fix_scale=[
                    [
                        0,
                        0,
                        0,
                    ],
                    [
                        0.45,
                        0,
                        0,
                    ],
                    [
                        -0.45,
                        0,
                        0,
                    ],
                    [
                        0,
                        0.45,
                        0,
                    ],
                    [
                        0,
                        -0.45,
                        0,
                    ],
                    [
                        0,
                        0,
                        0.45,
                    ],
                    [
                        0,
                        0,
                        -0.45,
                    ],
                ],
                num_learnable_pts=2,
                pc_range=[
                    -50.0,
                    -50.0,
                    -5.0,
                    50.0,
                    50.0,
                    3.0,
                ],
                phi_activation='sigmoid',
                scale_range=[
                    0.08,
                    0.32,
                ],
                type='SparseGaussian3DKeyPointsGenerator',
                xyz_coordinate='cartesian'),
            num_cams=6,
            num_groups=4,
            num_levels=4,
            residual_mode='cat',
            type='DeformableFeatureAggregation',
            use_camera_embed=True,
            use_deformable_func=True),
        ffn=dict(
            act_cfg=dict(inplace=True, type='ReLU'),
            embed_dims=128,
            feedforward_channels=512,
            ffn_drop=0.1,
            in_channels=256,
            num_fcs=2,
            pre_norm=dict(type='LN'),
            type='AsymmetricFFN'),
        norm_layer=dict(normalized_shape=128, type='LN'),
        num_decoder=4,
        num_single_frame_decoder=1,
        operation_order=[
            'deformable',
            'ffn',
            'norm',
            'refine',
            'spconv',
            'norm',
            'deformable',
            'ffn',
            'norm',
            'refine',
            'spconv',
            'norm',
            'deformable',
            'ffn',
            'norm',
            'refine',
            'spconv',
            'norm',
            'deformable',
            'ffn',
            'norm',
            'refine',
        ],
        refine_layer=dict(
            embed_dims=128,
            include_opa=False,
            pc_range=[
                -50.0,
                -50.0,
                -5.0,
                50.0,
                50.0,
                3.0,
            ],
            phi_activation='sigmoid',
            refine_manual=[
                0,
                1,
                2,
            ],
            restrict_xyz=True,
            scale_range=[
                0.08,
                0.32,
            ],
            semantic_dim=18,
            semantics=True,
            semantics_activation='identity',
            type='SparseGaussian3DRefinementModule',
            unit_xyz=[
                2.0,
                2.0,
                0.5,
            ],
            xyz_coordinate='cartesian'),
        spconv_layer=dict(
            embed_channels=128,
            grid_size=[
                0.5,
                0.5,
                0.5,
            ],
            in_channels=128,
            pc_range=[
                -50.0,
                -50.0,
                -5.0,
                50.0,
                50.0,
                3.0,
            ],
            phi_activation='sigmoid',
            type='SparseConv3D',
            xyz_coordinate='cartesian'),
        type='GaussianOccEncoder'),
    head=dict(
        apply_loss_type='all',
        cuda_kwargs=dict(
            D=16,
            H=200,
            W=200,
            grid_size=0.5,
            pc_min=[
                -50.0,
                -50.0,
                -5.0,
            ],
            scale_multiplier=3),
        empty_args=None,
        num_classes=18,
        type='GaussianHead',
        with_empty=False),
    img_backbone=dict(
        dcn=dict(deform_groups=1, fallback_on_stride=False, type='DCNv2'),
        depth=101,
        frozen_stages=1,
        norm_cfg=dict(requires_grad=False, type='BN2d'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        stage_with_dcn=(
            False,
            False,
            True,
            True,
        ),
        style='caffe',
        type='ResNet',
        with_cp=True),
    img_backbone_out_indices=[
        0,
        1,
        2,
        3,
    ],
    img_neck=dict(
        add_extra_convs='on_output',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=4,
        out_channels=128,
        relu_before_extra_convs=True,
        start_level=1,
        type='FPN'),
    lifter=dict(
        anchor_grad=True,
        embed_dims=128,
        feat_grad=False,
        include_opa=False,
        num_anchor=144000,
        phi_activation='sigmoid',
        semantic_dim=18,
        semantics=True,
        type='GaussianLifter'),
    type='BEVSegmentor')
num_decoder = 4
num_groups = 4
num_levels = 4
num_single_frame_decoder = 1
occ_path = 'data/surroundocc/samples'
optimizer = dict(
    optimizer=dict(lr=0.0002, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))))
pc_range = [
    -50.0,
    -50.0,
    -5.0,
    50.0,
    50.0,
    3.0,
]
phi_activation = 'sigmoid'
print_freq = 50
scale_range = [
    0.08,
    0.32,
]
semantic_dim = 18
semantics = True
test_pipeline = [
    dict(to_float32=True, type='LoadMultiViewImageFromFiles'),
    dict(
        occ_path='data/surroundocc/samples',
        semantic=True,
        type='LoadOccupancySurroundOcc',
        use_ego=False),
    dict(type='ResizeCropFlipImage'),
    dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        to_rgb=True,
        type='NormalizeMultiviewImage'),
    dict(type='DefaultFormatBundle'),
    dict(num_cams=6, type='NuScenesAdaptor', use_ego=False),
]
train_dataset_config = dict(
    data_aug_conf=dict(
        H=900,
        W=1600,
        bot_pct_lim=(
            0.0,
            0.0,
        ),
        final_dim=(
            864,
            1600,
        ),
        rand_flip=True,
        resize_lim=(
            1.0,
            1.0,
        ),
        rot_lim=(
            0.0,
            0.0,
        )),
    data_root='data/nuscenes/',
    imageset='data/nuscenes_cam/nuscenes_infos_train_sweeps_occ.pkl',
    phase='train',
    pipeline=[
        dict(to_float32=True, type='LoadMultiViewImageFromFiles'),
        dict(
            occ_path='data/surroundocc/samples',
            semantic=True,
            type='LoadOccupancySurroundOcc',
            use_ego=False),
        dict(type='ResizeCropFlipImage'),
        dict(type='PhotoMetricDistortionMultiViewImage'),
        dict(
            mean=[
                123.675,
                116.28,
                103.53,
            ],
            std=[
                58.395,
                57.12,
                57.375,
            ],
            to_rgb=True,
            type='NormalizeMultiviewImage'),
        dict(type='DefaultFormatBundle'),
        dict(num_cams=6, type='NuScenesAdaptor', use_ego=False),
    ],
    type='NuScenesDataset')
train_loader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    pin_memory=False,
    prefetch_factor=1,
    shuffle=True)
train_pipeline = [
    dict(to_float32=True, type='LoadMultiViewImageFromFiles'),
    dict(
        occ_path='data/surroundocc/samples',
        semantic=True,
        type='LoadOccupancySurroundOcc',
        use_ego=False),
    dict(type='ResizeCropFlipImage'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        to_rgb=True,
        type='NormalizeMultiviewImage'),
    dict(type='DefaultFormatBundle'),
    dict(num_cams=6, type='NuScenesAdaptor', use_ego=False),
]
use_deformable_func = True
val_dataset_config = dict(
    data_aug_conf=dict(
        H=900,
        W=1600,
        bot_pct_lim=(
            0.0,
            0.0,
        ),
        final_dim=(
            864,
            1600,
        ),
        rand_flip=True,
        resize_lim=(
            1.0,
            1.0,
        ),
        rot_lim=(
            0.0,
            0.0,
        )),
    data_root='data/nuscenes/',
    imageset='data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl',
    phase='val',
    pipeline=[
        dict(to_float32=True, type='LoadMultiViewImageFromFiles'),
        dict(
            occ_path='data/surroundocc/samples',
            semantic=True,
            type='LoadOccupancySurroundOcc',
            use_ego=False),
        dict(type='ResizeCropFlipImage'),
        dict(
            mean=[
                123.675,
                116.28,
                103.53,
            ],
            std=[
                58.395,
                57.12,
                57.375,
            ],
            to_rgb=True,
            type='NormalizeMultiviewImage'),
        dict(type='DefaultFormatBundle'),
        dict(num_cams=6, type='NuScenesAdaptor', use_ego=False),
    ],
    type='NuScenesDataset')
val_loader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    pin_memory=False,
    prefetch_factor=1,
    shuffle=False)
work_dir = 'out/xxxx'
xyz_coordinate = 'cartesian'
