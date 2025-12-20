# eval.py (BEV semantic dump + visualization + optional 3D distillation targets)
import os
import os.path as osp
import time
import argparse
import warnings
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.distributed as dist

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor

warnings.filterwarnings("ignore")


# -------------------------
# BEV color map (same spirit as FlashOcc vis_occ.py in your pasted version)
# class id: 0..17 (17=free/empty)
# -------------------------
OCC_CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

COLOR_MAP = np.array([
    [0,   0,   0,   255],   # 0 others/unknown
    [255, 120, 50,  255],   # 1 barrier
    [255, 192, 203, 255],   # 2 bicycle
    [255, 255, 0,   255],   # 3 bus
    [0,   150, 245, 255],   # 4 car
    [0,   255, 255, 255],   # 5 construction_vehicle
    [200, 180, 0,   255],   # 6 motorcycle
    [255, 0,   0,   255],   # 7 pedestrian
    [255, 240, 150, 255],   # 8 traffic_cone
    [135, 60,  0,   255],   # 9 trailer
    [160, 32,  240, 255],   # 10 truck
    [255, 0,   255, 255],   # 11 driveable_surface
    [175, 0,   75,  255],   # 12 other_flat
    [75,  0,   75,  255],   # 13 sidewalk
    [150, 240, 80,  255],   # 14 terrain
    [230, 230, 250, 255],   # 15 manmade
    [0,   175, 0,   255],   # 16 vegetation
    [255, 255, 255, 255],   # 17 free
], dtype=np.uint8)


def pass_print(*args, **kwargs):
    pass


def parse_spatial_shape(s: str) -> Tuple[int, int, int]:
    # "200,200,16" -> (200,200,16)
    parts = [int(x.strip()) for x in s.split(",")]
    assert len(parts) == 3
    return parts[0], parts[1], parts[2]


def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)


def get_sample_name(metas: Dict[str, Any], b_idx: int, i_iter: int) -> str:
    """
    盡量用 sample_idx / token 當檔名，確保跨 rank / 多次跑也穩定。
    """
    if 'sample_idx' in metas:
        try:
            sid = metas['sample_idx'][b_idx]
            if isinstance(sid, torch.Tensor):
                sid = sid.item()
            return f"{int(sid):06d}"
        except Exception:
            pass

    if 'token' in metas:
        try:
            tok = metas['token'][b_idx]
            if isinstance(tok, bytes):
                tok = tok.decode('utf-8')
            return str(tok)
        except Exception:
            pass

    return f"{i_iter:06d}_{b_idx}"


def apply_bev_transform(bev: np.ndarray, rotate_k: int = 0, flip_x: bool = False, flip_y: bool = False) -> np.ndarray:
    """
    rotate_k: 0/1/2/3 means rotate 0/90/180/270 degrees CCW using np.rot90
    flip_x: flip along H axis (vertical)
    flip_y: flip along W axis (horizontal)
    """
    out = bev
    if rotate_k % 4 != 0:
        out = np.rot90(out, k=rotate_k % 4)
    if flip_x:
        out = np.flip(out, axis=0)
    if flip_y:
        out = np.flip(out, axis=1)
    return out


def bev_sem_to_rgb(bev_sem: np.ndarray) -> np.ndarray:
    """
    bev_sem: (H,W) int class ids
    return: (H,W,3) uint8
    """
    # clamp to valid range
    bev = np.clip(bev_sem, 0, len(COLOR_MAP) - 1).astype(np.int32)
    rgb = COLOR_MAP[bev][:, :, :3].astype(np.uint8)
    return rgb


def project_3d_to_bev_sem(
    sem_3d: torch.Tensor,          # (H,W,D) long
    free_id: int = 17,
    direction: str = "low2high",   # "low2high" or "high2low"
) -> torch.Tensor:
    """
    FlashOcc/vis_occ.py-style: initialize BEV=free, then scan z and overwrite where non-free.
    """
    assert sem_3d.ndim == 3
    H, W, D = sem_3d.shape
    bev = torch.full((H, W), free_id, device=sem_3d.device, dtype=torch.long)

    if direction == "low2high":
        z_iter = range(D)
    elif direction == "high2low":
        z_iter = reversed(range(D))
    else:
        raise ValueError(f"Unknown direction: {direction}")

    for z in z_iter:
        layer = sem_3d[:, :, z]
        non_free = (layer != free_id)
        bev[non_free] = layer[non_free]
    return bev


def try_extract_3d_logits_or_prob(
    result_dict: Dict[str, Any],
    spatial_shape: Tuple[int, int, int],
    num_classes: int,
    b_idx: int,
    prefer_key: str = "",
) -> Optional[torch.Tensor]:
    """
    嘗試從 result_dict 抓出 'logits/prob' 的 3D 分佈，回傳 (C,H,W,D) float tensor。
    如果抓不到就回 None。

    你可以用 --dump-3d-key 指定 prefer_key。
    """
    H, W, D = spatial_shape

    # 常見候選 key（你之後如果知道 GF-2 的輸出 key，就用 --dump-3d-key 指定）
    candidates = []
    if prefer_key:
        candidates.append(prefer_key)
    candidates += [
        "final_logits", "final_logit", "occ_logits", "occ_logit", "logits", "logit",
        "final_probs", "final_prob", "occ_probs", "occ_prob", "probs", "prob"
    ]

    found = None
    found_key = None
    for k in candidates:
        if k in result_dict and isinstance(result_dict[k], torch.Tensor):
            found = result_dict[k]
            found_key = k
            break
        # 有些 repo 可能是 list[Tensor]（per-sample）
        if k in result_dict and isinstance(result_dict[k], (list, tuple)) and len(result_dict[k]) > 0:
            if isinstance(result_dict[k][0], torch.Tensor):
                found = result_dict[k]
                found_key = k
                break

    if found is None:
        return None

    # Case A: list/tuple per-sample
    if isinstance(found, (list, tuple)):
        t = found[b_idx]
        if t.ndim == 4 and t.shape[0] == num_classes and t.shape[1:] == (H, W, D):
            return t
        if t.ndim == 2 and t.shape[0] == H * W * D and t.shape[1] == num_classes:
            # (N,C) -> (C,H,W,D)
            return t.transpose(0, 1).contiguous().view(num_classes, H, W, D)
        if t.ndim == 2 and t.shape[1] == H * W * D and t.shape[0] == num_classes:
            # (C,N) -> (C,H,W,D)
            return t.contiguous().view(num_classes, H, W, D)
        return None

    t = found

    # Case B: batched tensors
    # Try to normalize a few typical layouts into (C,H,W,D) for this b_idx
    # 1) (B,C,H,W,D)
    if t.ndim == 5 and t.shape[0] > b_idx and t.shape[1] == num_classes:
        if tuple(t.shape[2:]) == (H, W, D):
            return t[b_idx]

    # 2) (B,H,W,D,C)
    if t.ndim == 5 and t.shape[0] > b_idx and t.shape[-1] == num_classes:
        if tuple(t.shape[1:4]) == (H, W, D):
            return t[b_idx].permute(3, 0, 1, 2).contiguous()

    # 3) (B,N,C)
    if t.ndim == 3 and t.shape[0] > b_idx and t.shape[-1] == num_classes:
        N = H * W * D
        if t.shape[1] == N:
            return t[b_idx].transpose(0, 1).contiguous().view(num_classes, H, W, D)

    # 4) (B,C,N)
    if t.ndim == 3 and t.shape[0] > b_idx and t.shape[1] == num_classes:
        N = H * W * D
        if t.shape[2] == N:
            return t[b_idx].contiguous().view(num_classes, H, W, D)

    return None


def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    # init DDP
    if args.gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20507")
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node

        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{ip}:{port}",
            world_size=hosts * gpus,
            rank=rank * gpus + local_rank)

        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
        world_size = 1

    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    # output dirs
    dump_bev_dir = args.dump_bev_dir
    dump_bev_vis_dir = args.dump_bev_vis_dir
    dump_3d_dir = args.dump_3d_dir

    if (not distributed) or (local_rank == 0):
        ensure_dir(dump_bev_dir)
        ensure_dir(dump_bev_vis_dir)
        ensure_dir(dump_3d_dir)
    # 多 rank 同時 mkdir 也沒關係
    if distributed:
        if dump_bev_dir: ensure_dir(dump_bev_dir)
        if dump_bev_vis_dir: ensure_dir(dump_bev_vis_dir)
        if dump_3d_dir: ensure_dir(dump_3d_dir)

    if dump_bev_dir:
        logger.info(f'Dump BEV semantic npy to: {dump_bev_dir}')
    if dump_bev_vis_dir:
        logger.info(f'Dump BEV visualization png to: {dump_bev_vis_dir}')
    if dump_3d_dir:
        logger.info(f'Dump 3D distill targets to: {dump_3d_dir} (type={args.dump_3d_type})')

    # build model
    import model  # noqa: F401
    from dataset import get_dataloader

    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')

    if distributed:
        if cfg.get('syncBN', True):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        my_model = torch.nn.parallel.DistributedDataParallel(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        raw_model = my_model.module
    else:
        my_model = my_model.cuda()
        raw_model = my_model

    logger.info('done ddp model')

    # dataloader
    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        val_only=True)

    # resume / load
    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from

    logger.info('resume from: ' + cfg.resume_from)
    logger.info('work dir: ' + args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        ckpt = torch.load(cfg.resume_from, map_location='cpu')
        raw_model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
        logger.info('successfully resumed.')
    elif getattr(cfg, "load_from", None):
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        try:
            logger.info(str(raw_model.load_state_dict(state_dict, strict=False)))
        except Exception:
            from misc.checkpoint_util import refine_load_from_sd
            logger.info(str(raw_model.load_state_dict(refine_load_from_sd(state_dict), strict=False)))

    # metric (keep for sanity)
    print_freq = cfg.print_freq
    from misc.metric_util import MeanIoU
    miou_metric = MeanIoU(
        list(range(1, 17)),
        17,
        ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation'],
        True, 17, filter_minmax=False)
    miou_metric.reset()

    # args
    spatial_shape = parse_spatial_shape(args.spatial_shape)  # (H,W,D)
    H, W, D = spatial_shape
    free_id = args.free_id
    num_classes = args.num_classes
    rotate_k = args.bev_rotate_k
    flip_x = args.bev_flip_x
    flip_y = args.bev_flip_y

    my_model.eval()
    os.environ['eval'] = 'true'

    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):
            if args.max_samples > 0 and i_iter_val >= args.max_samples:
                break

            # move tensors to GPU
            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda(non_blocking=True)

            input_imgs = data.pop('img')
            result_dict = my_model(imgs=input_imgs, metas=data)

            if 'final_occ' not in result_dict:
                if local_rank == 0:
                    logger.info(f'[WARN] result_dict has no final_occ at iter={i_iter_val}. keys={list(result_dict.keys())[:30]}')
                continue

            # final_occ: usually list[Tensor] per-sample, each is (N,) where N=H*W*D
            final_occ = result_dict['final_occ']

            # iterate each sample in batch
            batch_size = len(final_occ) if isinstance(final_occ, (list, tuple)) else (final_occ.shape[0] if torch.is_tensor(final_occ) else 0)
            for b_idx in range(batch_size):
                # fetch pred classes
                if isinstance(final_occ, (list, tuple)):
                    pred_occ_1d = final_occ[b_idx].long()
                else:
                    pred_occ_1d = final_occ[b_idx].long()

                # reshape to (H,W,D)
                try:
                    sem_3d = pred_occ_1d.view(H, W, D)  # (H,W,D)
                except Exception:
                    if local_rank == 0:
                        logger.info(f"[WARN] cannot view final_occ into ({H},{W},{D}) at iter={i_iter_val}, b={b_idx}, numel={pred_occ_1d.numel()}")
                    continue

                # occ mask: unobserved -> free
                if args.use_occ_mask and ('occ_mask' in result_dict):
                    occ_mask = result_dict['occ_mask'][b_idx]
                    if occ_mask.numel() == H * W * D:
                        mask_3d = occ_mask.view(H, W, D).bool()
                        sem_3d = torch.where(mask_3d, sem_3d, torch.full_like(sem_3d, free_id))
                    else:
                        # fallback: try flatten
                        try:
                            mask_3d = occ_mask.flatten().view(H, W, D).bool()
                            sem_3d = torch.where(mask_3d, sem_3d, torch.full_like(sem_3d, free_id))
                        except Exception:
                            pass

                # -------------------------
                # mIoU update (optional sanity)
                # -------------------------
                if ('sampled_label' in result_dict) and ('occ_mask' in result_dict):
                    gt_occ = result_dict['sampled_label'][b_idx]
                    occ_mask = result_dict['occ_mask'][b_idx].flatten()
                    miou_metric._after_step(pred_occ_1d, gt_occ, occ_mask)

                # -------------------------
                # 1) Project 3D -> BEV semantic label (H,W)
                # -------------------------
                bev_sem_t = project_3d_to_bev_sem(
                    sem_3d=sem_3d,
                    free_id=free_id,
                    direction=args.bev_scan_direction
                )  # (H,W) long

                # to numpy + optional transform (rotate/flip if you need align)
                bev_sem_np = bev_sem_t.detach().cpu().numpy().astype(np.uint8)
                bev_sem_np = apply_bev_transform(
                    bev_sem_np,
                    rotate_k=rotate_k,
                    flip_x=flip_x,
                    flip_y=flip_y
                )

                # file name
                name = get_sample_name(data, b_idx, i_iter_val)

                # save .npy
                if dump_bev_dir:
                    np.save(osp.join(dump_bev_dir, f"{name}_bev_sem.npy"), bev_sem_np)

                # save visualization .png
                if dump_bev_vis_dir:
                    rgb = bev_sem_to_rgb(bev_sem_np)  # (H,W,3)
                    if args.vis_scale > 1:
                        import cv2
                        rgb = cv2.resize(rgb, (rgb.shape[1] * args.vis_scale, rgb.shape[0] * args.vis_scale),
                                         interpolation=cv2.INTER_NEAREST)
                    import cv2
                    cv2.imwrite(osp.join(dump_bev_vis_dir, f"{name}_bev_sem.png"), rgb[..., ::-1])  # BGR

                # -------------------------
                # 2) Optional: dump 3D distillation targets (logits/prob)
                # -------------------------
                if dump_3d_dir:
                    t_chwd = try_extract_3d_logits_or_prob(
                        result_dict=result_dict,
                        spatial_shape=spatial_shape,
                        num_classes=num_classes,
                        b_idx=b_idx,
                        prefer_key=args.dump_3d_key
                    )

                    if t_chwd is None:
                        if args.dump_3d_type == "hard_onehot":
                            # fallback from sem_3d hard labels -> onehot prob
                            onehot = torch.nn.functional.one_hot(
                                sem_3d.clamp(min=0, max=num_classes-1).long(),
                                num_classes=num_classes
                            ).permute(3, 0, 1, 2).contiguous().float()  # (C,H,W,D)
                            arr = onehot.cpu().numpy().astype(np.float16)
                            np.save(osp.join(dump_3d_dir, f"{name}_3d_prob_hard_onehot.npy"), arr)
                        else:
                            if local_rank == 0 and (i_iter_val % print_freq == 0) and (b_idx == 0):
                                logger.info(
                                    f"[WARN] Cannot find 3D logits/prob tensor in result_dict. "
                                    f"keys={list(result_dict.keys())[:30]}. "
                                    f"Set --dump-3d-type hard_onehot to force fallback, or specify --dump-3d-key."
                                )
                    else:
                        # t_chwd is (C,H,W,D)
                        if args.dump_3d_type == "logits":
                            arr = t_chwd.detach().cpu().numpy().astype(np.float16)
                            # 可用 np.savez_compressed 節省空間
                            if args.dump_3d_compress:
                                np.savez_compressed(osp.join(dump_3d_dir, f"{name}_3d_logits_fp16.npz"), logits=arr)
                            else:
                                np.save(osp.join(dump_3d_dir, f"{name}_3d_logits_fp16.npy"), arr)

                        elif args.dump_3d_type == "prob":
                            prob = torch.softmax(t_chwd.float(), dim=0)  # (C,H,W,D)
                            arr = prob.detach().cpu().numpy().astype(np.float16)
                            if args.dump_3d_compress:
                                np.savez_compressed(osp.join(dump_3d_dir, f"{name}_3d_prob_fp16.npz"), prob=arr)
                            else:
                                np.save(osp.join(dump_3d_dir, f"{name}_3d_prob_fp16.npy"), arr)

                        elif args.dump_3d_type == "hard_onehot":
                            # if t_chwd exists but you still want hard
                            hard = torch.argmax(t_chwd, dim=0).long()  # (H,W,D)
                            onehot = torch.nn.functional.one_hot(
                                hard.clamp(min=0, max=num_classes-1),
                                num_classes=num_classes
                            ).permute(3, 0, 1, 2).contiguous().float()
                            arr = onehot.detach().cpu().numpy().astype(np.float16)
                            np.save(osp.join(dump_3d_dir, f"{name}_3d_prob_hard_onehot.npy"), arr)
                        else:
                            raise ValueError(f"Unknown --dump-3d-type: {args.dump_3d_type}")

            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info('[EVAL] Iter %5d' % (i_iter_val))

    miou, iou2 = miou_metric._after_epoch()
    logger.info(f'mIoU: {miou}, iou2: {iou2}')
    miou_metric.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval + dump BEV semantic npy/png + optional 3D distill targets')

    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)

    # Spatial & label settings
    parser.add_argument('--spatial-shape', type=str, default='200,200,16', help='H,W,D for occupancy grid')
    parser.add_argument('--num-classes', type=int, default=18, help='semantic classes including free')
    parser.add_argument('--free-id', type=int, default=17, help='free/empty label id')
    parser.add_argument('--use-occ-mask', action='store_true', default=True, help='unobserved -> free before BEV projection')

    # BEV projection (FlashOcc-style overwrite along Z)
    parser.add_argument('--bev-scan-direction', type=str, default='low2high', choices=['low2high', 'high2low'])

    # Output dirs
    parser.add_argument('--dump-bev-dir', type=str, default='', help='save BEV semantic labels as .npy')
    parser.add_argument('--dump-bev-vis-dir', type=str, default='', help='save visualization .png of BEV semantic maps')
    parser.add_argument('--vis-scale', type=int, default=4, help='png upsample factor (nearest)')

    # optional BEV alignment transforms
    parser.add_argument('--bev-rotate-k', type=int, default=0, help='np.rot90 k times (0/1/2/3)')
    parser.add_argument('--bev-flip-x', action='store_true', default=False)
    parser.add_argument('--bev-flip-y', action='store_true', default=False)

    # Optional: dump 3D distill targets
    parser.add_argument('--dump-3d-dir', type=str, default='', help='save 3D logits/prob for distillation')
    parser.add_argument('--dump-3d-type', type=str, default='prob', choices=['logits', 'prob', 'hard_onehot'])
    parser.add_argument('--dump-3d-key', type=str, default='', help='force pick result_dict[key] as 3D logits/prob')
    parser.add_argument('--dump-3d-compress', action='store_true', default=True, help='use np.savez_compressed for 3D (recommended)')

    # Control
    parser.add_argument('--max-samples', type=int, default=-1, help='debug: limit number of iters, -1 means full')

    args = parser.parse_args()

    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
