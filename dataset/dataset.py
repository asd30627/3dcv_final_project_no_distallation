import os
from copy import deepcopy
import numpy as np
from pyquaternion import Quaternion
from torch.utils.data import Dataset

import mmengine
from . import OPENOCC_DATASET, OPENOCC_TRANSFORMS
from .utils import get_img2global, get_lidar2global


@OPENOCC_DATASET.register_module()
class NuScenesDataset(Dataset):

    def __init__(
        self,
        data_root=None,
        imageset=None,
        data_aug_conf=None,
        pipeline=None,
        vis_indices=None,
        num_samples=0,
        vis_scene_index=-1,
        phase='train',
        return_keys=[
            'img',
            'projection_mat',
            'image_wh',
            'img_aug_matrix',
            'occ_label',
            'occ_xyz',
            'occ_cam_mask',
            'ori_img',
            'bda_mat',
            'gt_depth',
            "ego2lidar",
            # ✅ NEW
            "K", "R", "t",
        ]
    ):
        self.data_path = data_root
        data = mmengine.load(imageset)
        self.scene_infos = data['infos']
        self.keyframes = data['metadata']
        self.keyframes = sorted(self.keyframes, key=lambda x: x[0] + "{:0>3}".format(str(x[1])))

        self.data_aug_conf = data_aug_conf
        self.test_mode = (phase != 'train')
        self.pipeline = []
        for t in pipeline:
            self.pipeline.append(OPENOCC_TRANSFORMS.build(t))

        self.sensor_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        self.return_keys = return_keys
        if vis_scene_index >= 0:
            frame = self.keyframes[vis_scene_index]
            num_frames = len(self.scene_infos[frame[0]])
            self.keyframes = [(frame[0], i) for i in range(num_frames)]
            print(f'Scene length: {len(self.keyframes)}')
        elif vis_indices is not None:
            if len(vis_indices) > 0:
                vis_indices = [i % len(self.keyframes) for i in vis_indices]
                self.keyframes = [self.keyframes[idx] for idx in vis_indices]
            elif num_samples > 0:
                vis_indices = np.random.choice(len(self.keyframes), num_samples, False)
                self.keyframes = [self.keyframes[idx] for idx in vis_indices]
        elif num_samples > 0:
            vis_indices = np.random.choice(len(self.keyframes), num_samples, False)
            self.keyframes = [self.keyframes[idx] for idx in vis_indices]

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int(
                    (1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def __getitem__(self, index):
        scene_token, index = self.keyframes[index]
        info = deepcopy(self.scene_infos[scene_token][index])
        input_dict = self.get_data_info(info)

        if self.data_aug_conf is not None:
            input_dict["aug_configs"] = self._sample_augmentation()
        for t in self.pipeline:
            input_dict = t(input_dict)
        
        return_dict = {k: input_dict.get(k, None) for k in self.return_keys}
        return return_dict
    
    def get_data_info(self, info):
        image_paths = []
        lidar2img_rts = []
        ego2image_rts = []

        K_list, R_list, t_list = [], [], []

        def se3(rot_q, trans):
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = Quaternion(rot_q).rotation_matrix.astype(np.float32)
            T[:3, 3] = np.asarray(trans, dtype=np.float32)
            return T

        # -----------------------------
        # 1) lidar pose（純 SE3）
        # -----------------------------
        lidar_calib = info["data"]["LIDAR_TOP"]["calib"]
        lidar_pose  = info["data"]["LIDAR_TOP"]["pose"]

        lidar2ego    = se3(lidar_calib["rotation"], lidar_calib["translation"])   # lidar -> ego
        ego2global_L = se3(lidar_pose["rotation"],  lidar_pose["translation"])    # ego  -> global
        lidar2global = ego2global_L @ lidar2ego                                   # lidar -> global

        # ego2lidar (給別的模組用)
        ego2lidar = np.linalg.inv(lidar2ego).astype(np.float32)

        # 也留一份 ego2global（用 lidar 的 ego pose 近似即可）
        ego2global = ego2global_L

        # -----------------------------
        # 2) per-camera：cam pose（純 SE3）
        # -----------------------------
        for cam_type in self.sensor_types:
            cam_calib = info["data"][cam_type]["calib"]
            cam_pose  = info["data"][cam_type]["pose"]

            # cam -> ego, ego -> global, cam -> global
            cam2ego    = se3(cam_calib["rotation"], cam_calib["translation"])
            ego2global_C = se3(cam_pose["rotation"], cam_pose["translation"])
            cam2global = ego2global_C @ cam2ego

            # lidar -> cam：inv(cam->global) @ (lidar->global)
            lidar2cam = np.linalg.inv(cam2global) @ lidar2global
            lidar2cam = lidar2cam.astype(np.float32)

            # K
            K = np.asarray(cam_calib["camera_intrinsic"], dtype=np.float32)  # (3,3)
            K4 = np.eye(4, dtype=np.float32)
            K4[:3, :3] = K

            # lidar2img = K @ lidar2cam（4x4 padding）
            lidar2img = K4 @ lidar2cam

            image_paths.append(os.path.join(self.data_path, info["data"][cam_type]["filename"]))
            lidar2img_rts.append(lidar2img)

            # 這個是 ego -> cam（如果你還在用）
            ego2image_rts.append(np.linalg.inv(cam2global) @ ego2global)

            # ✅ 給 VT 用：純外參（一定要是剛體！）
            K_list.append(K)
            R_list.append(lidar2cam[:3, :3])
            t_list.append(lidar2cam[:3, 3])

        input_dict = dict(
            sample_idx=info.get("token", ""),
            occ_path=info.get("occ_path", ""),
            timestamp=info["timestamp"] / 1e6,
            img_filename=image_paths,
            pts_filename=os.path.join(self.data_path, info["data"]["LIDAR_TOP"]["filename"]),
            ego2lidar=ego2lidar,
            lidar2img=np.asarray(lidar2img_rts, dtype=np.float32),   # (N,4,4)
            ego2img=np.asarray(ego2image_rts, dtype=np.float32),     # (N,4,4)

            K=np.asarray(K_list, dtype=np.float32),   # (N,3,3)
            R=np.asarray(R_list, dtype=np.float32),   # (N,3,3)  <- 應該落在 [-1,1]
            t=np.asarray(t_list, dtype=np.float32),   # (N,3)    <- 幾公尺量級
        )
        return input_dict



    def __len__(self):
        return len(self.keyframes)