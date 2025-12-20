import os
import torch
import numpy as np
from numpy import random
import mmcv
from PIL import Image
import math
from copy import deepcopy

from . import OPENOCC_TRANSFORMS


@OPENOCC_TRANSFORMS.register_module()
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __init__(self, ):
        return

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in results['img']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
            else:
                imgs = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
            results['img'] = torch.from_numpy(imgs)
        return results

    def __repr__(self):
        return self.__class__.__name__


@OPENOCC_TRANSFORMS.register_module()
class NuScenesAdaptor(object):
    def __init__(self, num_cams, use_ego=False):
        self.num_cams = num_cams
        self.projection_key = 'ego2img' if use_ego else 'lidar2img'

    def __call__(self, input_dict):
        # 1. Projection Matrix
        input_dict["projection_mat"] = torch.from_numpy(
            np.float32(np.stack(input_dict[self.projection_key]))
        )  # (N,4,4)

        # 2. Image WH
        input_dict["image_wh"] = torch.from_numpy(
            np.ascontiguousarray(
                np.array(input_dict["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
            )
        )

        # 3. Image Augmentation Matrix (Route A)
        if "img_aug_matrix" in input_dict:
            input_dict["img_aug_matrix"] = torch.from_numpy(
                np.ascontiguousarray(input_dict["img_aug_matrix"], dtype=np.float32)
            )
        else:
            # Fallback
            input_dict["img_aug_matrix"] = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(self.num_cams, 1, 1)

        # 4. ✅ [關鍵修正] 處理 BDA Matrix
        if "bda_mat" in input_dict:
            input_dict["bda_mat"] = torch.from_numpy(
                np.array(input_dict["bda_mat"], dtype=np.float32)
            )
        else:
            # 若無 BDA，給單位矩陣 (3, 3)
            input_dict["bda_mat"] = torch.eye(3, dtype=torch.float32)

        # 5. ✅ [關鍵修正] 確保 gt_depth 存在
        # PointToMultiViewDepth 已經生成了 Tensor，這裡確認一下即可
        if "gt_depth" not in input_dict:
             # 如果沒有跑 depth generation，給個空的避免報錯 (雖然這樣訓練會爆)
             # 建議: 報錯提醒
             pass 
        
        # 6) ✅ 把 ego2lidar / lidar2ego 一起帶出去（修正 frame 用）
        if "ego2lidar" in input_dict:
            ego2lidar = np.asarray(input_dict["ego2lidar"], dtype=np.float32)  # (4,4)
            input_dict["ego2lidar"] = torch.from_numpy(ego2lidar)
            input_dict["lidar2ego"] = torch.from_numpy(np.linalg.inv(ego2lidar).astype(np.float32))

        # ✅ NEW: pass K/R/t
        if "K" in input_dict:
            input_dict["K"] = torch.from_numpy(np.asarray(input_dict["K"], dtype=np.float32))  # (N,3,3)
        if "R" in input_dict:
            input_dict["R"] = torch.from_numpy(np.asarray(input_dict["R"], dtype=np.float32))  # (N,3,3)
        if "t" in input_dict:
            input_dict["t"] = torch.from_numpy(np.asarray(input_dict["t"], dtype=np.float32))  # (N,3)


        return input_dict

@OPENOCC_TRANSFORMS.register_module()
class LoadPointsFromFile(object):
    """
    讀取原始 NuScenes .bin 點雲檔案
    """
    def __init__(self, coord_type, load_dim, use_dim, pc_range, num_pts):
        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.pc_range = pc_range
        self.num_pts = num_pts

    def __call__(self, results):
        # 1. 取得檔案路徑
        # NuScenesDataset 通常會提供 'pts_filename'，是相對路徑 (samples/LIDAR_TOP/xxx.bin)
        pts_filename = results['pts_filename']
        
        # 確保路徑完整 (如果 pts_filename 只是相對路徑)
        # results['img_filename'][0] 通常包含 data_root 的前綴，我們可以借鑑
        # 但通常 mmengine 的 dataset 會處理好，我們先假設路徑是對的
        # 如果報錯找不到檔案，通常是因為 data_root 沒接上去
        if not os.path.exists(pts_filename):
            # 嘗試接上 data_root (通常在 dataset 初始化時定義，或者在 img_filename 裡找線索)
            # 這裡做一個簡單的 fallback：假設 data_root 在 results 外部傳入或寫死
            # 比較穩的做法是依賴 Dataset 類別把完整路徑傳進來
            # 這裡假設 results['pts_filename'] 已經是完整路徑
            pass

        # 2. 讀取 .bin 檔案 (float32)
        try:
            points = np.fromfile(pts_filename, dtype=np.float32)
        except FileNotFoundError:
            # 容錯處理：有些 dataset 設定 pts_filename 是相對路徑
            # 嘗試手動拼接 data/nuscenes/
            data_root = "data/nuscenes/" 
            points = np.fromfile(os.path.join(data_root, pts_filename), dtype=np.float32)

        # 3. Reshape (N, 5) -> x, y, z, intensity, ring_index
        points = points.reshape(-1, self.load_dim)
        
        # 4. 取前幾維 (通常取前3或前4)
        points = points[:, :self.use_dim] # (N, 3) or (N, 4)

        # 5. Filter (過濾範圍外的點)
        mask = (points[:, 0] > self.pc_range[0]) & (points[:, 0] < self.pc_range[3]) & \
               (points[:, 1] > self.pc_range[1]) & (points[:, 1] < self.pc_range[4]) & \
               (points[:, 2] > self.pc_range[2]) & (points[:, 2] < self.pc_range[5])
        points = points[mask]

        # 6. Sampling (隨機採樣到固定點數，避免顯存爆炸)
        if points.shape[0] < self.num_pts:
            # 點不夠，重複補點
            choice = np.random.choice(points.shape[0], self.num_pts, replace=True)
        else:
            # 點太多，隨機選
            choice = np.random.choice(points.shape[0], self.num_pts, replace=False)
        
        points = points[choice]

        # 7. 存入 results (為了相容你的 PointToMultiViewDepth，這裡存 anchor_points)
        results['anchor_points'] = points.astype(np.float32)
        
        # 為了相容 mmdet3d 的某些習慣，也可以存 'points'
        # results['points'] = points 

        return results

    def __repr__(self):
        return self.__class__.__name__

@OPENOCC_TRANSFORMS.register_module()
class GlobalRotScaleTrans(object):
    """
    生成 BDA (BEV Data Augmentation) 參數與矩陣。
    這會影響 View Transformer 的投影，以及 Occupancy GT 的翻轉。
    """
    def __init__(self,
                 rot_range=[-0.3925, 0.3925], # +/- 22.5度
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0],
                 flip_dx_ratio=0.5,
                 flip_dy_ratio=0.5):
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std
        self.flip_dx_ratio = flip_dx_ratio
        self.flip_dy_ratio = flip_dy_ratio

    def __call__(self, input_dict):
        # 1. Random Sample Params
        rot_angle = np.random.uniform(*self.rot_range)
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        flip_dx = np.random.rand() < self.flip_dx_ratio
        flip_dy = np.random.rand() < self.flip_dy_ratio
        
        # 2. Build BDA Matrix (3x3)
        # Rotation
        rot_sin = np.sin(rot_angle)
        rot_cos = np.cos(rot_angle)
        rot_mat = np.array([[rot_cos, -rot_sin, 0], 
                            [rot_sin,  rot_cos, 0], 
                            [0,          0,     1]], dtype=np.float32)
        
        # Scale
        scale_mat = np.array([[scale_ratio, 0, 0], 
                              [0, scale_ratio, 0], 
                              [0, 0, scale_ratio]], dtype=np.float32)
        
        # Flip (X and Y)
        flip_mat = np.eye(3, dtype=np.float32)
        if flip_dx: 
            flip_mat = flip_mat @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        if flip_dy: 
            flip_mat = flip_mat @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)

        # Combine: Flip @ Scale @ Rot
        bda_mat = flip_mat @ (scale_mat @ rot_mat)
        
        # 3. Save to results
        input_dict['bda_mat'] = bda_mat
        input_dict['bda_rot_angle'] = rot_angle
        input_dict['bda_scale'] = scale_ratio
        input_dict['flip_dx'] = flip_dx
        input_dict['flip_dy'] = flip_dy

        # 4. 如果有點雲 (points/anchor_points)，也要跟著轉
        # 這是為了讓 Depth Loss 計算正確 (Depth 是由點雲生成的)
        # if 'anchor_points' in input_dict:
        #     points = input_dict['anchor_points']
        #     # points: (N, 3) or (N, 4)
        #     # apply rotation/scale/flip
        #     points[:, :3] = points[:, :3] @ bda_mat.T
        #     input_dict['anchor_points'] = points

        return input_dict

@OPENOCC_TRANSFORMS.register_module()
class PointToMultiViewDepth(object):
    def __init__(self, grid_config, downsample=16):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        """
        將點雲投影到圖像平面生成深度圖
        points: (N, 3) [u, v, d]
        """
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        
        # 座標縮放
        coor = torch.round(points[:, :2] / self.downsample) 
        depth = points[:, 2]

        # 篩選在圖像範圍內的點 + 深度範圍內的點
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & \
                (coor[:, 1] >= 0) & (coor[:, 1] < height) & \
                (depth < self.grid_config['depth'][1]) & \
                (depth >= self.grid_config['depth'][0])
        
        coor, depth = coor[kept1], depth[kept1]
        
        # 處理重疊點：保留深度最小的 (最近的)
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        
        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results.get("points_lidar", results["anchor_points"])
        if isinstance(points_lidar, np.ndarray):
            points_lidar = torch.from_numpy(points_lidar).float()
        
        # 2. 準備投影參數 (需要 lidar2img)
        # 你的 pipeline 中，lidar2img 通常存放在 results['lidar2img'] (List of arrays)
        # 這些矩陣已經包含了 extrinsic + intrinsic
        
        lidar2imgs = results['lidar2img'] # List of 4x4 numpy arrays (6 cams)
        img_aug_matrices = results.get('img_aug_matrix', None) # List of 4x4 (6 cams)
        
        # 圖像尺寸 (假設所有圖像尺寸相同，且已經 resize 過)
        # results['img_shape'] 是一個 list [(H,W), ...]
        H, W = results['img_shape'][0] 

        depth_map_list = []
        
        for i in range(len(lidar2imgs)):
            # A. 投影: Lidar -> Image (Pixel)
            # P = K @ R @ t
            l2i = torch.from_numpy(lidar2imgs[i]).float() # (4, 4)
            
            # 齊次坐標投影
            # points: (N, 3) -> (N, 4)
            points_h = torch.cat([points_lidar[:, :3], torch.ones_like(points_lidar[:, :1])], dim=1)
            
            # (4, 4) @ (4, N) -> (4, N) -> (N, 4)
            points_img = (l2i @ points_h.T).T 
            
            # 歸一化 (u, v, d)
            # u = x/z, v = y/z, d = z
            depth = points_img[:, 2]
            mask = depth > 1e-5
            points_img = points_img[mask]
            
            # [u, v, d]
            points_img = torch.cat([
                points_img[:, 0:1] / points_img[:, 2:3], 
                points_img[:, 1:2] / points_img[:, 2:3], 
                points_img[:, 2:3]
            ], dim=1)

            # B. 應用圖像增強 (Resize/Crop/Rotate)
            # 因為 img 已經被 ResizeCropFlipImage 變換過了，GT Depth 也要對齊
            if img_aug_matrices is not None:
                aug_mat = torch.from_numpy(img_aug_matrices[i]).float() # (4, 4)
                
                # 增強矩陣通常是 3x3 (在 2D 平面上) 或 4x4
                # u' = A @ u
                # points_img: (N, 3) -> (u, v, d)
                # 這裡要小心：aug_mat 是針對 (u, v, 1) 的，d 應該保持不變
                
                # 建構齊次 uv: (N, 3) [u, v, 1]
                uv1 = torch.cat([points_img[:, :2], torch.ones_like(points_img[:, :1])], dim=1)
                
                # 應用增強: (4, 4) @ (N, 4)^T ? 不，aug_mat 通常是 4x4 但只用前 3x3 作用於 2D
                # 你的 ResizeCropFlipImage 輸出的 4x4 是:
                # [R00 R01 0 tx]
                # [R10 R11 0 ty]
                # ...
                
                # 我們只取前 3x3 (2D homography) 作用於 pixel coords
                # 但要注意 aug_mat 4x4 的定義。在你的代碼中：
                # mat4[:2, :2] = ida_mat3[:2, :2]
                # mat4[:2,  3] = ida_mat3[:2,  2] 
                # 所以是標準的 projection matrix 格式
                
                # [u', v', w']^T = Aug @ [u, v, 0, 1]^T
                # 簡化算法：
                uv1_aug = (aug_mat[:3, :3] @ torch.cat([uv1[:, :2], torch.ones_like(uv1[:, 2:])], dim=1).T).T
                # 或是直接用 2D 變換邏輯
                
                u_new = uv1[:, 0] * aug_mat[0, 0] + uv1[:, 1] * aug_mat[0, 1] + aug_mat[0, 3]
                v_new = uv1[:, 0] * aug_mat[1, 0] + uv1[:, 1] * aug_mat[1, 1] + aug_mat[1, 3]
                
                points_img[:, 0] = u_new
                points_img[:, 1] = v_new
                # depth (index 2) 不變

            # C. 生成 Depth Map
            depth_map = self.points2depthmap(points_img, H, W)
            depth_map_list.append(depth_map)

        # Stack -> (N_views, H/down, W/down)
        results['gt_depth'] = torch.stack(depth_map_list)
        return results

@OPENOCC_TRANSFORMS.register_module()
class ResizeCropFlipImage(object):
    def __call__(self, results):
        aug_configs = results.get("aug_configs")
        if aug_configs is None:
            # 路線B：沒做 aug 就給 identity
            N = len(results["img"])
            results["img_aug_matrix"] = np.tile(np.eye(4, dtype=np.float32)[None], (N, 1, 1))
            return results

        resize, resize_dims, crop, flip, rotate = aug_configs
        imgs = results["img"]
        N = len(imgs)

        new_imgs = []
        ida_mats = []

        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            img, ida_mat3 = self._img_transform(
                img, resize=resize, resize_dims=resize_dims, crop=crop, flip=flip, rotate=rotate
            )

            new_imgs.append(np.array(img).astype(np.float32))

            # ✅ 3x3 -> 4x4 (把 tx,ty 放到 col=3)
            mat4 = np.eye(4, dtype=np.float32)
            ida_mat3 = ida_mat3.cpu().numpy().astype(np.float32)
            mat4[:2, :2] = ida_mat3[:2, :2]
            mat4[:2,  3] = ida_mat3[:2,  2]
            ida_mats.append(mat4)

            # ✅ 路線B：把 augmentation bake 進 lidar2img/ego2img
            # results["lidar2img"][i] = mat4 @ results["lidar2img"][i]
            # if "ego2img" in results:
            #     results["ego2img"][i] = mat4 @ results["ego2img"][i]

        results["img"] = new_imgs
        results["img_shape"] = [x.shape[:2] for x in new_imgs]  # [(H,W), ...]
        results["img_aug_matrix"] = np.stack(ida_mats, axis=0).astype(np.float32)  # [N,4,4]
        return results


    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)

        # --- 0) 先記錄原圖尺寸（PIL: W,H）
        oldW, oldH = img.size

        # --- 1) resize
        # resize_dims 是 (newW, newH)
        newW, newH = resize_dims
        img = img.resize(resize_dims)

        # ✅ 用 sx, sy（不要用單一 resize）
        sx = float(newW) / float(oldW)
        sy = float(newH) / float(oldH)
        S = torch.tensor([[sx, 0.0], [0.0, sy]], dtype=torch.float32)
        ida_rot = S.matmul(ida_rot)

        # --- 2) crop
        img = img.crop(crop)
        ida_tran -= torch.tensor(crop[:2], dtype=torch.float32)

        # --- 3) flip
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
            A = torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
            b = torch.tensor([crop[2] - crop[0], 0.0], dtype=torch.float32)
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b

        # --- 4) rotate (around center, expand=False)
        img = img.rotate(rotate)
        A = self._get_rot(rotate / 180 * np.pi).float()
        b = torch.tensor([crop[2] - crop[0], crop[3] - crop[1]], dtype=torch.float32) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b

        # --- 5) build 3x3
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat



@OPENOCC_TRANSFORMS.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if random.randint(2):
                delta = random.uniform(
                    -self.brightness_delta, self.brightness_delta
                )
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results["img"] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged', crop_size=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.crop_size = crop_size

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.crop_size is not None:
            img = img[:self.crop_size[0], :self.crop_size[1]]
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['ori_img'] = deepcopy(img)
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadPointFromFile(object):
    def __init__(self, pc_range, num_pts, use_ego=False, keep_normalized=False):
        self.use_ego = use_ego
        self.pc_range = pc_range
        self.num_pts = num_pts
        self.keep_normalized = keep_normalized  # 想保留 0~1 再開

    def __call__(self, results):
        pts_path = results["pts_filename"]
        scan = np.fromfile(pts_path, dtype=np.float32).reshape((-1, 5))[:, :4]
        scan[:, 3] = 1.0  # (N,4) homogeneous

        if self.use_ego:
            ego2lidar = results["ego2lidar"]          # ego -> lidar
            lidar2ego = np.linalg.inv(ego2lidar)      # lidar -> ego
            scan = (lidar2ego[None, ...] @ scan[..., None]).squeeze(-1)

        scan = scan[:, :3]  # (N,3) meters

        # -------------------------
        # Filter by pc_range (meters)
        # -------------------------
        norm = np.linalg.norm(scan, 2, axis=-1)
        mask = (
            (scan[:, 0] > self.pc_range[0]) & (scan[:, 0] < self.pc_range[3]) &
            (scan[:, 1] > self.pc_range[1]) & (scan[:, 1] < self.pc_range[4]) &
            (scan[:, 2] > self.pc_range[2]) & (scan[:, 2] < self.pc_range[5]) &
            (norm > 1.0)
        )
        scan = scan[mask]

        # 如果點數太少，避免後面 np.random.choice 爆掉
        if scan.shape[0] == 0:
            # fallback：塞一個原點附近的點（或你也可以 raise）
            scan = np.zeros((1, 3), dtype=np.float32)

        # -------------------------
        # Sampling to fixed num_pts (still meters)
        # -------------------------
        if scan.shape[0] < self.num_pts:
            choice = np.random.choice(scan.shape[0], self.num_pts, replace=True)
        else:
            choice = np.random.choice(scan.shape[0], self.num_pts, replace=False)
        scan_m = scan[choice].astype(np.float32)  # meters

        # ✅ meters 給 depth 投影用
        results["points_lidar"] = scan_m
        results["anchor_points"] = scan_m  # 建議 anchor_points 也放 meters，避免誤用

        # （可選）如果你真的有模組需要 0~1，再另外給一份
        if self.keep_normalized:
            scan_norm = scan_m.copy()
            scan_norm[:, 0] = (scan_norm[:, 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
            scan_norm[:, 1] = (scan_norm[:, 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
            scan_norm[:, 2] = (scan_norm[:, 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
            results["anchor_points_norm"] = scan_norm.astype(np.float32)

        return results

    def __repr__(self):
        return self.__class__.__name__



@OPENOCC_TRANSFORMS.register_module()
class LoadPseudoPointFromFile(object):

    def __init__(self, datapath, pc_range, num_pts, is_ego=True, use_ego=False):
        self.datapath = datapath
        self.is_ego = is_ego
        self.use_ego = use_ego
        self.pc_range = pc_range
        self.num_pts = num_pts
        pass

    def __call__(self, results):
        pts_path = os.path.join(self.datapath, f"{results['sample_idx']}.npy")
        scan = np.load(pts_path)
        if self.is_ego and (not self.use_ego):
            ego2lidar = results['ego2lidar']
            scan = np.concatenate([scan, np.ones_like(scan[:, :1])], axis=-1)
            scan = ego2lidar[None, ...] @ scan[..., None] # p, 4, 1
            scan = np.squeeze(scan, axis=-1)

        if (not self.is_ego) and self.use_ego:
            ego2lidar = results['ego2lidar']
            lidar2ego = np.linalg.inv(ego2lidar)
            scan = np.concatenate([scan, np.ones_like(scan[:, :1])], axis=-1)
            scan = lidar2ego[None, ...] @ scan[..., None]
            scan = np.squeeze(scan, axis=-1)
        
        scan = scan[:, :3] # n, 3

        ### filter
        norm = np.linalg.norm(scan, 2, axis=-1)
        mask = (scan[:, 0] > self.pc_range[0]) & (scan[:, 0] < self.pc_range[3]) & \
            (scan[:, 1] > self.pc_range[1]) & (scan[:, 1] < self.pc_range[4]) & \
            (scan[:, 2] > self.pc_range[2]) & (scan[:, 2] < self.pc_range[5]) & \
            (norm > 1.0)
        scan = scan[mask]

        ### append
        if scan.shape[0] < self.num_pts:
            multi = int(math.ceil(self.num_pts * 1.0 / scan.shape[0])) - 1
            scan_ = np.repeat(scan, multi, 0)
            scan_ = scan_ + np.random.randn(*scan_.shape) * 0.3
            scan_ = scan_[np.random.choice(scan_.shape[0], self.num_pts - scan.shape[0], False)]
            scan_[:, 0] = np.clip(scan_[:, 0], self.pc_range[0], self.pc_range[3])
            scan_[:, 1] = np.clip(scan_[:, 1], self.pc_range[1], self.pc_range[4])
            scan_[:, 2] = np.clip(scan_[:, 2], self.pc_range[2], self.pc_range[5])
            scan = np.concatenate([scan, scan_], 0)
        else:
            scan = scan[np.random.choice(scan.shape[0], self.num_pts, False)]
        
        scan[:, 0] = (scan[:, 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        scan[:, 1] = (scan[:, 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        scan[:, 2] = (scan[:, 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
        results['anchor_points'] = scan
        
        return results
    
    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadOccupancySurroundOcc(object):

    def __init__(self, occ_path, semantic=False, use_ego=False, use_sweeps=False, perturb=False):
        self.occ_path = occ_path
        self.semantic = semantic
        self.use_ego = use_ego
        assert semantic and (not use_ego)
        self.use_sweeps = use_sweeps
        self.perturb = perturb

        xyz = self.get_meshgrid([-50, -50, -5.0, 50, 50, 3.0], [200, 200, 16], 0.5)
        self.xyz = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1) # x, y, z, 4

    def get_meshgrid(self, ranges, grid, reso):
        xxx = torch.arange(grid[0], dtype=torch.float) * reso + 0.5 * reso + ranges[0]
        yyy = torch.arange(grid[1], dtype=torch.float) * reso + 0.5 * reso + ranges[1]
        zzz = torch.arange(grid[2], dtype=torch.float) * reso + 0.5 * reso + ranges[2]

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([
            xxx, yyy, zzz
        ], dim=-1).numpy()
        return xyz # x, y, z, 3

    def __call__(self, results):
        # 1. 原本的讀取邏輯
        label_file = os.path.join(self.occ_path, results['pts_filename'].split('/')[-1]+'.npy')
        
        # 預設全空 (17 = empty/free)
        # 假設 grid size 為 [200, 200, 16]
        new_label = np.ones((200, 200, 16), dtype=np.int64) * 17 

        if os.path.exists(label_file):
            label = np.load(label_file)
            # label 格式通常是 [x, y, z, cls]
            new_label[label[:, 0], label[:, 1], label[:, 2]] = label[:, 3]
        elif self.use_sweeps:
            # 如果是 sweeps 模式且沒檔案，保持全空
            pass
        else:
            raise NotImplementedError

        # ✅ BDA: 用 bda_mat 對 new_label 做 rot/scale/flip（FlashOcc/BEVDet 同方向）
        # =========================================================
        empty_idx = 17
        pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]

        Dx, Dy, Dz = new_label.shape
        vx = (pc_range[3] - pc_range[0]) / Dx  # 0.5
        vy = (pc_range[4] - pc_range[1]) / Dy  # 0.5

        if "bda_mat" in results and results["bda_mat"] is not None:
            bda = results["bda_mat"]

            # bda 可能是 torch / np
            if hasattr(bda, "detach"):
                bda = bda.detach().cpu().numpy()
            bda = np.asarray(bda, dtype=np.float32)

            # 允許 (4,4) -> 取 (3,3)
            if bda.shape == (4, 4):
                bda = bda[:3, :3]

            assert bda.shape == (3, 3), f"bda_mat shape should be (3,3), got {bda.shape}"

            # 取出非 empty 的 voxel（empty=17）
            idx = np.where(new_label != empty_idx)
            if idx[0].size > 0:
                ix, iy, iz = idx
                cls = new_label[ix, iy, iz]

                # voxel center -> metric (x,y)
                x = pc_range[0] + (ix.astype(np.float32) + 0.5) * vx
                y = pc_range[1] + (iy.astype(np.float32) + 0.5) * vy

                xy1 = np.stack([x, y, np.ones_like(x)], axis=1)  # (M,3)
                xy2 = (xy1 @ bda.T)[:, :2]                      # (M,2)

                ix2 = np.floor((xy2[:, 0] - pc_range[0]) / vx).astype(np.int64)
                iy2 = np.floor((xy2[:, 1] - pc_range[1]) / vy).astype(np.int64)

                inside = (ix2 >= 0) & (ix2 < Dx) & (iy2 >= 0) & (iy2 < Dy)

                out = np.ones_like(new_label, dtype=np.int64) * empty_idx
                out[ix2[inside], iy2[inside], iz[inside]] = cls[inside]
                new_label = out

        # 2. 產生 mask 和輸出
        mask = np.ones_like(new_label, dtype=bool)
        results['occ_label'] = new_label if self.semantic else new_label != 17
        results['occ_cam_mask'] = mask

        # 3. 處理 occ_xyz (座標網格)
        # 注意：如果你的 BDA 只有 flip/scale，occ_xyz 通常不需要動，
        # 因為 ViewTransformer 會根據 bda_mat 把特徵投影到正確的翻轉位置。
        xyz = self.xyz.copy()
        if getattr(self, "perturb", False):
            norm_distribution = np.clip(np.random.randn(*xyz.shape[:-1], 3) / 6, -0.5, 0.5)
            xyz[..., :3] = xyz[..., :3] + norm_distribution * 0.49

        if not self.use_ego:
            occ_xyz = xyz[..., :3]
        else:
            ego2lidar = np.linalg.inv(results['ego2lidar'])
            occ_xyz = ego2lidar[None, None, None, ...] @ xyz[..., None]
            occ_xyz = np.squeeze(occ_xyz, -1)[..., :3]
            
        results['occ_xyz'] = occ_xyz
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadOccupancyKITTI360(object):

    def __init__(self, occ_path, semantic=False, unknown_to_empty=False, training=False):
        self.occ_path = occ_path
        self.semantic = semantic

        xyz = self.get_meshgrid([0.0, -25.6, -2.0, 51.2, 25.6, 4.4], [256, 256, 32], 0.2)
        self.xyz = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1) # x, y, z, 4
        self.unknown_to_empty = unknown_to_empty
        self.training = training

    def get_meshgrid(self, ranges, grid, reso):
        xxx = torch.arange(grid[0], dtype=torch.float) * reso + 0.5 * reso + ranges[0]
        yyy = torch.arange(grid[1], dtype=torch.float) * reso + 0.5 * reso + ranges[1]
        zzz = torch.arange(grid[2], dtype=torch.float) * reso + 0.5 * reso + ranges[2]

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([
            xxx, yyy, zzz
        ], dim=-1).numpy()
        return xyz # x, y, z, 3

    def __call__(self, results):        
        occ_xyz = self.xyz[..., :3].copy()
        results['occ_xyz'] = occ_xyz

        ## read occupancy label
        label_path = os.path.join(
            self.occ_path, results['sequence'], "{}_1_1.npy".format(results['token']))
        label = np.load(label_path).astype(np.int64)
        if getattr(self, "unknown_to_empty", False) and getattr(self, "training", False):
            label[label == 255] = 0

        results['occ_cam_mask'] = (label != 255)
        results['occ_label'] = label
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str
