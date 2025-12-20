# model/student.py
import torch
import torch.nn as nn
import numpy as np

from model.backbone.backbone import MultiViewResNetBackbone
from model.view_transformer import GeometricLSSViewTransformer
from model.encoder.bev_encoder import CustomResNet_NoMM
from model.neck.bev_neck import FPNLSS_Torch
from model.head.occ_head import BEVOccHead2D_V2
from model.neck.neck import FPN_Torch


class OccStudent(nn.Module):
    """
    GaussianFormer-風格的 BEV 2D Student：
      img -> MultiViewResNetBackbone
          -> FPN
          -> GeometricLSSViewTransformer (完整 LSS)
          -> BEV encoder (CustomResNet_NoMM)
          -> BEV neck (FPNLSS_Torch)
          -> BEV Occ Head (BEVOccHead2D_V2)
    """

    def __init__(
        self,
        bev_h: int = 200,
        bev_w: int = 200,
        depth_bins: int = 16,
        num_classes: int = 18,
        backbone_pretrained: bool = True,
        backbone_frozen_stages: int = 1,
        input_size=(480, 640),
        numC_Trans: int = 128,
        pc_range=(-50.0, -50.0, -5.0, 50.0, 50.0, 3.0),
        vt_depth_min: float = 0.5,
        vt_depth_max: float = 45.0,
        vt_normalize: bool = True,
        vt_depthnet_mid: int = 256,
        vt_depthnet_with_cp: bool = False,
        vt_depthnet_use_aspp: bool = True,
        vt_hw_chunk: int = None,
        bev_encoder_channels=(128, 256, 512),
        bev_out_channels: int = 256,
    ):
        super().__init__()

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.depth_bins = depth_bins
        self.num_classes = num_classes
        self.last_depth_loss = None  # training/val 時會更新，給外面 loop 讀

        # 1) 多視角 backbone
        self.backbone = MultiViewResNetBackbone(
            pretrained=backbone_pretrained,
            return_layers=[1, 2, 3, 4],
        )

        # 2) FPN
        self.neck = FPN_Torch(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256,
        )

        # 3) Geometric LSS ViewTransformer
        self.vt = GeometricLSSViewTransformer(
            in_channels=256,
            out_channels=numC_Trans,
            bev_h=bev_h,
            bev_w=bev_w,
            pc_range=pc_range,
            input_size=input_size,
            num_depth_bins=depth_bins,
            depth_min=vt_depth_min,
            depth_max=vt_depth_max,
            normalize=vt_normalize,
            depthnet_mid=vt_depthnet_mid,
            depthnet_with_cp=vt_depthnet_with_cp,
            depthnet_use_aspp=vt_depthnet_use_aspp,
            hw_chunk=vt_hw_chunk,
        )

        # 4) BEV encoder（multi-scale）
        self.bev_encoder = CustomResNet_NoMM(
            numC_input=numC_Trans,
            num_layer=(2, 2, 2),
            num_channels=bev_encoder_channels,
            stride=(2, 2, 2),
            backbone_output_ids=(0, 1, 2),
            norm_cfg=dict(type="BN"),
            with_cp=False,
            block_type="Basic",
            downsample_with_norm=True,
        )

        # 5) BEV neck (FPN)
        self.bev_neck = FPNLSS_Torch(
            in_channels_list=self.bev_encoder.out_channels_list,
            out_channels=bev_out_channels,
        )

        # 6) Occ head
        self.head = BEVOccHead2D_V2(
            in_dim=bev_out_channels,
            out_dim=bev_out_channels,
            Dz=depth_bins,
            use_mask=True,
            num_classes=num_classes,
            use_predicter=True,
            class_balance=True,
            empty_idx=num_classes - 1,
        )

    # -------------------------
    # helpers
    # -------------------------
    @staticmethod
    def _stack_list_as_tensor(xs, device, dtype, name="tensor_list"):
        """
        xs: list/tuple of (Tensor/ndarray/list scalar...) with SAME SHAPE
        return: stacked tensor [B, ...]
        """
        if xs is None:
            return None
        if len(xs) == 0:
            raise ValueError(f"[{name}] empty list")

        ts = []
        for v in xs:
            if v is None:
                return None
            # mmcv DataContainer 相容（保險）
            if hasattr(v, "data"):
                v = v.data
            if torch.is_tensor(v):
                ts.append(v.to(device=device, dtype=dtype))
            else:
                ts.append(torch.as_tensor(v, device=device, dtype=dtype))
        try:
            return torch.stack(ts, dim=0)
        except Exception as e:
            shapes = [tuple(t.shape) for t in ts]
            raise ValueError(f"[{name}] cannot stack list; shapes={shapes}") from e

    @staticmethod
    def _to_tensor(x, device, dtype):
        """
        ✅ 修正版：支援 numpy.object_ / list-of-arrays（val 常見）
        """
        if x is None:
            return None

        # mmcv DataContainer 相容（保險）
        if hasattr(x, "data"):
            x = x.data

        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype)

        # list/tuple -> stack（最常見 val 爆點）
        if isinstance(x, (list, tuple)):
            return OccStudent._stack_list_as_tensor(x, device=device, dtype=dtype, name="batch_field")

        if isinstance(x, np.ndarray):
            if x.dtype == object:
                # 例如 array([arr1, arr2, ...], dtype=object)
                xs = list(x)
                return OccStudent._stack_list_as_tensor(xs, device=device, dtype=dtype, name="object_ndarray")
            # 一般數值 ndarray
            return torch.from_numpy(x).to(device=device, dtype=dtype)

        # 其他型別（例如 nested python list）
        return torch.as_tensor(x, device=device, dtype=dtype)

    @staticmethod
    def _to_depth_tensor(x, device):
        """
        ✅ 專門處理 gt_depth：避免 np.asarray(list)->dtype=object 造成 torch.from_numpy 爆掉
        """
        if x is None:
            return None

        if hasattr(x, "data"):
            x = x.data

        if torch.is_tensor(x):
            return x.to(device=device, dtype=torch.float32)

        if isinstance(x, np.ndarray):
            if x.dtype == object:
                xs = [np.asarray(v, dtype=np.float32) for v in list(x)]
                x = np.stack(xs, axis=0)
            else:
                x = x.astype(np.float32, copy=False)
            return torch.from_numpy(x).to(device=device, dtype=torch.float32)

        if isinstance(x, (list, tuple)):
            xs = []
            for v in x:
                if v is None:
                    return None
                if torch.is_tensor(v):
                    v = v.detach().cpu().numpy()
                xs.append(np.asarray(v, dtype=np.float32))
            x = np.stack(xs, axis=0)
            return torch.from_numpy(x).to(device=device, dtype=torch.float32)

        return torch.as_tensor(x, device=device, dtype=torch.float32)

    @staticmethod
    def _ensure_batch_cam(x: torch.Tensor, B: int, N: int, name: str):
        """
        允許輸入：
          (B,N,...) -> pass
          (N,...)   -> 自動補成 (1,N,...)
          (...)     -> 視情況補 (1,1,...)
        """
        if x is None:
            return None
        if x.dim() >= 2 and x.shape[0] == B and x.shape[1] == N:
            return x
        if x.dim() >= 1 and x.shape[0] == N:
            return x.unsqueeze(0)
        if x.dim() == 0:
            return x.view(1, 1)
        raise ValueError(f"[{name}] unexpected shape {tuple(x.shape)} (expect (B,N,...) or (N,...))")

    def _build_metas_from_batch(self, batch, device):
        # ---------------------------------------------------------
        # 1) projection_mat / lidar2img
        # ---------------------------------------------------------
        proj = batch.get("projection_mat", None)
        if proj is None:
            proj = batch.get("lidar2img", None)

        if proj is None:
            raise KeyError(
                "batch missing both 'projection_mat' and 'lidar2img'. "
                f"available keys={list(batch.keys())}"
            )

        proj = self._to_tensor(proj, device=device, dtype=torch.float32)  # 幾何用 fp32

        if proj.dim() == 3:
            # (N,4,4) -> (1,N,4,4)
            proj = proj.unsqueeze(0)
        if proj.dim() != 4 or proj.shape[-2:] != (4, 4):
            raise ValueError(f"[projection] expect (B,N,4,4), got {tuple(proj.shape)}")

        B, N = int(proj.shape[0]), int(proj.shape[1])

        # ---------------------------------------------------------
        # 2) img_aug_matrix (optional)
        # ---------------------------------------------------------
        img_aug = batch.get("img_aug_matrix", None)
        if img_aug is None:
            img_aug = torch.eye(4, device=device, dtype=torch.float32).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        else:
            img_aug = self._to_tensor(img_aug, device=device, dtype=torch.float32)
            if img_aug.dim() == 3:
                img_aug = img_aug.unsqueeze(0)
            img_aug = self._ensure_batch_cam(img_aug, B, N, "img_aug_matrix")

        # ---------------------------------------------------------
        # 3) bda_mat (optional, prefer (B,3,3))
        # ---------------------------------------------------------
        bda_mat = batch.get("bda_mat", None)
        if bda_mat is None:
            bda_mat = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1)
        else:
            bda_mat = self._to_tensor(bda_mat, device=device, dtype=torch.float32)
            if bda_mat.dim() == 2:
                bda_mat = bda_mat.unsqueeze(0).repeat(B, 1, 1)
            if bda_mat.dim() == 3 and bda_mat.shape[-1] == 4:
                bda_mat = bda_mat[..., :3, :3]
            if bda_mat.dim() != 3 or bda_mat.shape[-2:] != (3, 3):
                raise ValueError(f"[bda_mat] expect (B,3,3) or (3,3), got {tuple(bda_mat.shape)}")

        metas = {
            "lidar2img": proj,
            "img_aug_matrix": img_aug,
            "bda_mat": bda_mat,
        }

        # ---------------------------------------------------------
        # 4) K/R/t (optional)
        # ---------------------------------------------------------
        for key in ["K", "R", "t"]:
            v = batch.get(key, None)
            if v is None:
                continue
            v = self._to_tensor(v, device=device, dtype=torch.float32)
            if key in ["K", "R"]:
                if v.dim() == 3:
                    v = v.unsqueeze(0)  # (N,3,3)->(1,N,3,3)
                v = self._ensure_batch_cam(v, B, N, key)
                if v.shape[-2:] != (3, 3):
                    raise ValueError(f"[{key}] expect (...,3,3), got {tuple(v.shape)}")
            else:
                if v.dim() == 2:
                    v = v.unsqueeze(0)  # (N,3)->(1,N,3)
                v = self._ensure_batch_cam(v, B, N, key)
                if v.shape[-1] != 3:
                    raise ValueError(f"[t] expect (...,3), got {tuple(v.shape)}")
            metas[key] = v

        if "lidar2ego" in batch and batch["lidar2ego"] is not None:
            metas["lidar2ego"] = self._to_tensor(batch["lidar2ego"], device=device, dtype=torch.float32)

        # ---------------------------------------------------------
        # 5) Debug print once
        # ---------------------------------------------------------
        if not hasattr(self, "_dbg_once_metas_v3"):
            self._dbg_once_metas_v3 = True

            def _stat(x):
                if x is None:
                    return "(None)"
                if hasattr(x, "data"):
                    x = x.data
                if torch.is_tensor(x):
                    xd = x.detach()
                    try:
                        mn = xd.min().item()
                        mx = xd.max().item()
                        mm = f"min={mn:.3f} max={mx:.3f}"
                    except Exception:
                        mm = ""
                    return f"shape={tuple(xd.shape)} dtype={xd.dtype} device={xd.device} {mm}"
                if isinstance(x, np.ndarray):
                    return f"shape={x.shape} dtype={x.dtype} (numpy)"
                return f"type={type(x)}"

            print("\n========== [OccStudent] Batch & Metas Debug (Print Once) ==========")
            keys_to_check = ["img", "projection_mat", "lidar2img", "img_aug_matrix", "bda_mat", "K", "R", "t", "occ_label", "gt_depth"]
            for k in keys_to_check:
                if k in batch:
                    print(f"[Batch] {k:15s}: {_stat(batch[k])}")
                else:
                    print(f"[Batch] {k:15s}: (MISSING)")

            print("-" * 60)
            for k, v in metas.items():
                print(f"[Metas] {k:15s}: {_stat(v)}")

            print("-" * 60)
            bda_sample = metas["bda_mat"][0].detach().cpu().numpy()
            print(f"[Check] BDA Matrix[0]:\n{bda_sample}")
            if np.allclose(bda_sample, np.eye(3), atol=1e-5):
                print(">> ⚠️  BDA is Identity (可能 rot=0 或沒做 BDA aug).")
            else:
                print(">> ✅  BDA has transformation (Augmentation is working).")

            if ("K" in metas) and ("R" in metas) and ("t" in metas):
                K0 = metas["K"][0, 0]
                R0 = metas["R"][0, 0]
                t0 = metas["t"][0, 0]
                P0 = metas["lidar2img"][0, 0]

                K4 = torch.eye(4, device=device, dtype=torch.float32)
                K4[:3, :3] = K0

                Rt = torch.eye(4, device=device, dtype=torch.float32)
                Rt[:3, :3] = R0
                Rt[:3, 3] = t0

                P_hat = K4 @ Rt
                err = (P_hat - P0).abs().max().item()
                print(f"[Check] max|K@[R|t] - projection_mat| = {err:.6f}")
                if err > 1e-2:
                    print(">> ❌ 誤差偏大：通常代表 R/t 不是 lidar->cam，或 projection_mat 中途被額外乘了 aug/bda（double apply）。")
                else:
                    print(">> ✅ K/R/t 與 projection_mat 一致。")

            print("===================================================================\n")

        return metas

    # -------------------------
    # forward
    # -------------------------
    def forward(self, batch: dict, return_feats: bool = False) -> torch.Tensor:
        device = next(self.parameters()).device

        imgs = batch["img"].to(device)  # [B,N,3,H,W]
        B, N, C, H, W = imgs.shape

        # safety check
        assert (H, W) == self.vt.input_size, f"Config mismatch! Model expects {self.vt.input_size} but got {(H, W)}"

        # backbone + FPN
        img_feats = self.backbone(imgs)     # list of [B,N,C_i,H_i,W_i]
        fpn_feats = self.neck(img_feats)    # list of [B,N,256,Hf,Wf]

        # 你選的 FPN level（維持你原本用法）
        feature_for_vt = fpn_feats[2]       # (B,N,256,Hf,Wf)

        # metas
        metas = self._build_metas_from_batch(batch, device=device)

        # depth supervision (train/val 都可算，只要有 gt_depth)
        depth_loss = None
        has_depth = ("gt_depth" in batch) and (batch["gt_depth"] is not None)

        if has_depth:
            # ===== FIX: robust depth tensor conversion (handles numpy.object_ / list) =====
            gt_depth = self._to_depth_tensor(batch.get("gt_depth", None), device=device)

            if gt_depth is None:
                # depth 缺失就退化成不算 depth loss
                bev_feat_lss = self.vt(feature_for_vt, metas=metas)
                depth_loss = None
            else:
                Hf, Wf = feature_for_vt.shape[-2:]
                assert gt_depth.shape[:2] == (B, N), f"gt_depth BN mismatch: {gt_depth.shape[:2]} vs {(B, N)}"
                assert gt_depth.shape[-2:] == (Hf, Wf), f"gt_depth HW {gt_depth.shape[-2:]} != feat HW {(Hf, Wf)}"

                bev_feat_lss, depth_loss = self.vt(
                    feature_for_vt, metas=metas, gt_depths=gt_depth, return_loss=True
                )
        else:
            bev_feat_lss = self.vt(feature_for_vt, metas=metas)

        self.last_depth_loss = depth_loss

        # bev encoder + bev neck
        bev_feats_multi = self.bev_encoder(bev_feat_lss)
        bev_feat = self.bev_neck(bev_feats_multi)
        # occ head
        logits = self.head(bev_feat)
        if return_feats:
            return {
                "logits": logits,
                "bev_feat": bev_feat,         # distill point 1（BEV feature）
                "bev_feat_lss": bev_feat_lss, # 可選：你想更早對齊也行
            }
        else:
            return logits
