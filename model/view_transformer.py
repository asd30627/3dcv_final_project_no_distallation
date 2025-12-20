# model/view_transformer.py
from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.depthnet import DepthNet


# -------------------------
# Batched linear algebra helpers
# -------------------------
def _rq_decomposition(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched RQ decomposition for (...,3,3) matrix A.
    Returns R (upper-triangular), Q (orthonormal) such that A = R @ Q.
    Uses QR on reversed matrix.
    """
    A_flip = torch.flip(A, dims=(-2, -1)).transpose(-2, -1)  # (...,3,3)
    Q, R = torch.linalg.qr(A_flip)                            # (...,3,3)
    R = R.transpose(-2, -1)
    Q = Q.transpose(-2, -1)
    R = torch.flip(R, dims=(-2, -1))
    Q = torch.flip(Q, dims=(-2, -1))
    return R, Q


def _decompose_lidar2img_batch(
    lidar2img_4x4: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched decompose projection P = K [R|t] from lidar2img 4x4.
    lidar2img_4x4: (BN,4,4)
    Returns:
      K: (BN,3,3) upper-triangular with positive diag, normalized s.t. K[2,2]=1
      R: (BN,3,3)
      t: (BN,3)
    """
    P = lidar2img_4x4[:, :3, :4]      # (BN,3,4)
    M = P[:, :, :3]                   # (BN,3,3) = K @ R

    K, R = _rq_decomposition(M)       # (BN,3,3), (BN,3,3)

    diag = torch.diagonal(K, dim1=-2, dim2=-1)  # (BN,3)
    sign = torch.sign(diag)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    S = torch.diag_embed(sign)  # (BN,3,3)
    K = K @ S
    R = S @ R

    k22 = K[:, 2, 2].abs()
    scale = torch.where(k22 > 1e-6, k22, torch.ones_like(k22))
    K = K / scale.view(-1, 1, 1)

    p4 = P[:, :, 3]                    # (BN,3)
    K = K.float()
    p4 = p4.float()
    t = torch.linalg.solve(K, p4)      # (BN,3)

    return K, R, t


def _make_pixel_grid(
    Hf: int,
    Wf: int,
    H_in: int,
    W_in: int,
    device,
    dtype,
):
    """
    Pixel grid in network input image coordinates for feature map (Hf,Wf).
    Use pixel centers.
    Returns uv1: (HW,3) [u,v,1] in input pixel coords.
    """
    sx = float(W_in) / float(Wf)
    sy = float(H_in) / float(Hf)

    xs = (torch.arange(Wf, device=device, dtype=dtype) + 0.5) * sx
    ys = (torch.arange(Hf, device=device, dtype=dtype) + 0.5) * sy
    v, u = torch.meshgrid(ys, xs, indexing="ij")  # (Hf,Wf)
    ones = torch.ones_like(u)
    uv1 = torch.stack([u, v, ones], dim=-1).reshape(-1, 3)  # (HW,3)
    return uv1


class GeometricLSSViewTransformer(nn.Module):
    """
    Geometric Lift-Splat-Shoot view transformer (Pure PyTorch)

    ✅ 本版強化重點
    1) 支援 gt_depths 兩種尺寸：
       (A) (B,N,H_in,W_in)==input_size -> official min-pooling downsample
       (B) (B,N,Hf,Wf) feature-size -> 直接 bucketize (你目前就是這個)
    2) 支援 img_aug_matrix 3x3 / 4x4（你 pipeline 是 4x4）
    3) 加入一次性幾何自檢：抓 K/R/t mismatch & 抓 img_aug bake-in (double apply)
    4) 幾何運算全部 float32，避免 fp16 漂
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bev_h: int,
        bev_w: int,
        pc_range: Tuple[float, float, float, float, float, float],
        input_size: Tuple[int, int],
        num_depth_bins: int = 16,
        depth_min: float = 0.0,
        depth_max: float = 45.0,
        normalize: bool = True,
        depthnet_mid: int = 256,
        depthnet_with_cp: bool = False,
        depthnet_use_aspp: bool = True,
        hw_chunk: Optional[int] = None,
        loss_depth_weight: float = 3.0,
        apply_bda: bool = True,
        # ===== NEW =====
        geom_check_once: bool = True,
        geom_tol: float = 1e-3,
        warn_aug_baked: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.input_size = input_size
        self.num_depth_bins = num_depth_bins
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.normalize = normalize
        self.hw_chunk = hw_chunk
        self.loss_depth_weight = loss_depth_weight
        self.apply_bda = apply_bda

        # ===== NEW =====
        self.geom_check_once = geom_check_once
        self.geom_tol = float(geom_tol)
        self.warn_aug_baked = warn_aug_baked
        self._dbg_once_geom = False

        self.depthnet = DepthNet(
            in_channels=in_channels,
            mid_channels=depthnet_mid,
            context_channels=out_channels,
            depth_channels=num_depth_bins,
            use_aspp=depthnet_use_aspp,
            with_cp=depthnet_with_cp,
        )

        self.bev_refine = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        x_min, y_min, z_min, x_max, y_max, z_max = pc_range
        self.x_min, self.y_min, self.z_min = x_min, y_min, z_min
        self.x_max, self.y_max, self.z_max = x_max, y_max, z_max
        self.dx = (x_max - x_min) / float(bev_w)
        self.dy = (y_max - y_min) / float(bev_h)

        # depth bins (centers)
        self.depth_step = (depth_max - depth_min) / float(num_depth_bins)
        centers = depth_min + (torch.arange(num_depth_bins, dtype=torch.float32) + 0.5) * self.depth_step
        self.register_buffer("depth_values", centers, persistent=False)

        self._cached_grid = None

    def _get_uv1_aug(self, Hf, Wf, device, dtype):
        key = (Hf, Wf, self.input_size[0], self.input_size[1], str(device), str(dtype))
        if self._cached_grid is None or self._cached_grid[0] != key:
            uv1 = _make_pixel_grid(Hf, Wf, self.input_size[0], self.input_size[1], device, dtype)
            self._cached_grid = (key, uv1)
        return self._cached_grid[1]

    @torch.no_grad()
    def _metas_to_batched_tensors(
        self,
        metas: Union[List[Dict], Dict],
        device,
        dtype,
    ) -> Dict[str, torch.Tensor]:
        # metas 已經是 dict(B,N,...)（你目前就是這種）
        if isinstance(metas, Dict):
            out = {}
            for k, v in metas.items():
                if torch.is_tensor(v):
                    out[k] = v.to(device=device, dtype=dtype)
                else:
                    out[k] = v
            return out

        # metas 是 list[dict] 的舊路徑（保留）
        assert isinstance(metas, list) and len(metas) > 0
        B = len(metas)

        l2i_list = []
        aug_list = []
        for m in metas:
            l2i_list.append(m["lidar2img"])
            aug_list.append(m["img_aug_matrix"])

        N = len(l2i_list[0])

        def _to_tensor(x):
            if torch.is_tensor(x):
                return x.to(device=device, dtype=dtype)
            return torch.tensor(x, device=device, dtype=dtype)

        lidar2img = torch.stack(
            [torch.stack([_to_tensor(l2i_list[b][n]) for n in range(N)], dim=0) for b in range(B)],
            dim=0,
        )  # (B,N,4,4)

        img_aug = torch.stack(
            [torch.stack([_to_tensor(aug_list[b][n]) for n in range(N)], dim=0) for b in range(B)],
            dim=0,
        )  # (B,N,4,4)

        out: Dict[str, torch.Tensor] = {
            "lidar2img": lidar2img,
            "img_aug_matrix": img_aug,
        }

        optional_keys = ["K", "R", "t", "bda_mat", "gt_depth", "gt_depths"]
        for k in optional_keys:
            if k in metas[0]:
                vv = []
                for b in range(B):
                    v = metas[b][k]
                    if torch.is_tensor(v):
                        vv.append(v.to(device=device))
                    else:
                        vv.append(torch.tensor(v, device=device))
                out[k] = torch.stack(vv, dim=0)

        return out

    # ===== NEW: robust parse img_aug (3x3 or 4x4) =====
    def _parse_img_aug(self, aug: torch.Tensor, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input:
          aug: (BN,4,4) or (BN,3,3)
        Return:
          aug4: (BN,4,4)
          post_rot: (BN,3,3)  (affine A)
          post_tran: (BN,3)   (translation, z component = 0)
        """
        aug = aug.to(device=device)
        if aug.dim() != 3:
            raise ValueError(f"img_aug_matrix must be (BN,*,*), got {tuple(aug.shape)}")

        if aug.shape[-1] == 4:
            aug4 = aug.float()
            post_rot = aug4[:, :3, :3].float()
            post_tran = aug4[:, :3, 3].float()
            return aug4, post_rot, post_tran

        if aug.shape[-1] == 3:
            # 3x3 homography style: [a b tx; c d ty; 0 0 1]
            H = aug.float()
            aug4 = torch.eye(4, device=device, dtype=torch.float32).view(1, 4, 4).repeat(H.shape[0], 1, 1)
            aug4[:, :3, :3] = H
            post_rot = H
            post_tran = torch.zeros((H.shape[0], 3), device=device, dtype=torch.float32)
            post_tran[:, 0] = H[:, 0, 2]
            post_tran[:, 1] = H[:, 1, 2]
            return aug4, post_rot, post_tran

        raise ValueError(f"Unsupported img_aug_matrix shape: {tuple(aug.shape)}")

    def _get_bda_bn(self, meta: Dict[str, torch.Tensor], B: int, N: int, device) -> torch.Tensor:
        bda = meta.get("bda_mat", None)
        if bda is None:
            return torch.eye(3, device=device, dtype=torch.float32).view(1, 3, 3).repeat(B * N, 1, 1)

        if bda.dim() == 3:
            if bda.shape[0] == B:
                bda_bn = bda.to(device=device).view(B, 1, 3, 3).repeat(1, N, 1, 1).reshape(B * N, 3, 3)
            elif bda.shape[0] == B * N:
                bda_bn = bda.to(device=device)
            else:
                raise ValueError(f"Unexpected bda shape: {tuple(bda.shape)}")
        elif bda.dim() == 4:
            if bda.shape[0] == B and bda.shape[1] == N:
                bda_bn = bda.to(device=device).reshape(B * N, 3, 3)
            else:
                raise ValueError(f"Unexpected bda shape: {tuple(bda.shape)}")
        else:
            raise ValueError(f"Unexpected bda dim: {bda.dim()}")

        return bda_bn.float()

    def _get_downsample_factors(self, Hf: int, Wf: int) -> Tuple[int, int]:
        H_in, W_in = self.input_size
        if H_in % Hf != 0 or W_in % Wf != 0:
            raise ValueError(f"input_size {self.input_size} not divisible by feat {(Hf, Wf)}")
        return H_in // Hf, W_in // Wf

    @torch.no_grad()
    def get_downsampled_gt_depth(self, gt_depths: torch.Tensor, Hf: int, Wf: int) -> torch.Tensor:
        """
        支援兩種 gt_depths：
        (A) (B,N,H_in,W_in)==input_size -> min-pooling downsample
        (B) (B,N,Hf,Wf) feature-size -> 直接 bucketize (你目前這個)
        回傳 onehot: (B*N*Hf*Wf, D)
        """
        assert gt_depths.dim() == 4, f"gt_depths must be (B,N,H,W), got {tuple(gt_depths.shape)}"
        B, N, H, W = gt_depths.shape
        D = self.num_depth_bins

        interval = float(self.depth_step)
        depth0 = float(self.depth_min)

        # Case (B): already downsampled
        if (H, W) == (Hf, Wf):
            d = gt_depths.reshape(B * N, Hf, Wf).reshape(-1)  # (BN*Hf*Wf,)
            valid = (d > 0.0) & (d >= depth0) & (d < (depth0 + D * interval))

            d_bin = torch.floor((d - depth0) / interval).to(torch.int64)
            d_bin = torch.clamp(d_bin, 0, D - 1)

            onehot = F.one_hot(d_bin, num_classes=D).float()
            onehot = onehot * valid.float().unsqueeze(-1)
            return onehot

        # Case (A): full input_size
        if (H, W) == (self.input_size[0], self.input_size[1]):
            ds_h, ds_w = self._get_downsample_factors(Hf, Wf)

            x = gt_depths.view(B * N, Hf, ds_h, Wf, ds_w, 1)
            x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
            x = x.view(-1, ds_h * ds_w)

            x_tmp = torch.where(x == 0.0, 1e5 * torch.ones_like(x), x)
            d_min = torch.min(x_tmp, dim=-1).values

            valid = (d_min > 0.0) & (d_min >= depth0) & (d_min < (depth0 + D * interval))
            d_bin = torch.floor((d_min - depth0) / interval).to(torch.int64)
            d_bin = torch.clamp(d_bin, 0, D - 1)

            onehot = F.one_hot(d_bin, num_classes=D).float()
            onehot = onehot * valid.float().unsqueeze(-1)
            return onehot

        raise ValueError(
            f"gt_depths spatial {(H,W)} not supported. "
            f"Expected input_size {self.input_size} or feature size {(Hf,Wf)}."
        )

    def get_depth_loss(self, gt_depths: torch.Tensor, depth_prob: torch.Tensor) -> torch.Tensor:
        """
        gt_depths: (B,N,H_in,W_in) OR (B,N,Hf,Wf)
        depth_prob: (BN,D,Hf,Wf) (softmax)
        """
        BN, D, Hf, Wf = depth_prob.shape
        B, N = gt_depths.shape[0], gt_depths.shape[1]
        assert BN == B * N, f"depth_prob BN={BN} != B*N={B*N}"

        gt_depths = gt_depths.to(device=depth_prob.device, dtype=torch.float32)
        depth_labels = self.get_downsampled_gt_depth(gt_depths, Hf, Wf)  # (BN*Hf*Wf, D)

        depth_preds = depth_prob.permute(0, 2, 3, 1).contiguous().view(-1, D)

        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]

        if depth_preds.numel() == 0:
            return depth_prob.new_tensor(0.0)

        device_type = "cuda" if depth_prob.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            loss = F.binary_cross_entropy(
                depth_preds.float(),
                depth_labels.float(),
                reduction="none",
            ).sum() / max(1.0, float(fg_mask.sum().item()))

        return loss * float(self.loss_depth_weight)

    # ===== NEW: geometry sanity check once =====
    @torch.no_grad()
    def _geom_sanity_check_once(
        self,
        lidar2img: torch.Tensor,   # (BN,4,4) float32
        aug4: torch.Tensor,        # (BN,4,4) float32
        K: torch.Tensor,           # (BN,3,3) float32
        R: torch.Tensor,           # (BN,3,3) float32
        t: torch.Tensor,           # (BN,3)   float32
    ):
        if (not self.geom_check_once) or self._dbg_once_geom:
            return
        self._dbg_once_geom = True

        device = lidar2img.device
        BN = lidar2img.shape[0]

        # recon: K@[R|t]
        K4 = torch.eye(4, device=device, dtype=torch.float32).view(1, 4, 4).repeat(BN, 1, 1)
        K4[:, :3, :3] = K

        Rt = torch.eye(4, device=device, dtype=torch.float32).view(1, 4, 4).repeat(BN, 1, 1)
        Rt[:, :3, :3] = R
        Rt[:, :3, 3] = t

        P_rec = K4 @ Rt
        err_krt = (P_rec - lidar2img).abs().max().item()

        # check if aug seems baked in projection (double apply risk)
        P_aug = aug4 @ lidar2img
        err_aug_baked = (P_aug - lidar2img).abs().max().item()

        # check aug magnitude (is it identity?)
        eye4 = torch.eye(4, device=device, dtype=torch.float32).view(1, 4, 4)
        aug_mag = (aug4 - eye4).abs().max().item()

        print(f"[VT][GeomCheck] max|K@[R|t] - lidar2img| = {err_krt:.6f}")
        print(f"[VT][GeomCheck] max|(aug@lidar2img) - lidar2img| = {err_aug_baked:.6f} (aug_mag={aug_mag:.6f})")

        if err_krt > self.geom_tol:
            print(f"[VT][GeomCheck] ❌ K/R/t mismatch (tol={self.geom_tol}) -> check dataset K/R/t or projection_mat.")
        else:
            print(f"[VT][GeomCheck] ✅ K/R/t consistent with lidar2img (tol={self.geom_tol}).")

        # 若 aug 本身不是 identity，但 (aug@P) 幾乎等於 P，代表 P 已經含 aug（或 aug 幾乎沒動）
        if self.warn_aug_baked and (aug_mag > 1e-4) and (err_aug_baked < 1e-3):
            print("[VT][GeomCheck] ⚠️ Potential AUG baked into projection_mat -> beware double apply (aug used again in VT).")

    def _splat_bev_flat(self, context_bn, w_bn, idx_bn, bev_hw: int) -> torch.Tensor:
        B, N, C, HW = context_bn.shape
        D = w_bn.shape[2]
        device = context_bn.device
        dtype = context_bn.dtype

        bev = torch.zeros((B, C, bev_hw), device=device, dtype=dtype)
        cnt = torch.zeros((B, 1, bev_hw), device=device, dtype=dtype) if self.normalize else None

        L = N * D * HW
        for b in range(B):
            idx_flat = idx_bn[b].reshape(-1)
            w_flat = w_bn[b].reshape(-1)

            feat = context_bn[b].unsqueeze(2)          # (N,C,1,HW)
            w_ = w_bn[b].unsqueeze(1)                  # (N,1,D,HW)
            src = (feat * w_).permute(1, 0, 2, 3).reshape(C, L)  # (C,L)

            bev[b].scatter_add_(dim=1, index=idx_flat.view(1, -1).expand(C, -1), src=src)

            if self.normalize:
                cnt[b].scatter_add_(dim=1, index=idx_flat.view(1, -1), src=w_flat.view(1, -1))

        if self.normalize:
            bev = bev / (cnt + 1e-6)
        return bev

    def _splat_bev_chunked(self, context_bn, w_bn, idx_bn, bev_hw: int) -> torch.Tensor:
        B, N, C, HW = context_bn.shape
        D = w_bn.shape[2]
        device = context_bn.device
        dtype = context_bn.dtype

        bev = torch.zeros((B, C, bev_hw), device=device, dtype=dtype)
        cnt = torch.zeros((B, 1, bev_hw), device=device, dtype=dtype) if self.normalize else None

        chunk = max(256, int(self.hw_chunk))

        for b in range(B):
            ctx_b = context_bn[b]
            w_b = w_bn[b]
            idx_b = idx_bn[b]

            for s in range(0, HW, chunk):
                e = min(HW, s + chunk)
                chw = e - s

                ctx = ctx_b[:, :, s:e]     # (N,C,chw)
                ww  = w_b[:, :, s:e]       # (N,D,chw)
                ii  = idx_b[:, :, s:e]     # (N,D,chw)

                idx_flat = ii.reshape(-1)

                src = (ctx.unsqueeze(1) * ww.unsqueeze(2))              # (N,D,C,chw)
                src = src.permute(2, 0, 1, 3).reshape(C, N * D * chw)   # (C,L)

                bev[b].scatter_add_(dim=1, index=idx_flat.view(1, -1).expand(C, -1), src=src)

                if self.normalize:
                    w_flat = ww.reshape(-1)
                    cnt[b].scatter_add_(dim=1, index=idx_flat.view(1, -1), src=w_flat.view(1, -1))

        if self.normalize:
            bev = bev / (cnt + 1e-6)
        return bev

    def forward(
        self,
        feats: torch.Tensor,
        metas: Optional[Union[List[Dict], Dict]] = None,
        gt_depths: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ):
        """
        feats: [B, N, C_in, Hf, Wf]
        metas: dict (B,N,...) or list
        gt_depths: (B,N,H_in,W_in) or (B,N,Hf,Wf)
        """
        assert metas is not None, "need metas with lidar2img/img_aug_matrix"
        assert feats.dim() == 5, f"feats should be [B,N,C,Hf,Wf], got {tuple(feats.shape)}"

        B, N, C, Hf, Wf = feats.shape
        device = feats.device
        dtype = feats.dtype
        HW = Hf * Wf
        D = self.num_depth_bins
        BN = B * N

        meta = self._metas_to_batched_tensors(metas, device=device, dtype=dtype)

        # force float32 for geometry
        lidar2img = meta["lidar2img"].reshape(BN, 4, 4).float()
        aug_raw = meta["img_aug_matrix"].reshape(BN, meta["img_aug_matrix"].shape[-2], meta["img_aug_matrix"].shape[-1])
        aug4, post_rot, post_tran = self._parse_img_aug(aug_raw, device=device)  # all float32

        # -------- K,R,t --------
        if "K" in meta and "R" in meta and "t" in meta:
            K_raw = meta["K"]
            R_raw = meta["R"]
            t_raw = meta["t"]

            K_geom = K_raw.reshape(BN, 3, 3).float().to(device)
            R_geom = R_raw.reshape(BN, 3, 3).float().to(device)
            t_geom = t_raw.reshape(BN, 3).float().to(device)
        else:
            K_geom, R_geom, t_geom = _decompose_lidar2img_batch(lidar2img)
            K_geom = K_geom.to(device=device)
            R_geom = R_geom.to(device=device)
            t_geom = t_geom.to(device=device)
        use_krt = ("K" in meta) and ("R" in meta) and ("t" in meta)
        if not hasattr(self, "_dbg_once_use_krt"):
            self._dbg_once_use_krt = True
            print(f"[VT] use batch K/R/t = {use_krt}")

        if use_krt:
            K_raw = meta["K"]; R_raw = meta["R"]; t_raw = meta["t"]
            K_geom = K_raw.reshape(BN, 3, 3).float().to(device)
            R_geom = R_raw.reshape(BN, 3, 3).float().to(device)
            t_geom = t_raw.reshape(BN, 3).float().to(device)
        else:
            K_geom, R_geom, t_geom = _decompose_lidar2img_batch(lidar2img.float())


        # ===== NEW: bda =====
        bda_bn = self._get_bda_bn(meta, B, N, device=device)  # (BN,3,3) float32

        # ===== NEW: geom sanity check once =====
        self._geom_sanity_check_once(lidar2img, aug4, K_geom, R_geom, t_geom)

        # -----------------------
        # Build official-style mlp_input (BN,27)
        # -----------------------
        fx = K_geom[:, 0, 0]
        fy = K_geom[:, 1, 1]
        cx = K_geom[:, 0, 2]
        cy = K_geom[:, 1, 2]

        # post_rot/post_tran from img_aug
        # NOTE: your aug is affine stored as 4x4: u' = A u + t
        # we use post_rot = A (3x3), post_tran = t (3,)
        post_rot_f32 = post_rot
        post_tran_f32 = post_tran

        # cam2(lidar-as-ego): invert lidar->cam
        R_inv = torch.linalg.inv(R_geom)  # (BN,3,3) float32
        t3 = t_geom.view(BN, 3, 1)        # (BN,3,1)

        cam2ego_R = R_inv
        cam2ego_t = (-torch.bmm(R_inv, t3)).squeeze(-1)  # (BN,3)
        cam2ego_3x4 = torch.cat([cam2ego_R, cam2ego_t.unsqueeze(-1)], dim=-1)  # (BN,3,4)

        mlp_15 = torch.stack(
            [
                fx, fy, cx, cy,
                post_rot_f32[:, 0, 0], post_rot_f32[:, 0, 1], post_tran_f32[:, 0],
                post_rot_f32[:, 1, 0], post_rot_f32[:, 1, 1], post_tran_f32[:, 1],
                bda_bn[:, 0, 0], bda_bn[:, 0, 1],
                bda_bn[:, 1, 0], bda_bn[:, 1, 1],
                bda_bn[:, 2, 2],
            ],
            dim=-1,
        )  # (BN,15)
        mlp_input = torch.cat([mlp_15, cam2ego_3x4.reshape(BN, -1)], dim=-1).float()  # (BN,27)

        # -----------------------
        # DepthNet
        # -----------------------
        x = feats.reshape(BN, C, Hf, Wf)
        y = self.depthnet(x, mlp_input)  # (BN, D + Cout, Hf, Wf)
        depth_logits = y[:, :D]
        context = y[:, D:D + self.out_channels]

        depth_prob = depth_logits.float().softmax(dim=1)  # float32
        context = context.to(dtype)

        depth_loss = None
        if return_loss:
            if gt_depths is None:
                if "gt_depths" in meta:
                    gt_depths = meta["gt_depths"]
                elif "gt_depth" in meta:
                    gt_depths = meta["gt_depth"]

            if gt_depths is not None:
                depth_loss = self.get_depth_loss(gt_depths, depth_prob)
            else:
                depth_loss = depth_prob.new_tensor(0.0)

        # -----------------------
        # Pixel grid in augmented input coords
        # -----------------------
        uv1_aug = self._get_uv1_aug(Hf, Wf, device=device, dtype=dtype).float()  # (HW,3)

        # un-augment: uv_orig = (uv_aug - t) @ inv(A)^T
        H_aug = post_rot_f32  # (BN,3,3) float32
        t_aug = post_tran_f32 # (BN,3) float32
        H_inv = torch.linalg.inv(H_aug)  # float32

        uv = uv1_aug.unsqueeze(0) - t_aug.unsqueeze(1)      # (BN,HW,3)
        uv1_orig = torch.bmm(uv, H_inv.transpose(1, 2))     # (BN,HW,3)

        u = uv1_orig[:, :, 0]
        v = uv1_orig[:, :, 1]

        x_over_z = (u - cx.unsqueeze(1)) / (fx.unsqueeze(1) + 1e-6)
        y_over_z = (v - cy.unsqueeze(1)) / (fy.unsqueeze(1) + 1e-6)

        # -----------------------
        # Lift to 3D (cam)
        # -----------------------
        z = self.depth_values.to(device=device).float()              # (D,)
        xcam = x_over_z.float().unsqueeze(1) * z.view(1, D, 1)       # (BN,D,HW)
        ycam = y_over_z.float().unsqueeze(1) * z.view(1, D, 1)
        zcam = z.view(1, D, 1).expand(BN, D, HW)

        X_cam = torch.stack([xcam, ycam, zcam], dim=1).reshape(BN, 3, D * HW)  # (BN,3,DHW)

        # cam -> lidar-as-ego
        X_ego = torch.bmm(R_inv, X_cam - t3)  # (BN,3,DHW) float32

        # apply BDA ONCE here
        if self.apply_bda:
            X_ego = torch.bmm(bda_bn, X_ego)

        X_ego = X_ego.reshape(BN, 3, D, HW)
        x_l = X_ego[:, 0]
        y_l = X_ego[:, 1]

        ix = torch.floor((x_l - float(self.x_min)) / float(self.dx)).to(torch.int64)
        iy = torch.floor((y_l - float(self.y_min)) / float(self.dy)).to(torch.int64)

        valid = (ix >= 0) & (ix < self.bev_w) & (iy >= 0) & (iy < self.bev_h)
        bev_index = (iy * self.bev_w + ix).to(torch.int64)
        bev_index = torch.where(valid, bev_index, torch.zeros_like(bev_index))

        w = depth_prob.to(dtype).reshape(BN, D, HW).masked_fill(~valid, 0)

        bev_hw = self.bev_h * self.bev_w
        context = context.reshape(BN, self.out_channels, HW)
        context_bn = context.reshape(B, N, self.out_channels, HW)
        w_bn = w.reshape(B, N, D, HW)
        idx_bn = bev_index.reshape(B, N, D, HW)

        if self.hw_chunk is not None and self.hw_chunk > 0:
            bev = self._splat_bev_chunked(context_bn, w_bn, idx_bn, bev_hw)
        else:
            bev = self._splat_bev_flat(context_bn, w_bn, idx_bn, bev_hw)

        bev = bev.view(B, self.out_channels, self.bev_h, self.bev_w)
        bev = self.bev_refine(bev)

        if return_loss:
            return bev, depth_loss
        return bev
