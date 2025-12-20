# models/neck.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN_Torch(nn.Module):
    """
    純 PyTorch 實現的 FPN (Feature Pyramid Network)
    支援輸入 [B, N, C, H, W] 或 [M, C, H, W]
    """
    def __init__(
        self,
        in_channels_list=[256, 512, 1024, 2048], # 對應 ResNet Layer 1-4
        out_channels=256,
        top_down_levels=[3, 2, 1, 0], # 處理順序：Layer4 -> Layer1 (C5->C2)
    ):
        super().__init__()
        self.out_channels = out_channels
        self.top_down_levels = top_down_levels
        
        # 建立 Lateral Convolution (1x1)
        self.lateral_convs = nn.ModuleList()
        # 建立 Output Convolution (3x3)
        self.fpn_convs = nn.ModuleList()

        for in_c in in_channels_list:
            # 1x1 conv: 降低通道數
            l_conv = nn.Conv2d(in_c, out_channels, kernel_size=1)
            # 3x3 conv: 消除上採樣帶來的混疊效應 (Aliasing)
            fpn_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            
            # 初始化權重 (Xavier/Kaiming)
            nn.init.kaiming_uniform_(l_conv.weight, a=1)
            nn.init.constant_(l_conv.bias, 0)
            nn.init.kaiming_uniform_(fpn_conv.weight, a=1)
            nn.init.constant_(fpn_conv.bias, 0)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs):
        """
        inputs: List of Tensors.
                可以是 [[B,N,C1,H1,W1], ...] 或 [[M,C1,H1,W1], ...]
        Return: List of Tensors [[B,N,C_out,H1,W1], ...]
        """
        # 1. 處理維度: 如果是 5D (B,N,C,H,W)，先壓扁成 4D (B*N,C,H,W)
        is_5d = False
        inputs_4d = []
        B, N = 0, 0
        
        for x in inputs:
            if x.dim() == 5:
                is_5d = True
                B, N, C, H, W = x.shape
                inputs_4d.append(x.view(B * N, C, H, W))
            else:
                inputs_4d.append(x)
        
        # 2. 建立 Laterals (全部先過 1x1 conv)
        # inputs_4d[i] 對應 in_channels_list[i]
        laterals = [
            conv(inputs_4d[i]) 
            for i, conv in enumerate(self.lateral_convs)
        ]

        # 3. Top-Down Pathway (自頂向下融合)
        # 通常是從最後一層 (C5) 往前走到第一層 (C2)
        # range(start, stop, step) -> range(3, 0, -1) 意味著 3, 2, 1
        num_levels = len(laterals)
        for i in range(num_levels - 1, 0, -1):
            # 取得上一層 (較小尺寸, i) 的特徵，上採樣後，加到下一層 (較大尺寸, i-1)
            # scale_factor=2 假設每一層解析度差 2 倍 (ResNet 標準)
            top_down_feat = F.interpolate(
                laterals[i], scale_factor=2.0, mode="nearest"
            )
            laterals[i - 1] = laterals[i - 1] + top_down_feat

        # 4. 輸出 (全部過 3x3 conv)
        outs = [
            self.fpn_convs[i](laterals[i]) 
            for i in range(num_levels)
        ]

        # 5. 如果原本是 5D，還原回去
        if is_5d:
            outs = [
                o.view(B, N, self.out_channels, o.shape[2], o.shape[3]) 
                for o in outs
            ]

        return tuple(outs) # 回傳 (P2, P3, P4, P5)