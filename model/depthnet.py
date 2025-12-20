# model/depthnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ====================================================================
# 1. 基礎組件 (BasicBlock, MLP, SELayer)
#    我們手動定義 BasicBlock，就不依賴 mmdet 了
# ====================================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        # x_se: (BN, C, 1, 1)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

# ====================================================================
# 2. DCN (Deformable Conv) 安全處理
# ====================================================================
class DCNWrapper(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, groups=1):
        super().__init__()
        # 嘗試使用 mmcv 的 DCN，如果沒有則退化為普通 Conv
        self.has_dcn = False
        try:
            from mmcv.ops import DeformConv2dPack
            self.conv = DeformConv2dPack(in_channels, out_channels, kernel_size=kernel_size, 
                                         padding=padding, groups=groups)
            self.has_dcn = True
        except ImportError:
            # Fallback
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                                  padding=padding, groups=groups, bias=False)
    
    def forward(self, x):
        return self.conv(x)

# ====================================================================
# 3. ASPP 模組
# ====================================================================

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]
        self.aspp1 = _ASPPModule(inplanes, mid_channels, 1, 0, dilations[0], BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, mid_channels, 3, dilations[1], dilations[1], BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, mid_channels, 3, dilations[2], dilations[2], BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, mid_channels, 3, dilations[3], dilations[3], BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5), inplanes, 1, bias=False)
        self.bn1 = BatchNorm(inplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

# ====================================================================
# 4. 完整的 DepthNet
# ====================================================================

class DepthNet(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 use_dcn=True,
                 use_aspp=False,
                 with_cp=False,
                 aspp_mid_channels=-1,
                 stereo=False, # 為了兼容接口
                 bias=0.0):    # 為了兼容接口
        super(DepthNet, self).__init__()
        
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.context_channels = context_channels
        self.depth_channels = depth_channels
        self.with_cp = with_cp
        
        # 1. Reduce Conv
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        
        # 2. Context Branch
        self.context_conv = nn.Conv2d(mid_channels, context_channels, kernel_size=1, stride=1, padding=0)

        # 3. Camera Awareness (MLP + SE)
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)
        
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)

        # 4. Depth Branch (ResNet Blocks)
        # 我們手動構建 Residual Blocks
        self.depth_conv_layers = nn.ModuleList([
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels)
        ])

        # ASPP
        self.use_aspp = use_aspp
        if use_aspp:
            if aspp_mid_channels < 0:
                aspp_mid_channels = mid_channels
            self.aspp = ASPP(mid_channels, aspp_mid_channels)

        # DCN
        self.use_dcn = use_dcn
        if use_dcn:
            self.dcn = DCNWrapper(mid_channels, mid_channels)

        # Output Head
        self.depth_out = nn.Conv2d(mid_channels, depth_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, mlp_input, stereo_metas=None):
        """
        x: (BN, C, H, W)
        mlp_input: (BN, 27)
        stereo_metas: 兼容接口用，這裡暫不使用
        """
        # Batch Norm for MLP input (BN, 27)
        mlp_input = self.bn(mlp_input)
        
        # Base Feature
        x = self.reduce_conv(x) # (BN, mid, H, W)

        # [Context Branch] Camera Aware Reweighting
        context_se = self.context_mlp(mlp_input)[..., None, None] # (BN, mid, 1, 1)
        context = self.context_se(x, context_se)
        context = self.context_conv(context) # (BN, context_C, H, W)
        
        # [Depth Branch] Camera Aware Reweighting
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se) # (BN, mid, H, W)
        
        # Internal function for Checkpoint
        def _inner_forward(feat):
            for block in self.depth_conv_layers:
                feat = block(feat)
            
            if self.use_aspp:
                feat = self.aspp(feat)
            
            if self.use_dcn:
                feat = self.dcn(feat)
            return feat

        # Gradient Checkpointing (Save memory)
        if self.with_cp and x.requires_grad:
            depth = checkpoint(_inner_forward, depth)
        else:
            depth = _inner_forward(depth)
            
        depth = self.depth_out(depth) # (BN, D, H, W)
        
        # Concat Depth + Context
        return torch.cat([depth, context], dim=1)