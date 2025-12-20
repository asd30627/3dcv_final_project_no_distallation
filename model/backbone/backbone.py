# models/backbone.py
import torch
import torch.nn as nn
from torchvision.models import resnet50

class MultiViewResNetBackbone(nn.Module):
    """
    純淨的多視角 ResNet-50 Backbone
    輸入: [B, N, 3, H, W]
    輸出: Tuple([B, N, C1, H1, W1], [B, N, C2, H2, W2], ...)
    """

    def __init__(
        self,
        pretrained: bool = True,
        frozen_stages: int = 1,
        return_layers: list = [1, 2, 3, 4], # 指定要回傳哪些層 (1=layer1/C2, 4=layer4/C5)
    ):
        super().__init__()
        # 使用 torchvision 的標準 resnet50
        backbone = resnet50(pretrained=pretrained)

        # 拆解 ResNet 的各個階段
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1 # C2: 256, stride 4
        self.layer2 = backbone.layer2 # C3: 512, stride 8
        self.layer3 = backbone.layer3 # C4: 1024, stride 16
        self.layer4 = backbone.layer4 # C5: 2048, stride 32

        self.return_layers = return_layers
        self._freeze_stages(frozen_stages)

    def _freeze_stages(self, frozen_stages: int):
        if frozen_stages >= 0:
            # 永遠凍結 BN 參數 (通常在 BS 較小時這是必須的)
            self.eval()
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval() 
                    for param in m.parameters():
                        param.requires_grad = False

        if frozen_stages >= 1:
            for param in self.stem.parameters():
                param.requires_grad = False
            for param in self.layer1.parameters():
                param.requires_grad = False

        if frozen_stages >= 2:
            for param in self.layer2.parameters():
                param.requires_grad = False

        if frozen_stages >= 3:
            for param in self.layer3.parameters():
                param.requires_grad = False
        
        # layer4 通常不凍結

    def forward(self, imgs: torch.Tensor):
        """
        imgs: [B, N, 3, H, W]
        """
        B, N, C, H, W = imgs.shape
        # 合併 B 和 N 以進行批次處理: [B*N, 3, H, W]
        x = imgs.view(B * N, C, H, W)

        # 逐層提取特徵
        x = self.stem(x)
        c2 = self.layer1(x) # 256
        c3 = self.layer2(c2) # 512
        c4 = self.layer3(c3) # 1024
        c5 = self.layer4(c4) # 2048

        # 整理所有特徵層
        all_feats = [c2, c3, c4, c5] # 對應 layer 1, 2, 3, 4

        # 篩選需要回傳的層，並 reshape 回 [B, N, C, H, W]
        outs = []
        for i in range(4):
            if (i + 1) in self.return_layers:
                f = all_feats[i]
                _, C_feat, H_feat, W_feat = f.shape
                outs.append(f.view(B, N, C_feat, H_feat, W_feat))
        
        return tuple(outs)