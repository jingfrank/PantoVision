import torch.nn as nn
import torch
import os
import timm
from fastad_common import get_pdn_small,get_pdn_medium
from cdfnet import CDFNet
class FeatureExtractor(nn.Module):
    def __init__(self,backbone_name ="resnet50",layer_to_extract=3,device = None,pretrained_weights_path=None,use_fp16 = False):
        super(FeatureExtractor, self).__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #加载timm预训练模型
        # 加载本地权重（如果提供了路径）
        self.use_fp16 = use_fp16
        self.backbone_name = backbone_name
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            if self.backbone_name == "pdn_small":
                self.backbone = get_pdn_small(out_channels=384, padding=False)
            elif self.backbone_name == "pdn_medium":
                self.backbone = get_pdn_medium(out_channels=384, padding=False)
            elif self.backbone_name == "cdfnet":
                self.backbone = CDFNet()
            else:
                # 明确映射到 racm_in1k 版本，保证一致性，若老版本 timm 不支持则降级
                effective_name = self.backbone_name
                try:
                    if effective_name == "wide_resnet50_2":
                        effective_name = "wide_resnet50_2.racm_in1k"
                    self.backbone = timm.create_model(
                        effective_name,
                        pretrained=False,  # 禁用自动下载
                        features_only=True,
                    )
                except Exception:
                    self.backbone = timm.create_model(
                        self.backbone_name,
                        pretrained=False,
                        features_only=True,
                    )
            state_dict = torch.load(pretrained_weights_path, map_location=self.device)
            self.backbone.load_state_dict(state_dict, strict=False)
            self.backbone.eval()
            # 转换为 FP16
        
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            effective_name = self.backbone_name
            self.backbone = timm.create_model(
                effective_name,
                pretrained=True,  # 尝试在线下载
                features_only=True,
            )
            # 与本地加载保持一致：设为评估模式并冻结参数
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        if use_fp16:
            self.backbone= self.backbone.half()
        self.layer_to_extract = layer_to_extract
        # 设置设备
        self.to(self.device)

    def forward(self,x):
        # 自动转换输入
        if self.use_fp16 and x.dtype != torch.float16:
            x = x.half()
        with torch.no_grad():
            # if self.device == 'cuda':
            #     with torch.cuda.amp.autocast():
            #         features = self.backbone(x)
            # else:
            #     features = self.backbone(x)
            features = self.backbone(x)
        if self.backbone_name in ["pdn_small","pdn_medium","cdfnet"]:
            return features
        if isinstance(self.layer_to_extract, list):
            return [features[i] for i in self.layer_to_extract]
        return features[self.layer_to_extract]
