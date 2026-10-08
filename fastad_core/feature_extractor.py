import torch.nn as nn
import torch
import os
import logging
import timm
from fastad_common import get_pdn_small, get_pdn_medium
from cdfnet import CDFNet

logger = logging.getLogger(__name__)


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone_name="resnet50",
        layer_to_extract=3,
        device=None,
        pretrained_weights_path=None,
        use_fp16=False,
        engine_path=None,
        prefer_tensorrt=True,
    ):
        super(FeatureExtractor, self).__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16
        self.backbone_name = backbone_name
        self.layer_to_extract = layer_to_extract
        self.engine_path = engine_path
        self.prefer_tensorrt = prefer_tensorrt
        self.trt_runner = None

        # 1. 尝试优先加载 TensorRT 引擎
        if self.prefer_tensorrt and torch.cuda.is_available() and "cuda" in str(self.device):
            trt_candidates = []
            if self.engine_path:
                trt_candidates.append(self.engine_path)
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            trt_candidates.extend([
                os.path.join(repo_root, "models", "fastad_layer2_256.engine"),
                os.path.join(repo_root, "models", "fastad_384.engine"),
            ])
            for cand in trt_candidates:
                if cand and os.path.exists(cand):
                    try:
                        from fastad_core.trt_backbone import TrtFastADBackboneRunner
                        self.trt_runner = TrtFastADBackboneRunner(cand, device=str(self.device))
                        print(f"[FeatureExtractor] 成功加载 FastAD TensorRT 骨干引擎: {cand}")
                        break
                    except Exception as e:
                        logger.warning(f"[FeatureExtractor] TensorRT 骨干加载失败 ({cand}): {e}")
                        self.trt_runner = None

        # 2. 加载本地权重作为 PyTorch 骨干 (同时作为 Zero-Downtime Fallback 兜底)
        if not pretrained_weights_path:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for candidate in [
                os.path.join(repo_root, "models", "model.safetensors"),
                os.path.join(repo_root, "models", "wide_resnet50_2.safetensors"),
                os.path.join(repo_root, "models", "wide_resnet50_2.pth"),
                os.path.join(repo_root, "models", "wide_resnet50_2.pt"),
            ]:
                if os.path.exists(candidate):
                    pretrained_weights_path = candidate
                    print(f"[FeatureExtractor] 自动命中本地骨干网络权重: {candidate}")
                    break

        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            if self.backbone_name == "pdn_small":
                self.backbone = get_pdn_small(out_channels=384, padding=False)
                state_dict = torch.load(pretrained_weights_path, map_location=self.device)
                self.backbone.load_state_dict(state_dict, strict=False)
            elif self.backbone_name == "pdn_medium":
                self.backbone = get_pdn_medium(out_channels=384, padding=False)
                state_dict = torch.load(pretrained_weights_path, map_location=self.device)
                self.backbone.load_state_dict(state_dict, strict=False)
            elif self.backbone_name == "cdfnet":
                self.backbone = CDFNet()
                state_dict = torch.load(pretrained_weights_path, map_location=self.device)
                self.backbone.load_state_dict(state_dict, strict=False)
            else:
                effective_name = self.backbone_name
                if effective_name == "wide_resnet50_2":
                    effective_name = "wide_resnet50_2.racm_in1k"
                try:
                    self.backbone = timm.create_model(
                        effective_name,
                        pretrained=False,
                        features_only=True,
                    )
                except Exception:
                    self.backbone = timm.create_model(
                        self.backbone_name,
                        pretrained=False,
                        features_only=True,
                    )

                # 使用 strict=False 加载权重，过滤掉多余的分类头 (fc.weight, fc.bias)
                try:
                    from timm.models import load_checkpoint
                    load_checkpoint(self.backbone, pretrained_weights_path, strict=False)
                except Exception:
                    if pretrained_weights_path.endswith(".safetensors"):
                        from safetensors.torch import load_file
                        state_dict = load_file(pretrained_weights_path, device="cpu")
                    else:
                        state_dict = torch.load(pretrained_weights_path, map_location="cpu")
                    self.backbone.load_state_dict(state_dict, strict=False)
            self.backbone.eval()
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
            self.backbone = self.backbone.half()
        self.to(self.device)

    def forward(self, x):
        # 1. 优先调用 TensorRT 骨干引擎进行推理 (Zero-Copy)
        if self.trt_runner is not None and self.trt_runner.is_available():
            try:
                out = self.trt_runner.forward(x)
                if isinstance(self.layer_to_extract, list) and not isinstance(out, list):
                    return [out]
                return out
            except Exception as e:
                logger.warning(f"[FeatureExtractor] TensorRT 前向推理遇到异常，回退 PyTorch 骨干: {e}")

        # 2. PyTorch 原生推理
        if self.use_fp16 and x.dtype != torch.float16:
            x = x.half()
        with torch.no_grad():
            features = self.backbone(x)
        if self.backbone_name in ["pdn_small", "pdn_medium", "cdfnet"]:
            return features
        if isinstance(self.layer_to_extract, list):
            return [features[i] for i in self.layer_to_extract]
        return features[self.layer_to_extract]

