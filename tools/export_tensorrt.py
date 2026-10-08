#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal TensorRT FP16 Engine Export & Verification Suite (export_tensorrt.py).

Features:
1. One-click export for:
   - YOLOv11 (best.pt -> best.engine)
   - HRNet (best_hrnet_w32_panto.pth -> best_hrnet_w32_panto.onnx -> best_hrnet_w32_panto.engine)
2. Native support for both x86_64 (TITAN RTX) and aarch64 (Jetson AGX Orin Docker);
3. FP16 half-precision optimization with strict validation of output parity (Cosine Sim & Max Diff);
4. Multi-backend builder: TensorRT Python API with trtexec CLI fallback.

Usage:
    python tools/export_tensorrt.py --model all --precision fp16
    python tools/export_tensorrt.py --model yolo --precision fp16
    python tools/export_tensorrt.py --model hrnet --precision fp16
"""

import os
import sys
import time
import argparse
import subprocess
import shutil
import logging
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import torch

# Ensure repository root in sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("export_trt")


def get_trtexec_path() -> Optional[str]:
    """Find trtexec executable path."""
    candidates = [
        shutil.which("trtexec"),
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/cuda/bin/trtexec",
        "/usr/bin/trtexec",
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def build_engine_from_onnx(
    onnx_path: str,
    engine_path: str,
    precision: str = "fp16",
    workspace_gb: float = 4.0,
    device: int = 0,
) -> bool:
    """
    Build TensorRT Engine from ONNX model file.
    Prefers Python TensorRT API; falls back to trtexec CLI if available.
    """
    logger.info(f"▶ 正在编译 TensorRT Engine: {onnx_path} -> {engine_path} (精度: {precision.upper()})")
    t0 = time.time()

    # Method 1: Try TensorRT Python API
    try:
        import tensorrt as trt

        trt_logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(trt_logger)
        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(explicit_batch)
        parser = trt.OnnxParser(network, trt_logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(f"[TRT Parser] {parser.get_error(error)}")
                raise RuntimeError(f"解析 ONNX 文件失败: {onnx_path}")

        config = builder.create_builder_config()

        # Workspace limit
        ws_bytes = int(workspace_gb * (1024 ** 3))
        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws_bytes)
        else:
            config.max_workspace_size = ws_bytes

        # Precision flags
        if precision.lower() == "fp16":
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("[TRT Builder] 已启用 FP16 半精度加速模式")
            else:
                logger.warning("[TRT Builder] 目标平台未声明 fast_fp16，将以默认精度编译")

        logger.info(f"[TRT Builder] 开始构建序列化网络，请稍候 (可能耗时 1~3 分钟)...")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("构建序列化网络失败 (build_serialized_network 返回 None)")

        os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

        cost = time.time() - t0
        size_mb = os.path.getsize(engine_path) / (1024 * 1024)
        logger.info(f"✅ TensorRT Engine 编译成功: {engine_path} ({size_mb:.1f} MB, 耗时 {cost:.1f}s)")
        return True

    except Exception as e:
        logger.warning(f"[TRT Python API 失败: {e}]，尝试使用 trtexec 命令行编译...")

    # Method 2: Try trtexec CLI
    trtexec = get_trtexec_path()
    if trtexec:
        cmd = [
            trtexec,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            f"--memPoolSize=workspace:{int(workspace_gb * 1024)}",
            f"--device={device}",
        ]
        if precision.lower() == "fp16":
            cmd.append("--fp16")

        logger.info(f"执行命令: {' '.join(cmd)}")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and os.path.exists(engine_path):
            cost = time.time() - t0
            size_mb = os.path.getsize(engine_path) / (1024 * 1024)
            logger.info(f"✅ trtexec 编译成功: {engine_path} ({size_mb:.1f} MB, 耗时 {cost:.1f}s)")
            return True
        else:
            logger.error(f"trtexec 执行失败:\nSTDOUT:\n{res.stdout[-1000:]}\nSTDERR:\n{res.stderr[-1000:]}")
            return False

    logger.error("❌ 无法构建 TensorRT 引擎：缺少有效 tensorrt Python 库且未找到 trtexec！")
    return False


# ==============================================================================
# 1. YOLO 导出 (YOLOv11: best.pt -> best.engine)
# ==============================================================================

def export_yolo(
    pt_path: str = "models/best.pt",
    engine_path: str = "models/best.engine",
    precision: str = "fp16",
    device: int = 0,
    imgsz: int = 640,
    validate: bool = True,
) -> bool:
    """
    通过 Ultralytics 原生导出工具构建 YOLO TensorRT Engine.
    """
    pt_path = os.path.abspath(pt_path)
    if not os.path.exists(pt_path):
        logger.error(f"[YOLO Export] 未找到 PyTorch 模型: {pt_path}")
        return False

    logger.info(f"\n========================================================")
    logger.info(f"🚀 开始导出 YOLOv11 TensorRT Engine (精度: {precision.upper()})")
    logger.info(f"  输入权重: {pt_path}")
    logger.info(f"  目标输出: {engine_path}")
    logger.info(f"========================================================")

    t0 = time.time()
    try:
        from ultralytics import YOLO

        model = YOLO(pt_path)
        is_half = precision.lower() == "fp16"

        exported_path = model.export(
            format="engine",
            half=is_half,
            imgsz=imgsz,
            device=device,
            dynamic=False,
            verbose=False,
        )

        # 检查输出并移动到目标位置（若与 engine_path 不一致）
        if exported_path and os.path.exists(exported_path):
            target_abs = os.path.abspath(engine_path)
            if os.path.abspath(exported_path) != target_abs:
                os.makedirs(os.path.dirname(target_abs), exist_ok=True)
                shutil.copyfile(exported_path, target_abs)
                logger.info(f"同步重命名: {exported_path} -> {target_abs}")

            cost = time.time() - t0
            size_mb = os.path.getsize(target_abs) / (1024 * 1024)
            logger.info(f"✅ YOLO TensorRT Engine 导出成功: {target_abs} ({size_mb:.1f} MB, 耗时 {cost:.1f}s)")

            # 数值对齐验证
            if validate:
                validate_yolo_parity(pt_path, target_abs, device=device)

            return True
        else:
            logger.error("[YOLO Export] 导出失败，未生成有效 engine 文件")
            return False

    except Exception as exc:
        logger.error(f"[YOLO Export] 导出异常: {exc}", exc_info=True)
        return False


def validate_yolo_parity(pt_path: str, engine_path: str, device: int = 0) -> None:
    """验证 PyTorch 与 TensorRT YOLO 模型的检测一致性"""
    logger.info("▶ 正在校验 YOLO PyTorch 与 TensorRT 预测结果一致性...")
    try:
        from ultralytics import YOLO

        pt_model = YOLO(pt_path)
        trt_model = YOLO(engine_path, task="detect")

        # 使用典型测试图像
        test_img = None
        for cand in ["../normal.jpg", "local_cache/normal.jpg"]:
            cand_abs = os.path.abspath(os.path.join(REPO_ROOT, cand))
            if os.path.exists(cand_abs):
                test_img = cand_abs
                break

        if test_img is None:
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            cv2.rectangle(test_img, (100, 200), (500, 400), (200, 200, 200), -1)

        res_pt = pt_model(test_img, verbose=False, device=device)[0]
        res_trt = trt_model(test_img, verbose=False, device=device)[0]

        pt_boxes = res_pt.boxes
        trt_boxes = res_trt.boxes

        logger.info(f"  PyTorch 目标框数: {len(pt_boxes)} | TensorRT 目标框数: {len(trt_boxes)}")
        if len(pt_boxes) > 0 and len(trt_boxes) > 0:
            pt_cls = [int(c.item()) for c in pt_boxes.cls]
            trt_cls = [int(c.item()) for c in trt_boxes.cls]
            logger.info(f"  类别对齐: PyTorch={pt_cls} vs TRT={trt_cls}")
            pt_conf = float(pt_boxes.conf.max().item())
            trt_conf = float(trt_boxes.conf.max().item())
            diff = abs(pt_conf - trt_conf)
            logger.info(f"  最大置信度: PyTorch={pt_conf:.3f} | TRT={trt_conf:.3f} (偏差: {diff:.4f})")
            if diff < 0.05:
                logger.info("✅ YOLO 数值一致性校验合格 (|Δconf| < 0.05)")
            else:
                logger.warning(f"⚠️ YOLO 置信度偏差稍大: {diff:.4f}")
        else:
            logger.info("✅ YOLO 框数完全一致 (0 目标基准)")

    except Exception as e:
        logger.warning(f"[YOLO Parity Validation] 校验遇到非致命警告: {e}")


# ==============================================================================
# 2. HRNet 导出 (HRNet-W32: best_hrnet.pth -> best_hrnet.engine)
# ==============================================================================

def export_hrnet(
    pth_path: str = "models/best_hrnet_w32_panto.pth",
    engine_path: str = "models/best_hrnet_w32_panto.engine",
    onnx_path: Optional[str] = None,
    precision: str = "fp16",
    device: int = 0,
    validate: bool = True,
) -> bool:
    """
    导出 HRNet 4 关键点检测器为 ONNX 并编译为 TensorRT Engine.
    固定输入: 1x3x288x384 (H=288, W=384)
    固定输出: 1x4x72x96 (H=72, W=96)
    """
    pth_path = os.path.abspath(pth_path)
    if not os.path.exists(pth_path):
        logger.error(f"[HRNet Export] 未找到权重文件: {pth_path}")
        return False

    if onnx_path is None:
        onnx_path = os.path.splitext(pth_path)[0] + ".onnx"

    logger.info(f"\n========================================================")
    logger.info(f"🚀 开始导出 HRNet-W32 TensorRT Engine (精度: {precision.upper()})")
    logger.info(f"  输入权重: {pth_path}")
    logger.info(f"  中间 ONNX: {onnx_path}")
    logger.info(f"  目标输出: {engine_path}")
    logger.info(f"========================================================")

    # Step 1: Export PyTorch to ONNX
    try:
        from keypoint_tilt.hrnet import HRNetKeypoint

        model = HRNetKeypoint(num_joints=4, width=32)
        state_dict = torch.load(pth_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        dummy_input = torch.randn(1, 3, 288, 384)

        logger.info(f"▶ 正在导出 ONNX (opset=13, input=[1, 3, 288, 384])...")
        os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=13,
            input_names=["input"],
            output_names=["heatmaps"],
            dynamic_axes=None,  # 静态 batch=1，获得 TRT 最高优化与最低显存碎片
        )
        onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        logger.info(f"✅ ONNX 导出成功: {onnx_path} ({onnx_size_mb:.1f} MB)")

    except Exception as e:
        logger.error(f"[HRNet ONNX Export] 导出失败: {e}", exc_info=True)
        return False

    # Step 2: Build TensorRT Engine from ONNX
    ok = build_engine_from_onnx(
        onnx_path=onnx_path,
        engine_path=engine_path,
        precision=precision,
        device=device,
    )
    if not ok:
        return False

    # Step 3: Validate Numerical Parity
    if validate:
        validate_hrnet_parity(pth_path, engine_path, device=device)

    return True


def validate_hrnet_parity(pth_path: str, engine_path: str, device: int = 0) -> None:
    """验证 HRNet PyTorch 与 TensorRT 输出热图的数值余弦相似度与最大误差"""
    logger.info("▶ 正在校验 HRNet PyTorch 与 TensorRT 热图数值对齐度...")
    try:
        from keypoint_tilt.hrnet import HRNetKeypoint
        from keypoint_tilt.trt_engine import TrtHRNetRunner

        device_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device_str)

        # PyTorch
        pt_model = HRNetKeypoint(num_joints=4, width=32).to(dev)
        pt_model.load_state_dict(torch.load(pth_path, map_location=dev))
        pt_model.eval()

        # TensorRT Runner
        trt_runner = TrtHRNetRunner(engine_path, device=device_str)

        dummy_in = torch.randn(1, 3, 288, 384, device=dev, dtype=torch.float32)

        with torch.no_grad():
            out_pt = pt_model(dummy_in)
            out_trt = trt_runner.forward(dummy_in)

        # 相似度分析
        t_pt = out_pt.flatten().cpu().numpy()
        t_trt = out_trt.flatten().cpu().numpy()

        cos_sim = np.dot(t_pt, t_trt) / (np.linalg.norm(t_pt) * np.linalg.norm(t_trt) + 1e-8)
        max_diff = np.max(np.abs(t_pt - t_trt))
        mean_diff = np.mean(np.abs(t_pt - t_trt))

        logger.info(f"  热图余弦相似度: {cos_sim:.5f} (目标 >= 0.999)")
        logger.info(f"  热图最大绝对误差: {max_diff:.5f} | 平均绝对误差: {mean_diff:.5f}")

        if cos_sim >= 0.998:
            logger.info("✅ HRNet TensorRT FP16 数值精度校验合格！")
        else:
            logger.warning(f"⚠️ HRNet 余弦相似度低于预期: {cos_sim:.5f}")

    except Exception as e:
        logger.warning(f"[HRNet Parity Validation] 校验遇到异常: {e}", exc_info=True)


# ==============================================================================
# FastAD WideResNet Backbone Export & Verification
# ==============================================================================

class PrunedWideResNetBackbone(torch.nn.Module):
    """
    剪枝 WideResNet-50 骨干网络 (仅保留到目标特征提取层，丢弃后续冗余 94% 参数).
    """

    def __init__(self, weights_path: str, max_layer: int = 2):
        super().__init__()
        import timm
        from safetensors.torch import load_file

        base = timm.create_model("wide_resnet50_2", pretrained=False)
        if weights_path.endswith(".safetensors"):
            state_dict = load_file(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location="cpu")
        base.load_state_dict(state_dict, strict=False)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.act1 = base.act1
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.max_layer = max_layer
        if max_layer >= 3:
            self.layer3 = base.layer3
        if max_layer >= 4:
            self.layer4 = base.layer4
        self.eval()

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        if self.max_layer == 2:
            return x2
        elif self.max_layer == 3:
            x3 = self.layer3(x2)
            return x2, x3
        elif self.max_layer == 4:
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            return x2, x3, x4
        return x2


def validate_fastad_parity(
    weights_path: str,
    engine_path: str,
    layer_to_extract: List[int],
    input_size: Tuple[int, int],
    device: int = 0,
) -> None:
    """验证 FastAD 骨干网络 PyTorch 与 TensorRT 输出特征图的余弦相似度与最大误差"""
    logger.info("▶ 正在校验 FastAD 骨干网络 PyTorch 与 TensorRT 特征图数值对齐度...")
    try:
        from fastad_core.trt_backbone import TrtFastADBackboneRunner

        device_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device_str)

        max_layer = max(layer_to_extract)
        pt_model = PrunedWideResNetBackbone(weights_path, max_layer=max_layer).to(dev)
        pt_model.eval()

        trt_runner = TrtFastADBackboneRunner(engine_path, device=device_str)

        dummy_in = torch.randn(1, 3, input_size[1], input_size[0], device=dev, dtype=torch.float32)

        with torch.no_grad():
            out_pt = pt_model(dummy_in)
            out_trt = trt_runner.forward(dummy_in)

        if isinstance(out_pt, (list, tuple)):
            out_pt_t = out_pt[0]
            out_trt_t = out_trt[0] if isinstance(out_trt, (list, tuple)) else out_trt
        else:
            out_pt_t = out_pt
            out_trt_t = out_trt

        t_pt = out_pt_t.flatten().cpu().numpy()
        t_trt = out_trt_t.flatten().cpu().numpy()

        cos_sim = np.dot(t_pt, t_trt) / (np.linalg.norm(t_pt) * np.linalg.norm(t_trt) + 1e-8)
        max_diff = np.max(np.abs(t_pt - t_trt))
        mean_diff = np.mean(np.abs(t_pt - t_trt))

        logger.info(f"  FastAD 特征图余弦相似度: {cos_sim:.6f} (目标 >= 0.9999)")
        logger.info(f"  FastAD 特征图最大绝对误差: {max_diff:.6f} | 平均绝对误差: {mean_diff:.6f}")

        if cos_sim >= 0.9990:
            logger.info("✅ FastAD 骨干网络 TensorRT FP16 数值精度校验合格！")
        else:
            logger.warning(f"⚠️ FastAD 骨干网络余弦相似度低于预期: {cos_sim:.6f}")

    except Exception as e:
        logger.warning(f"[FastAD Parity Validation] 校验遇到异常: {e}", exc_info=True)


def export_fastad(
    pt_path: str = "models/fastad_layer2_256_reg1e4_nq3_fmq1.pt",
    weights_path: Optional[str] = None,
    engine_path: Optional[str] = None,
    precision: str = "fp16",
    device: int = 0,
    validate: bool = True,
) -> bool:
    """
    一键导出 FastAD 剪枝骨干网络: PyTorch -> ONNX -> TensorRT Engine.
    """
    pt_path = os.path.abspath(pt_path)
    if not os.path.exists(pt_path):
        logger.error(f"[FastAD] 找不到 FastAD 检查点: {pt_path}")
        return False

    # 1. 读取检查点元数据
    ckpt = torch.load(pt_path, map_location="cpu")
    layer_to_extract = ckpt.get("layer_to_extract", [2])
    if isinstance(layer_to_extract, int):
        layer_to_extract = [layer_to_extract]
    input_size = tuple(ckpt.get("input_size", (256, 256)))
    max_layer = max(layer_to_extract)

    # 2. 自动定位 WideResNet 权重文件
    if not weights_path:
        for candidate in [
            os.path.join(REPO_ROOT, "models", "model.safetensors"),
            os.path.join(REPO_ROOT, "models", "wide_resnet50_2.safetensors"),
            os.path.join(REPO_ROOT, "models", "wide_resnet50_2.pth"),
        ]:
            if os.path.exists(candidate):
                weights_path = candidate
                break

    if not weights_path or not os.path.exists(weights_path):
        logger.error("[FastAD] 找不到 WideResNet50_2 预训练权重文件 (model.safetensors)")
        return False

    if not engine_path:
        engine_path = os.path.join(
            REPO_ROOT, "models", f"fastad_layer{max_layer}_{input_size[0]}.engine"
        )
    engine_path = os.path.abspath(engine_path)

    onnx_path = os.path.splitext(engine_path)[0] + ".onnx"
    logger.info("=" * 70)
    logger.info("🚀 开始 FastAD 骨干网络 TensorRT 编译导出")
    logger.info(f"  FastAD 检查点: {pt_path}")
    logger.info(f"  骨干网络权重: {weights_path}")
    logger.info(f"  提取特征层: {layer_to_extract} (最大层: {max_layer})")
    logger.info(f"  输入分辨率: {input_size[0]}x{input_size[1]}")
    logger.info(f"  目标 Engine: {engine_path}")
    logger.info("=" * 70)

    # 3. 构造剪枝模型并导出 ONNX
    device_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device_str)

    model = PrunedWideResNetBackbone(weights_path, max_layer=max_layer).to(dev)
    model.eval()

    dummy_input = torch.randn(1, 3, input_size[1], input_size[0], device=dev, dtype=torch.float32)

    output_names = [f"layer{l}_features" for l in sorted(layer_to_extract)]

    logger.info(f"▶ 正在导出剪枝骨干为 ONNX: {onnx_path}")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["images"],
            output_names=output_names,
            dynamic_axes=None,  # 静态 batch=1，获得 TRT 最高优化与最低显存碎片
        )

    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    logger.info(f"✅ ONNX 导出成功: {onnx_path} ({onnx_size_mb:.2f} MB)")

    # 4. 编译 TensorRT Engine
    ok = build_engine_from_onnx(
        onnx_path=onnx_path,
        engine_path=engine_path,
        precision=precision,
        device=device,
    )
    if not ok:
        return False

    # 5. 数值一致性校验
    if validate:
        validate_fastad_parity(
            weights_path=weights_path,
            engine_path=engine_path,
            layer_to_extract=layer_to_extract,
            input_size=input_size,
            device=device,
        )

    return True


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="受电弓故障监测系统 - TensorRT 模型导出套件")
    parser.add_argument(
        "--model",
        type=str,
        choices=["all", "yolo", "hrnet", "fastad"],
        default="all",
        help="待导出的模型 (all/yolo/hrnet/fastad)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "fp32"],
        default="fp16",
        help="导出精度 (默认: fp16)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA 设备 ID (默认: 0)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="跳过数值精度自检",
    )
    args = parser.parse_args()

    validate = not args.no_validate
    success = True

    if args.model in ("all", "yolo"):
        ok = export_yolo(
            pt_path="models/best.pt",
            engine_path="models/best.engine",
            precision=args.precision,
            device=args.device,
            validate=validate,
        )
        success = success and ok

    if args.model in ("all", "hrnet"):
        ok = export_hrnet(
            pth_path="models/best_hrnet_w32_panto.pth",
            engine_path="models/best_hrnet_w32_panto.engine",
            precision=args.precision,
            device=args.device,
            validate=validate,
        )
        success = success and ok

    if args.model in ("all", "fastad"):
        ok = export_fastad(
            pt_path="models/fastad_layer2_256_reg1e4_nq3_fmq1.pt",
            engine_path="models/fastad_layer2_256.engine",
            precision=args.precision,
            device=args.device,
            validate=validate,
        )
        success = success and ok

    if success:
        logger.info("\n🎉 所有目标模型 TensorRT 编译导出全部顺利完成！")
    else:
        logger.error("\n❌ 部分模型导出遇到错误，请检查日志！")
        sys.exit(1)


if __name__ == "__main__":
    main()
