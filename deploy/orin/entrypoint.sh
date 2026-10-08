#!/bin/bash
# ==============================================================================
# 受电弓网故障监测系统 - Jetson AGX Orin 容器自启动入口 (entrypoint.sh)
# 适用版本: gongwang_clean (YOLOv11 + FastAD + HRNet 插件化架构)
# ==============================================================================

set -e

echo "======================================================================"
echo "🚀 [Jetson AGX Orin] 弓网故障监测系统正在启动 (gongwang_clean)..."
echo "======================================================================"

cd /app

# 确保模型目录存在
if [ ! -f "models/best.pt" ]; then
    echo "⚠️ 警告: models/best.pt 模型权重未找到，请检查挂载目录！"
fi

# 自动检测与构建适配 Jetson AGX Orin 本地架构 (SM 8.7) 的 TensorRT FP16 引擎
if [ -f "tools/export_tensorrt.py" ]; then
    NEED_BUILD=0
    if [ ! -f "models/best.engine" ] && [ -f "models/best.pt" ]; then
        NEED_BUILD=1
    fi
    if [ ! -f "models/best_hrnet_w32_panto.engine" ] && [ -f "models/best_hrnet_w32_panto.pth" ]; then
        NEED_BUILD=1
    fi
    if [ ! -f "models/fastad_layer2_256.engine" ] && [ -f "models/fastad_layer2_256_reg1e4_nq3_fmq1.pt" ]; then
        NEED_BUILD=1
    fi

    if [ "$NEED_BUILD" -eq 1 ]; then
        echo "⚙️ [Jetson TRT] 检测到缺少当前硬件架构对应的 TensorRT 引擎，正在执行首次自动编译 (FP16)..."
        python3 tools/export_tensorrt.py --model all --precision fp16 || echo "⚠️ [Jetson TRT] 引擎编译遇到异常，将使用 PyTorch 自动保底运行"
    else
        echo "✅ [Jetson TRT] 已检测到本地 TensorRT 加速引擎就绪"
    fi
fi

# 启动弓网监测系统
echo ">> 正在拉起弓网监测主进程 (main.py)..."
exec python3 main.py --config ./config_prod.yaml
