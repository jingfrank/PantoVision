#!/bin/bash
# ==============================================================================
# Qwen 大模型受电弓报警图片目录实时监听守护启动脚本
# ==============================================================================

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WATCH_DIR="/hirain/hi_opt/nvme1/Algo-result"
MODEL_PATH="./models/Qwen3-VL-4B-Instruct"

# 若未挂载 nvme1，则自动尝试本地 cache 路径
if [ ! -d "$WATCH_DIR" ]; then
    if [ -d "$BASE_DIR/local_cache/Algo-result" ]; then
        WATCH_DIR="$BASE_DIR/local_cache/Algo-result"
    elif [ -d "$BASE_DIR/local_cache/benchmark_results_v2" ]; then
        WATCH_DIR="$BASE_DIR/local_cache/benchmark_results_v2"
    fi
fi

# 允许外部传入自定义监听目录
if [ -n "$1" ]; then
    WATCH_DIR="$1"
fi

echo "========================================================================"
echo " 🚀 启动 Qwen 大模型报警图片实时监听复核守护服务"
echo " 📁 监控目录: $WATCH_DIR"
echo " 🧠 模型权重: $MODEL_PATH"
echo "========================================================================"

python3 "$BASE_DIR/qwen_alarm_verifier.py" \
    --watch-dir "$WATCH_DIR" \
    --model-path "$MODEL_PATH" \
    --device "cuda:0" \
    --interval 2.0
