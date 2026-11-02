#!/bin/bash
# ==============================================================================
# 动车组弓网监测系统 - 启动脚本（自动时间戳日志归档版）
# 参考: gongwang_jetson_vllm/start.sh
# ==============================================================================

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

# ==============================================================================
# 自动关联持久化模型库（自适应 B/S 主程序随机 UUID 子目录部署，免去重复拷贝 1GB 模型）
# ==============================================================================
NEED_LINK=true

# 1. 如果当前 models 已经存在且包含有效模型权重文件，则无需重新关联
if [ -e "models" ] && { [ -f "models/best.pt" ] || [ -f "models/best.engine" ] || [ -f "models/best.onnx" ]; }; then
    NEED_LINK=false
fi

if [ "$NEED_LINK" = true ]; then
    # 2. 如果存在死循环目录、空目录或已失效的软链接，先彻底清理
    if [ -L "models" ]; then
        rm -f "models"
    elif [ -d "models" ]; then
        rm -rf "models"
    fi

    # 3. 按优先级寻找有效模型库（采用绝对路径，彻底杜绝相对路径递归死循环）
    TARGET_MODELS=""
    PARENT_DIR="$(cd .. 2>/dev/null && pwd)/models"
    for CANDIDATE in "$PARENT_DIR" "/hirain/hi_opt/algo/container_gongwang/models" "/models"; do
        if [ -d "$CANDIDATE" ] && { [ -f "$CANDIDATE/best.pt" ] || [ -f "$CANDIDATE/best.engine" ] || [ -f "$CANDIDATE/best.onnx" ]; }; then
            TARGET_MODELS="$CANDIDATE"
            break
        fi
    done

    # 4. 建立绝对路径软链接
    if [ -n "$TARGET_MODELS" ]; then
        ln -sfn "$TARGET_MODELS" models
        echo "🔗 [模型探测] 已安全关联模型库: ${BASE_DIR}/models -> ${TARGET_MODELS}"
    else
        echo "⚠️ [模型探测] 未在父级或标准路径探测到有效模型库，请检查模型文件是否存在！"
    fi
else
    echo "✅ [模型探测] 本地 models 目录已存在有效模型，无需重新关联"
fi

# 1. 创建日志目录（放在 local_cache/logs 下，宿主机桌面直接可见）
LOG_DIR="${BASE_DIR}/local_cache/logs"
mkdir -p "$LOG_DIR"

# 2. 生成当前时间戳文件名（如: monitor_20260915_104523.log）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/monitor_${TIMESTAMP}.log"
LATEST_LOG="${LOG_DIR}/latest.log"

echo "======================================================================"
echo "🚀 [弓网监测系统] 正在启动..."
echo "📄 本次日志文件: ${LOG_FILE}"
echo "📄 最新日志快捷链接: ${LATEST_LOG}"
echo "======================================================================"

# 3. 强制 UTF-8 编码，彻底杜绝 Windows 查看乱码；全局配置 RTSP TCP 传输与超时双保险
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp|stimeout;3000000|max_delay;500000"

# 4. 允许通过参数指定配置文件，默认读取 config/config_prod.yaml
CONFIG_FILE="${1:-config/config_prod.yaml}"

# 5. 启动主程序：
#    - python3 -u 禁用缓冲，日志秒级写入
#    - sed 自动剥离终端 ANSI 彩色转义字符，确保文本绝对纯净
#    - tee 同时在屏幕打印并保存进时间戳日志
python3 -u main.py "$CONFIG_FILE" 2>&1 \
    | sed -u 's/\x1b\[[0-9;]*[a-zA-Z]//g' \
    | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -ne 0 ]; then
    echo "⚠️ [系统提示] 主程序异常退出，退出码: ${EXIT_CODE}" | tee -a "$LOG_FILE"
    if [ $EXIT_CODE -eq 139 ]; then
        echo "💥 [致命提示] 发生 Segmentation fault (段错误 139)，请检查底层 C/CUDA 动态链接库或解码库。" | tee -a "$LOG_FILE"
    fi
fi

# 运行结束后同步生成一份 latest.log，方便一键查看最新结果
cp -f "$LOG_FILE" "$LATEST_LOG" 2>/dev/null || true
