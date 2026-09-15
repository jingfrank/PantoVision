#!/bin/bash
# ==============================================================================
# 动车组弓网监测系统 - 启动脚本（自动时间戳日志归档版）
# 参考: gongwang_jetson_vllm/start.sh
# ==============================================================================

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

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

# 3. 强制 UTF-8 编码，彻底杜绝 Windows 查看乱码
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

# 4. 允许通过参数指定配置文件，默认读取 config/config_prod.yaml
CONFIG_FILE="${1:-config/config_prod.yaml}"

# 5. 启动主程序：
#    - python3 -u 禁用缓冲，日志秒级写入
#    - sed 自动剥离终端 ANSI 彩色转义字符，确保文本绝对纯净
#    - tee 同时在屏幕打印并保存进时间戳日志
python3 -u main.py "$CONFIG_FILE" 2>&1 \
    | sed -u 's/\x1b\[[0-9;]*[a-zA-Z]//g' \
    | tee -a "$LOG_FILE"

# 运行结束后同步生成一份 latest.log，方便一键查看最新结果
cp -f "$LOG_FILE" "$LATEST_LOG" 2>/dev/null || true
