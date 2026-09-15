#!/bin/bash
# ==============================================================================
# 受电弓监测系统 - Jetson AGX Orin 极速增量网络同步脚本
# ==============================================================================
# 默认优先尝试:
# 1. 命令行指定的 IP (例如: ./deploy/sync_to_jetson.sh 172.10.1.188)
# 2. 环境变量 JETSON_IP
# 3. Jetson 原生 USB Type-C 虚拟网卡 IP (192.168.55.1，插线即连)
# ==============================================================================

JETSON_IP="${1:-${JETSON_IP:-192.168.55.1}}"
JETSON_USER="${JETSON_USER:-hirain}"
TARGET_DIR="${TARGET_DIR:-/home/hirain/Desktop/gongwang_clean}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/"

echo "======================================================================"
echo "🚀 开始增量同步至 Jetson 设备"
echo "   目标主机: ${JETSON_USER}@${JETSON_IP}:${TARGET_DIR}"
echo "   源工程目录: ${LOCAL_DIR}"
echo "======================================================================"

# 测试网络连通性 (ping 1包)
if ! ping -c 1 -W 2 "${JETSON_IP}" > /dev/null 2>&1; then
    echo "❌ 无法 ping 通目标 IP: ${JETSON_IP}"
    echo "💡 提示:"
    echo "   1. 若通过网线/局域网连接，请指定 IP，例如: ./deploy/sync_to_jetson.sh 192.168.0.xxx"
    echo "   2. 若通过 Type-C 数据线连接，请确保插入 Jetson 调试口 (默认 192.168.55.1)"
    exit 1
fi

# 执行增量差分同步 (仅传输修改文件，自动跳过已有模型权重)
rsync -avzP \
    --exclude='.git' \
    --exclude='local_cache' \
    --exclude='events' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.docker-compose.yml.swp' \
    "${LOCAL_DIR}" "${JETSON_USER}@${JETSON_IP}:${TARGET_DIR}/"

if [ $? -eq 0 ]; then
    echo "✅ 同步完成！修改已在 Jetson 端生效。"
else
    echo "⚠️ 同步中断，请检查 SSH 权限 (推荐先配置: ssh-copy-id ${JETSON_USER}@${JETSON_IP})。"
fi
