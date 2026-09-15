#!/bin/bash
# ==============================================================================
# 受电弓监测系统 - Jetson AGX Orin 极速增量网络同步脚本 (sync_to_jetson.sh)
# ==============================================================================

set -eo pipefail

JETSON_IP="172.10.1.5"
JETSON_USER="${JETSON_USER:-hirain}"
TARGET_DIR="${TARGET_DIR:-/home/hirain/Desktop/gongwang_clean}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/"
INCLUDE_MODELS=false

# 参数解析
while [[ $# -gt 0 ]]; do
    case "$1" in
        --include-models|-m)
            INCLUDE_MODELS=true
            shift
            ;;
        --user|-u)
            JETSON_USER="$2"
            shift 2
            ;;
        --dir|-d)
            TARGET_DIR="$2"
            shift 2
            ;;
        *)
            if [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
                JETSON_IP="$1"
            fi
            shift
            ;;
    esac
done

echo "======================================================================"
echo "🚀 开始增量同步至 Jetson 设备"
echo "   目标主机: ${JETSON_USER}@${JETSON_IP}:${TARGET_DIR}"
echo "   源工程目录: ${LOCAL_DIR}"
if [ "$INCLUDE_MODELS" = true ]; then
    echo "   传输模式: [包含 models 权重 (~950MB)]"
else
    echo "   传输模式: [纯代码/配置增量模式 (~200KB)] (自动跳过 models，秒级完成)"
fi
echo "======================================================================"

# 1. 网络连通性测试 (优先 ping，其次探测 22 端口)
CONNECT_OK=false
if ping -c 1 -W 2 "${JETSON_IP}" > /dev/null 2>&1; then
    CONNECT_OK=true
elif timeout 2 bash -c "</dev/tcp/${JETSON_IP}/22" > /dev/null 2>&1; then
    CONNECT_OK=true
fi

if [ "$CONNECT_OK" = false ]; then
    echo "❌ 无法连通目标 IP: ${JETSON_IP}"
    echo ""
    echo "💡 网线直连排查指南:"
    SUBNET_PREFIX="${JETSON_IP%.*}"
    echo "   1. 确认网线已插入本机网口与 Jetson 网口。"
    echo "   2. 本机网卡需配置同网段 IP (例如 ${SUBNET_PREFIX}.100)。"
    echo "      可执行命令快速配置临时 IP (以 eno1 为例):"
    echo "      👉 sudo ip addr add ${SUBNET_PREFIX}.100/24 dev eno1 && sudo ip link set eno1 up"
    echo "   3. 若 Jetson 为其他 IP，可直接指定运行:"
    echo "      👉 ./deploy/sync_to_jetson.sh 192.168.0.xxx"
    echo "   4. 若使用 Type-C 数据线直连，默认 IP 为 192.168.55.1:"
    echo "      👉 ./deploy/sync_to_jetson.sh 192.168.55.1"
    exit 1
fi

# 2. 构造排除列表
EXCLUDES=(
    "--exclude=.git"
    "--exclude=local_cache"
    "--exclude=events"
    "--exclude=__pycache__"
    "--exclude=*.pyc"
    "--exclude=*.swp"
    "--exclude=*.tmp"
)

if [ "$INCLUDE_MODELS" = false ]; then
    EXCLUDES+=("--exclude=models")
fi

# 3. 执行极速增量传输
rsync -avzP "${EXCLUDES[@]}" "${LOCAL_DIR}" "${JETSON_USER}@${JETSON_IP}:${TARGET_DIR}/"

if [ $? -eq 0 ]; then
    echo "======================================================================"
    echo "✅ 同步成功！修改已在 Jetson 端实时生效。"
    if [ "$INCLUDE_MODELS" = false ]; then
        echo "💡 提示: 若需同步 models 权重，请运行: ./deploy/sync_to_jetson.sh --include-models"
    fi
    echo "======================================================================"
else
    echo "⚠️ 同步中断，请检查 SSH 权限 (推荐执行免密配置: ssh-copy-id ${JETSON_USER}@${JETSON_IP})。"
fi
