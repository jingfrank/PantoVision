#!/bin/bash
# ==============================================================================
# 受电弓监测系统 - U盘一键极速增量镜像同步脚本
# ==============================================================================

USB_DIR="${1:-/media/ibd/USB_DISK/gongwang_clean}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/"

if [ ! -d "$(dirname "${USB_DIR}")" ]; then
    echo "❌ 未检测到 U 盘挂载点: $(dirname "${USB_DIR}")"
    echo "💡 请插入 U 盘后再试，或手动指定路径: ./deploy/sync_to_usb.sh /media/xxx/xxx/gongwang_clean"
    exit 1
fi

mkdir -p "${USB_DIR}"

echo "======================================================================"
echo "💾 开始增量同步至 U 盘 (仅同步变动代码，跳过重复模型拷贝)"
echo "   本地源目录: ${LOCAL_DIR}"
echo "   目标 U 盘: ${USB_DIR}"
echo "======================================================================"

rsync -avh --progress \
    --exclude='.git' \
    --exclude='local_cache' \
    --exclude='events' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.swp' \
    --exclude='.pytest_cache' \
    --exclude='*.engine' \
    --exclude='Qwen3-VL-4B-Instruct' \
    "${LOCAL_DIR}" "${USB_DIR}/"

sync
echo "✅ U 盘增量同步完成，已刷新写入缓存 (sync)，可安全拔出 U 盘！"
