#!/bin/bash
# ==============================================================================
# 受电弓监测系统 - 从 U 盘增量同步回本机工程脚本 (sync_from_usb.sh)
# ==============================================================================

set -euo pipefail

USB_SRC_DIR="${1:-/media/ibd/USB_DISK/gongwang_clean}"
LOCAL_DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "======================================================================"
echo "📥 开始从 U 盘增量同步更新至本机工程"
echo "   源目录 (U 盘): ${USB_SRC_DIR}/"
echo "   目标目录 (本机): ${LOCAL_DEST_DIR}/"
echo "======================================================================"

if [ ! -d "${USB_SRC_DIR}" ]; then
    echo "❌ 错误: 未检测到 U 盘源目录: ${USB_SRC_DIR}"
    echo "   请确认 U 盘已插入并挂载，或手动传入路径，例如:"
    echo "   ./deploy/sync_from_usb.sh /media/xxx/USB_DISK/gongwang_clean"
    exit 1
fi

# 排除 FAT32/exFAT 权限污染，仅同步内容与时间戳
rsync -rvltD \
    --no-perms \
    --exclude='.git' \
    --exclude='local_cache' \
    --exclude='events' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.swp' \
    "${USB_SRC_DIR}/" "${LOCAL_DEST_DIR}/"

echo "======================================================================"
echo "✅ 同步完成！U 盘最新修改已成功合入本机工程。"
echo "======================================================================"
