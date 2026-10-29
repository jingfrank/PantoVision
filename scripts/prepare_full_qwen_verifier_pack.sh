#!/usr/bin/env bash
set -e

DEST="/mnt/T4/jyb/qwen_vlm_defense_workspace"

echo "=== 1. 拷贝工控防线核心代码 (fastad_detector & fastad_core) ==="
mkdir -p "$DEST/fastad_detector"
mkdir -p "$DEST/fastad_core"
cp -r /mnt/T4/jyb/gongwang_clean/fastad_detector/* "$DEST/fastad_detector/"
cp -r /mnt/T4/jyb/gongwang_clean/fastad_core/* "$DEST/fastad_core/"

echo "=== 2. 拷贝工控初检模型权重 (FastAD & YOLO) ==="
cp -v /mnt/T4/jyb/gongwang_clean/models/fastad_layer2_256_reg1e4_nq3_fmq1.pt "$DEST/models/"
cp -v /mnt/T4/jyb/gongwang_clean/models/best.pt "$DEST/models/"

echo "=== 3. 拷贝工控防线后的 Qwen 核验核心脚本与流水线 ==="
cp -v /mnt/T4/jyb/gongwang_qwen3_vl/test_fastad_classifier_qwen3_verifier.py "$DEST/tests_and_benchmarks/"
cp -v /mnt/T4/jyb/gongwang_qwen3_vl/test_fastad_trigger_qwen3_vl_raw.py "$DEST/tests_and_benchmarks/"
cp -v /mnt/T4/jyb/gongwang_qwen3_vl/test_fastad_heatmap_qwen3_vl.py "$DEST/tests_and_benchmarks/"
cp -v /mnt/T4/jyb/gongwang_qwen3_vl/test_qwen3_vl_pure_blind_review.py "$DEST/tests_and_benchmarks/"
cp -v /mnt/T4/jyb/gongwang_qwen3_vl/test_qwen3_vl_on_clean_heatmaps.py "$DEST/tests_and_benchmarks/"

echo "=== 4. 拷贝误报基准清单 ==="
cp -v /media/ibd/USB_DISK/manifest_false_alarms.json "$DEST/tests_and_benchmarks/"
cp -v /media/ibd/USB_DISK/manifest_false_alarms.csv "$DEST/tests_and_benchmarks/"

echo "=== 5. 准备典型测试样例图片集 ==="
mkdir -p "$DEST/test_samples"
cp -v /mnt/T4/jyb/gongwang_jetson_vllm/foreign_obj.png "$DEST/test_samples/"
cp -v /mnt/T4/jyb/gongwang_fastad/foreign_obj_crop.jpg "$DEST/test_samples/" 2>/dev/null || true
cp -v /mnt/T4/jyb/gongwang_fastad/spark_current_roi_vis.jpg "$DEST/test_samples/" 2>/dev/null || true
cp -v /mnt/T4/jyb/gongwang_qwen3_vl/debug_spark_100s.jpg "$DEST/test_samples/" 2>/dev/null || true
cp -v /media/ibd/USB_DISK/pantograph_broken.jpeg "$DEST/test_samples/" 2>/dev/null || true
cp -v /media/ibd/USB_DISK/pantograph_tilt.jpg "$DEST/test_samples/" 2>/dev/null || true
cp -v /media/ibd/USB_DISK/pantograph_piao.jpg "$DEST/test_samples/" 2>/dev/null || true

echo "=== 同步完成 ==="
