#!/usr/bin/env bash
set -e

SRC_JETSON="/home/ibd/jyb/gongwang_jetson_vllm"
SRC_CLEAN="/mnt/T4/jyb/gongwang_clean"
DEST="/mnt/T4/jyb/qwen_vlm_defense_workspace"

echo "=== 1. 创建目标目录结构 ==="
mkdir -p "$DEST/models/Qwen3-VL-4B-Instruct-AWQ-4bit"
mkdir -p "$DEST/config"
mkdir -p "$DEST/tests_and_benchmarks"
mkdir -p "$DEST/deploy"
mkdir -p "$DEST/docs"

echo "=== 2. 拷贝核心 Python 代码与主服务 ==="
cp "$SRC_JETSON/twostage_vlm_engine.py" "$DEST/"
cp "$SRC_JETSON/vllm_client_engine.py" "$DEST/"
cp "$SRC_JETSON/platform_utils.py" "$DEST/"
cp "$SRC_JETSON/ai_detector.py" "$DEST/"
cp "$SRC_JETSON/monitor_main.py" "$DEST/"
cp "$SRC_JETSON/main.py" "$DEST/"
cp "$SRC_JETSON/usrcode.py" "$DEST/"
cp "$SRC_JETSON/phm_service.py" "$DEST/"
cp "$SRC_JETSON/ftp_uploader.py" "$DEST/"
cp "$SRC_JETSON/AppConfig.xlsx" "$DEST/"
cp "$SRC_JETSON/start.sh" "$DEST/"
cp "$SRC_JETSON/close.py" "$DEST/"
cp "$SRC_JETSON/close.sh" "$DEST/"
cp "$SRC_JETSON/download_model.py" "$DEST/"
cp "$SRC_JETSON/foreign_obj.png" "$DEST/"

echo "=== 3. 拷贝配置文件 ==="
cp -r "$SRC_JETSON/config/"* "$DEST/config/"

echo "=== 4. 拷贝部署与技术文档 ==="
cp -r "$SRC_JETSON/deploy/"* "$DEST/deploy/"
cp -r "$SRC_JETSON/docs/"* "$DEST/docs/"

echo "=== 5. 拷贝测试脚本与基准脚本 ==="
# gongwang_jetson_vllm 内部的评测
cp "$SRC_JETSON/foreign_localization.py" "$DEST/tests_and_benchmarks/"
cp "$SRC_JETSON/test_foreign_object_vlm.py" "$DEST/tests_and_benchmarks/"
cp "$SRC_JETSON/test_image_vs_video_twostage.py" "$DEST/tests_and_benchmarks/"
cp "$SRC_JETSON/test_qwen3_understanding.py" "$DEST/tests_and_benchmarks/"
cp "$SRC_JETSON/rtsp_twostage_vlm.py" "$DEST/tests_and_benchmarks/"
cp "$SRC_JETSON/rtsp_twostage_temporal_consistent.py" "$DEST/tests_and_benchmarks/"
cp "$SRC_JETSON/rtsp_twostage_hybrid_vlm.py" "$DEST/tests_and_benchmarks/"
cp "$SRC_JETSON/rtsp_sliding_window_vlm.py" "$DEST/tests_and_benchmarks/"
cp "$SRC_JETSON/run_full_video_test.py" "$DEST/tests_and_benchmarks/"
cp "$SRC_JETSON/benchmark_vllm_vs_transformers.py" "$DEST/tests_and_benchmarks/"

# gongwang_clean 中的 Qwen 防线专项评测
cp "$SRC_CLEAN/tests/test_qwen3_vl_inference.py" "$DEST/tests_and_benchmarks/"
cp "$SRC_CLEAN/tests/test_qwen3_20_false_alarm_videos.py" "$DEST/tests_and_benchmarks/"
cp "$SRC_CLEAN/tests/test_qwen3_vs_industrial_false_alarms.py" "$DEST/tests_and_benchmarks/"
cp "$SRC_CLEAN/tests/test_qwen_full_dataset_benchmark.py" "$DEST/tests_and_benchmarks/"
cp "$SRC_CLEAN/scratch/test_foreign_object_qwen_deep.py" "$DEST/tests_and_benchmarks/"
cp "$SRC_CLEAN/scratch/run_actual_faults_qwen.py" "$DEST/tests_and_benchmarks/"

echo "=== 6. 拷贝模型配置文件（排除 model.safetensors 大权重）==="
rsync -av --exclude="*.safetensors" "$SRC_JETSON/models/Qwen3-VL-4B-Instruct-AWQ-4bit/" "$DEST/models/Qwen3-VL-4B-Instruct-AWQ-4bit/"

echo "=== 拷贝完成 ==="
