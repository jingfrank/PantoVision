#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
弓网多算法分层检测体系 - 典型视频离线基准评测套件 (test_benchmark_suite.py)
对 ./弓网典型视频/ 目录下的所有工况视频运行统一检测调度层与融合引擎，
验证 6 大类故障（火花、脏污、倾斜、飘弓、异物、结构异常）的识别准确率与推理时延。
"""

import os
import sys
import time
import argparse
import glob
import cv2
import numpy as np
from typing import Dict, List, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
FASTAD_CORE = os.path.join(BASE_DIR, "fastad_core")
if os.path.exists(FASTAD_CORE) and FASTAD_CORE not in sys.path:
    sys.path.append(FASTAD_CORE)

from core.data_types import FramePacket, FusionResult, DetectionContext
from core.scheduler import DetectionScheduler
from core.fusion import FusionEngine, FAULT_NAME_MAP
from core.output import Visualizer


def evaluate_video(
    video_path: str,
    scheduler: DetectionScheduler,
    fusion: FusionEngine,
    visualizer: Visualizer,
    output_dir: str,
    start_frame: int = 0,
    max_frames: int = 0,
    stride: int = 1,
    save_snapshots: bool = True
) -> Dict[str, Any]:
    """单视频离线评测"""
    video_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return {"error": "cannot open video"}

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_frames = total_frames if max_frames <= 0 else min(total_frames, max_frames)
    print(f"\n========================================================")
    print(f"▶ 正在测试视频: {video_name}")
    print(f"  分辨率: {width}x{height} | 帧率: {fps:.1f} FPS | 目标总帧: {target_frames} (采样步长: {stride})")
    print(f"========================================================")

    # 重置调度器所有插件的状态（时序窗口、EMA历史等）及融合决策层缓存
    scheduler.reset_all()
    fusion.reset()

    save_subdir = os.path.join(output_dir, os.path.splitext(video_name)[0])
    if save_snapshots:
        os.makedirs(save_subdir, exist_ok=True)

    frame_idx = start_frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    processed_count = 0
    latencies = []
    fault_counts: Dict[str, int] = {k: 0 for k in FAULT_NAME_MAP.keys()}
    first_trigger_frame: Dict[str, int] = {}
    alarm_events: List[Dict[str, Any]] = []

    last_active_set = set()

    t_start = time.time()
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_idx += 1
            if max_frames > 0 and processed_count >= max_frames:
                break

            if stride > 1 and (frame_idx % stride != 0):
                continue

            processed_count += 1
            pts = (frame_idx - 1) / fps
            packet = FramePacket(
                frame=frame,
                timestamp=pts,
                frame_idx=frame_idx,
                car_number=6,
                camera_id=0,
                is_panto_raised=True,
                speed=0.0,
                train_id="CR400BF"
            )

            # 1. 调度层沙箱安全分发
            t0 = time.time()
            raw_results = scheduler.dispatch(packet)
            
            # 2. 融合层综合决策
            fusion_res = fusion.decide(raw_results, packet)
            t_cost = (time.time() - t0) * 1000.0
            latencies.append(t_cost)

            # 3. 统计生效告警
            current_active = {a["fault_type"] for a in fusion_res.active_alarms}
            new_alarms = current_active - last_active_set

            if new_alarms:
                for f_type in new_alarms:
                    cn_name = FAULT_NAME_MAP.get(f_type, f_type)
                    fault_counts[f_type] = fault_counts.get(f_type, 0) + 1
                    if f_type not in first_trigger_frame:
                        first_trigger_frame[f_type] = frame_idx

                    # 记录事件
                    alarm_events.append({
                        "frame": frame_idx,
                        "timestamp_s": round(pts, 2),
                        "fault_type": f_type,
                        "chinese_name": cn_name,
                        "score": round(fusion_res.overall_score, 3),
                        "level": fusion_res.overall_level,
                    })

                    print(f"  [Frame {frame_idx:05d} / {pts:6.2f}s] [{t_cost:5.1f}ms] 🚨 触发新告警: 【{cn_name}】(score={fusion_res.overall_score:.2f})")

                    if save_snapshots:
                        hud_img = visualizer.render_hud(frame, fusion_res, packet)
                        snap_name = f"alarm_f{frame_idx:05d}_{pts:06.1f}s_{f_type}.jpg"
                        snap_path = os.path.join(save_subdir, snap_name)
                        cv2.imwrite(snap_path, hud_img)

            last_active_set = set(current_active)

            if processed_count % 30 == 0 or processed_count == 1:
                active_str = ", ".join([FAULT_NAME_MAP.get(a, a) for a in current_active]) if current_active else "正常"
                print(f"  进度: [{frame_idx:05d}/{target_frames}] 时延: {t_cost:5.1f}ms | 状态: {active_str}")

    except KeyboardInterrupt:
        print("\n[WARN] 用户手动中断视频评测")
    finally:
        cap.release()

    total_time = time.time() - t_start
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
    fps_measured = processed_count / total_time if total_time > 0 else 0.0

    print(f"\n--- 视频 {video_name} 评测结果汇总 ---")
    print(f"处理帧数: {processed_count} 帧 | 耗时: {total_time:.2f}s | 实测平均帧率: {fps_measured:.1f} FPS")
    print(f"平均时延: {avg_latency:.2f}ms | P95 时延: {p95_latency:.2f}ms")
    print(f"告警统计: {', '.join([f'{FAULT_NAME_MAP.get(k, k)}={v}次' for k, v in fault_counts.items() if v > 0]) or '全段正常无告警'}")

    return {
        "video": video_name,
        "total_frames": total_frames,
        "processed_frames": processed_count,
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "fps": round(fps_measured, 1),
        "fault_counts": fault_counts,
        "first_trigger": first_trigger_frame,
        "alarm_events": alarm_events,
    }


def evaluate_image(
    image_path: str,
    scheduler: DetectionScheduler,
    fusion: FusionEngine,
    visualizer: Visualizer,
    output_dir: str,
    save_snapshot: bool = True
) -> Dict[str, Any]:
    """单张图片静态离线评测与诊断"""
    img_name = os.path.basename(image_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] 无法读取图片文件: {image_path}")
        return {"error": "cannot open image"}

    h, w = img.shape[:2]
    print(f"\n========================================================")
    print(f"▶ 正在测试单张图片: {img_name} (分辨率: {w}x{h})")
    print(f"========================================================")

    # 重置调度器所有插件的状态及融合层状态
    scheduler.reset_all()
    fusion.reset()

    packet = FramePacket(
        frame=img,
        timestamp=0.0,
        frame_idx=1,
        car_number=6,
        camera_id=0,
        is_panto_raised=True,
        speed=0.0,
        train_id="CR400BF"
    )
    context = DetectionContext(packet=packet, shared_data={"single_image_mode": True})

    t0 = time.time()
    # 强制调度运行所有启用的检测器插件 (force_run_all=True)
    raw_results = scheduler.dispatch(packet, context=context, force_run_all=True)
    fusion_res = fusion.decide(raw_results, packet)
    latency_ms = (time.time() - t0) * 1000.0

    print(f"【推理耗时】: {latency_ms:.1f} ms")
    print(f"【综合诊断】: {fusion_res.primary_fault} (全局存在异常: {fusion_res.has_fault})")
    print("-" * 75)
    print(f"{'检测器插件':<22} | {'是否异常':<8} | {'置信度/分数':<10} | {'告警分级':<8} | {'详细定位/诊断信息'}")
    print("-" * 75)

    fault_counts = {k: 0 for k in FAULT_NAME_MAP.keys()}
    for name, res in fusion_res.detector_results.items():
        cn_name = FAULT_NAME_MAP.get(res.fault_type, res.fault_type)
        if res.is_abnormal:
            abn_str = "🚨 异常"
            fault_counts[res.fault_type] = fault_counts.get(res.fault_type, 0) + 1
        elif res.extra.get("suppressed_by"):
            abn_str = "🛡️ 抑制"
        else:
            abn_str = "正常"

        extra_info = []
        if res.bbox and res.is_abnormal:
            extra_info.append(f"bbox={res.bbox}")
        if res.extra.get("suppressed_by"):
            extra_info.append(f"suppressed_by={res.extra['suppressed_by']}")
        if res.fault_type == "tilt":
            extra_info.append(f"rel_tilt={res.extra.get('relative_tilt_deg', 0)}°({res.extra.get('status', 'NORMAL')})")
        elif res.fault_type == "dirty":
            extra_info.append(f"cells={res.extra.get('persistent_cells', 0)}, anom={res.extra.get('image_score', 0)}")
        elif res.fault_type in ("foreign_object", "structure_abnormal") and res.is_abnormal:
            cnt = res.extra.get('foreign_count', res.extra.get('defect_count', 0))
            extra_info.append(f"count={cnt}")
        detail_str = ", ".join(extra_info) if extra_info else "-"

        tag = f"{name}({cn_name})"
        print(f"{tag:<22} | {abn_str:<8} | {res.score:<10.2f} | {res.level:<8} | {detail_str}")

    snap_path = None
    if save_snapshot:
        save_dir = os.path.join(output_dir, "images")
        os.makedirs(save_dir, exist_ok=True)
        hud_frame = visualizer.render_hud(img, fusion_res, packet)
        snap_path = os.path.join(save_dir, f"eval_{img_name}")
        cv2.imwrite(snap_path, hud_frame)
        print("-" * 75)
        print(f"📸 诊断标注图已保存至: {snap_path}")

    return {
        "image": img_name,
        "primary_fault": fusion_res.primary_fault,
        "has_fault": fusion_res.has_fault,
        "latency_ms": round(latency_ms, 2),
        "fault_counts": fault_counts,
        "active_alarms": fusion_res.active_alarms,
        "snapshot_path": snap_path,
    }


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".flv"}


def main():
    parser = argparse.ArgumentParser(description="弓网视频与图像离线基准评测套件")
    parser.add_argument("--config", type=str, default="config/detectors.yaml", help="检测器配置文件")
    parser.add_argument("--input", type=str, default=None, help="统一输入路径 (支持单张图片、单视频、或包含图片/视频的目录)")
    parser.add_argument("--image", type=str, default=None, help="指定单张测试图片路径")
    parser.add_argument("--image-dir", type=str, default=None, help="指定测试图片目录路径")
    parser.add_argument("--video", type=str, default=None, help="指定单个测试视频路径")
    parser.add_argument("--dir", type=str, default=None, help="基准测试目录路径 (默认: 弓网典型视频)")
    parser.add_argument("--start-frame", type=int, default=0, help="视频起始测试帧号 (默认0)")
    parser.add_argument("--max-frames", type=int, default=0, help="视频最多评测帧数 (0为全部)")
    parser.add_argument("--stride", type=int, default=1, help="抽帧步长 (1为逐帧)")
    parser.add_argument("--output-dir", type=str, default="local_cache/benchmark_results", help="结果抓拍保存目录")
    args = parser.parse_args()

    cfg_path = os.path.join(BASE_DIR, args.config) if not os.path.isabs(args.config) else args.config
    out_dir = os.path.join(BASE_DIR, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    print("========================================================")
    print(" 🚀 初始化弓网分层检测调度引擎")
    print(f" 配置文件: {cfg_path}")
    print(f" 结果目录: {out_dir}")
    print("========================================================")

    scheduler = DetectionScheduler(cfg_path)
    fusion = FusionEngine()
    visualizer = Visualizer()

    # 1. 判断是否为单张图片评测
    single_img_path = None
    if args.image:
        single_img_path = args.image
    elif args.input and os.path.isfile(args.input) and os.path.splitext(args.input)[1].lower() in IMAGE_EXTS:
        single_img_path = args.input
    elif args.video and os.path.isfile(args.video) and os.path.splitext(args.video)[1].lower() in IMAGE_EXTS:
        single_img_path = args.video

    if single_img_path:
        full_path = os.path.join(BASE_DIR, single_img_path) if not os.path.isabs(single_img_path) else single_img_path
        evaluate_image(
            image_path=full_path,
            scheduler=scheduler,
            fusion=fusion,
            visualizer=visualizer,
            output_dir=out_dir
        )
        return

    # 2. 判断是否为图片目录批量评测
    img_dir_path = args.image_dir
    if not img_dir_path and args.input and os.path.isdir(args.input):
        # 检查目录下是否主要是图片
        imgs_in_dir = [f for f in os.listdir(args.input) if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
        if imgs_in_dir:
            img_dir_path = args.input

    if img_dir_path:
        full_dir = os.path.join(BASE_DIR, img_dir_path) if not os.path.isabs(img_dir_path) else img_dir_path
        img_list = sorted([
            os.path.join(full_dir, f) for f in os.listdir(full_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        ])
        print(f"\n📁 正在批量评测图片目录: {full_dir} (共 {len(img_list)} 张图像)")
        img_results = []
        for ipath in img_list:
            res = evaluate_image(
                image_path=ipath,
                scheduler=scheduler,
                fusion=fusion,
                visualizer=visualizer,
                output_dir=out_dir
            )
            img_results.append(res)

        print("\n" + "=" * 80)
        print(" 📊 图片批量离线评测最终汇总大表")
        print("=" * 80)
        print(f"{'图片文件名':<25} | {'耗时(ms)':<10} | {'综合主诊断':<18} | {'检出异常项'}")
        print("-" * 80)
        for r in img_results:
            if "error" in r:
                continue
            iname = r["image"]
            l_ms = r["latency_ms"]
            p_fault = r["primary_fault"]
            active = [f"{FAULT_NAME_MAP.get(k, k)}:{v}" for k, v in r["fault_counts"].items() if v > 0]
            active_str = ", ".join(active) if active else "无异常"
            print(f"{iname:<25} | {l_ms:<10.1f} | {p_fault:<18} | {active_str}")
        print("=" * 80 + "\n")
        return

    # 3. 视频评测逻辑
    if args.video:
        video_list = [args.video]
    elif args.input and os.path.isfile(args.input):
        video_list = [args.input]
    else:
        vdir_name = args.dir or "弓网典型视频"
        vdir = os.path.join(BASE_DIR, vdir_name) if not os.path.isabs(vdir_name) else vdir_name
        video_list = sorted(glob.glob(os.path.join(vdir, "*.mp4")))

    all_results = []
    for vpath in video_list:
        if vpath.endswith(".bak"):
            continue
        res = evaluate_video(
            video_path=vpath,
            scheduler=scheduler,
            fusion=fusion,
            visualizer=visualizer,
            output_dir=out_dir,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            stride=args.stride,
        )
        all_results.append(res)

    # 打印最终视频对比大表
    print("\n" + "=" * 80)
    print(" 📊 全场景基准离线评测最终汇总大表")
    print("=" * 80)
    print(f"{'视频文件名':<18} | {'帧数':<6} | {'FPS':<6} | {'均延(ms)':<8} | {'P95(ms)':<8} | {'检出故障类别与次数'}")
    print("-" * 80)
    for r in all_results:
        if "error" in r:
            continue
        vname = r["video"]
        n_f = r["processed_frames"]
        m_fps = r["fps"]
        avg_l = r["avg_latency_ms"]
        p95_l = r["p95_latency_ms"]
        active = [f"{FAULT_NAME_MAP.get(k, k)}:{v}" for k, v in r["fault_counts"].items() if v > 0]
        active_str = ", ".join(active) if active else "无告警(正常)"
        print(f"{vname:<18} | {n_f:<6} | {m_fps:<6.1f} | {avg_l:<8.1f} | {p95_l:<8.1f} | {active_str}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

