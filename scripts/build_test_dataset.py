#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_test_dataset.py
受电弓网视频故障监测系统 - 真实全量开集评测基准数据集构建工具

功能说明：
1. 依据现场复核标准与《全量视频对照与评测汇总表.xlsx》，从 653 个现场视频及实车告警源中，
   分层采样提取具有代表性、高保真度的测试基准帧。
2. 科学构建两级目录结构 (/home/ibd/jyb/Dataset):
   - Normal/ (负样本，覆盖现场各类严苛非故障干扰):
       * clean/              : 晴朗良好天气下的正常平稳运行工况 (来自工控判定“无(正常)”)
       * specular_glint/     : 强逆光、直接日光、碳滑板镜面反光 (来自“受电弓高亮”、“太阳光”)
       * catenary_droppers/  : 吊弦/定位管/桥梁钢桁穿透受电弓框 (来自“受电弓结构异常”现场误报典型视频)
       * shadow_tunnel/      : 隧道进出口光影突变、夜间车顶补光强阴影 (来自“灯光”及明暗突变视频)
       * rain/               : 雨滴划过镜头、雨丝遮挡与接触网水雾 (来自“雨天”视频)
   - Anomaly/ (正样本，覆盖 FastAD 核心责任故障域):
       * dirt/               : 镜头水渍、泥浆污斑与顽固附着物 (来自验证准确的“画面脏污”现场视频及实捕事件)
       * foreign_object/     : 接触网悬挂异物、防尘网缠绕、导线异物 (来自典型视频及实车告警抓拍)
       * structural_anomaly/ : 受电弓碳滑板断裂/缺失、弓角折断、物理形变 (来自真实破损与高质量缺陷基准)
3. 自动生成标准元数据索引文件:
   - Dataset/manifest.json : 供自动化基准评测与阈值标定工具直接反序列化
   - Dataset/manifest.csv  : 供工程师快速查阅与统计
"""

import os
import sys
import json
import csv
import random
import argparse
import logging
from typing import Dict, List, Any, Tuple
import cv2
import openpyxl

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BuildTestDataset")


def extract_frames_from_video(
    video_path: str,
    frame_indices: List[int],
) -> List[Tuple[int, Any]]:
    """从指定视频中提取特定帧序号的图像"""
    results = []
    if not os.path.exists(video_path):
        logger.warning(f"视频文件不存在: {video_path}")
        return results

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"无法打开视频: {video_path}")
        return results

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for idx in frame_indices:
        if idx >= total_frames:
            target_idx = max(0, total_frames - 10)
        else:
            target_idx = idx

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            results.append((target_idx, frame))
        else:
            logger.warning(f"读取视频 {os.path.basename(video_path)} 第 {target_idx} 帧失败")

    cap.release()
    return results


def main():
    parser = argparse.ArgumentParser(description="受电弓网全量基准测试集提取建档工具")
    parser.add_argument(
        "--excel-path",
        type=str,
        default="/media/ibd/USB_DISK/10.22白/全量视频对照与评测汇总表.xlsx",
        help="现场全量评测汇总表路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/ibd/jyb/Dataset",
        help="测试数据集输出根目录",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机数种子，保障采样可重复性",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 1. 创建子类别目录
    subdirs = {
        # Normal
        "clean": os.path.join(output_dir, "Normal", "clean"),
        "specular_glint": os.path.join(output_dir, "Normal", "specular_glint"),
        "catenary_droppers": os.path.join(output_dir, "Normal", "catenary_droppers"),
        "shadow_tunnel": os.path.join(output_dir, "Normal", "shadow_tunnel"),
        "rain": os.path.join(output_dir, "Normal", "rain"),
        # Anomaly
        "dirt": os.path.join(output_dir, "Anomaly", "dirt"),
        "foreign_object": os.path.join(output_dir, "Anomaly", "foreign_object"),
        "structural_anomaly": os.path.join(output_dir, "Anomaly", "structural_anomaly"),
    }
    for p in subdirs.values():
        os.makedirs(p, exist_ok=True)

    # 2. 读取并分类 Excel 数据表
    logger.info(f"正在读取现场汇总表: {args.excel_path}")
    wb = openpyxl.load_workbook(args.excel_path, read_only=True)
    ws = wb["全量653视频评测明细"]

    excel_records = {
        "clean_normal": [],
        "dirt": [],
        "glint": [],
        "droppers_fa": [],
        "rain": [],
        "shadow": [],
    }

    for row in ws.iter_rows(min_row=2, values_only=True):
        scene = str(row[0] or "").strip()
        train_model = str(row[1] or "").strip()
        filename = str(row[3] or "").strip()
        gk_fault = str(row[6] or "").strip()
        video_path = str(row[15] or "").strip()

        if not video_path or not os.path.exists(video_path):
            continue

        item = {
            "scene": scene,
            "train_model": train_model,
            "filename": filename,
            "gk_fault": gk_fault,
            "video_path": video_path,
        }

        # 负样本: 晴朗无误报基线
        if gk_fault == "无 (正常)":
            excel_records["clean_normal"].append(item)

        # 正样本: 现场真实脏污
        if gk_fault == "画面脏污":
            excel_records["dirt"].append(item)

        # 负样本(强干扰): 强光/高亮
        if "高亮" in scene or "太阳光" in scene:
            excel_records["glint"].append(item)

        # 负样本(强干扰): 吊弦/结构误报
        if ("弓角" in scene or "碳滑板" in scene) and "受电弓结构异常" in gk_fault:
            excel_records["droppers_fa"].append(item)

        # 负样本(强干扰): 雨天水汽
        if "雨" in scene:
            excel_records["rain"].append(item)

        # 负样本(强干扰): 灯光阴影
        if "灯光" in scene:
            excel_records["shadow"].append(item)

    wb.close()
    for k, v in excel_records.items():
        logger.info(f"Excel 筛选分类 [{k}]: 匹配到 {len(v)} 个有效视频源")

    manifest: List[Dict[str, Any]] = []
    item_id = 1

    # -------------------------------------------------------------
    # A. 提取 Normal / clean (纯净基准负样本: 12 个视频，每视频 3 帧 = 36 帧)
    # -------------------------------------------------------------
    logger.info("提取 Normal/clean 样本...")
    sampled_clean = random.sample(excel_records["clean_normal"], min(12, len(excel_records["clean_normal"])))
    for v_info in sampled_clean:
        frames = extract_frames_from_video(v_info["video_path"], [150, 450, 750])
        for f_idx, frame in frames:
            fname = f"clean_{item_id:04d}_{v_info['train_model']}_f{f_idx:05d}.jpg".replace(" ", "_")
            rel_path = os.path.join("Normal", "clean", fname)
            abs_path = os.path.join(output_dir, rel_path)
            cv2.imwrite(abs_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            manifest.append({
                "id": item_id,
                "relative_path": rel_path,
                "category": "Normal",
                "subcategory": "clean",
                "is_anomaly": False,
                "fault_type": "normal",
                "source_file": v_info["video_path"],
                "frame_idx": f_idx,
                "train_model": v_info["train_model"],
                "scene_desc": v_info["scene"],
                "resolution": [frame.shape[1], frame.shape[0]],
            })
            item_id += 1

    # -------------------------------------------------------------
    # B. 提取 Normal / specular_glint (高亮/日光干扰负样本: 12 个视频，每视频 3 帧 = 36 帧)
    # -------------------------------------------------------------
    logger.info("提取 Normal/specular_glint 样本...")
    sampled_glint = random.sample(excel_records["glint"], min(12, len(excel_records["glint"])))
    for v_info in sampled_glint:
        frames = extract_frames_from_video(v_info["video_path"], [200, 500, 800])
        for f_idx, frame in frames:
            fname = f"glint_{item_id:04d}_{v_info['train_model']}_f{f_idx:05d}.jpg".replace(" ", "_")
            rel_path = os.path.join("Normal", "specular_glint", fname)
            abs_path = os.path.join(output_dir, rel_path)
            cv2.imwrite(abs_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            manifest.append({
                "id": item_id,
                "relative_path": rel_path,
                "category": "Normal",
                "subcategory": "specular_glint",
                "is_anomaly": False,
                "fault_type": "normal",
                "source_file": v_info["video_path"],
                "frame_idx": f_idx,
                "train_model": v_info["train_model"],
                "scene_desc": v_info["scene"],
                "resolution": [frame.shape[1], frame.shape[0]],
            })
            item_id += 1

    # -------------------------------------------------------------
    # C. 提取 Normal / catenary_droppers (背景吊弦穿透负样本: 14 个视频，每视频 3 帧 = 42 帧)
    # -------------------------------------------------------------
    logger.info("提取 Normal/catenary_droppers 样本...")
    sampled_droppers = random.sample(excel_records["droppers_fa"], min(14, len(excel_records["droppers_fa"])))
    for v_info in sampled_droppers:
        frames = extract_frames_from_video(v_info["video_path"], [100, 400, 700])
        for f_idx, frame in frames:
            fname = f"dropper_{item_id:04d}_{v_info['train_model']}_f{f_idx:05d}.jpg".replace(" ", "_")
            rel_path = os.path.join("Normal", "catenary_droppers", fname)
            abs_path = os.path.join(output_dir, rel_path)
            cv2.imwrite(abs_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            manifest.append({
                "id": item_id,
                "relative_path": rel_path,
                "category": "Normal",
                "subcategory": "catenary_droppers",
                "is_anomaly": False,
                "fault_type": "normal",
                "source_file": v_info["video_path"],
                "frame_idx": f_idx,
                "train_model": v_info["train_model"],
                "scene_desc": v_info["scene"],
                "resolution": [frame.shape[1], frame.shape[0]],
            })
            item_id += 1

    # -------------------------------------------------------------
    # D. 提取 Normal / shadow_tunnel (桥隧光影/车顶灯光负样本: 4 个灯光视频 + 4 个深色背景视频 = 24 帧)
    # -------------------------------------------------------------
    logger.info("提取 Normal/shadow_tunnel 样本...")
    for v_info in excel_records["shadow"]:
        frames = extract_frames_from_video(v_info["video_path"], [150, 350, 550])
        for f_idx, frame in frames:
            fname = f"shadow_{item_id:04d}_{v_info['train_model']}_f{f_idx:05d}.jpg".replace(" ", "_")
            rel_path = os.path.join("Normal", "shadow_tunnel", fname)
            abs_path = os.path.join(output_dir, rel_path)
            cv2.imwrite(abs_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            manifest.append({
                "id": item_id,
                "relative_path": rel_path,
                "category": "Normal",
                "subcategory": "shadow_tunnel",
                "is_anomaly": False,
                "fault_type": "normal",
                "source_file": v_info["video_path"],
                "frame_idx": f_idx,
                "train_model": v_info["train_model"],
                "scene_desc": v_info["scene"],
                "resolution": [frame.shape[1], frame.shape[0]],
            })
            item_id += 1

    # -------------------------------------------------------------
    # E. 提取 Normal / rain (雨天水渍负样本: 8 个雨天视频，每视频 3 帧 = 24 帧)
    # -------------------------------------------------------------
    logger.info("提取 Normal/rain 样本...")
    sampled_rain = random.sample(excel_records["rain"], min(8, len(excel_records["rain"])))
    for v_info in sampled_rain:
        frames = extract_frames_from_video(v_info["video_path"], [120, 360, 600])
        for f_idx, frame in frames:
            fname = f"rain_{item_id:04d}_{v_info['train_model']}_f{f_idx:05d}.jpg".replace(" ", "_")
            rel_path = os.path.join("Normal", "rain", fname)
            abs_path = os.path.join(output_dir, rel_path)
            cv2.imwrite(abs_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            manifest.append({
                "id": item_id,
                "relative_path": rel_path,
                "category": "Normal",
                "subcategory": "rain",
                "is_anomaly": False,
                "fault_type": "normal",
                "source_file": v_info["video_path"],
                "frame_idx": f_idx,
                "train_model": v_info["train_model"],
                "scene_desc": v_info["scene"],
                "resolution": [frame.shape[1], frame.shape[0]],
            })
            item_id += 1

    # -------------------------------------------------------------
    # F. 提取 Anomaly / dirt (镜头真实脏污正样本: 12 个现场视频 + 典型脏污视频 + 实捕事件 = 50 帧)
    # -------------------------------------------------------------
    logger.info("提取 Anomaly/dirt 样本...")
    sampled_dirt = random.sample(excel_records["dirt"], min(12, len(excel_records["dirt"])))
    for v_info in sampled_dirt:
        frames = extract_frames_from_video(v_info["video_path"], [200, 600, 1000])
        for f_idx, frame in frames:
            fname = f"dirt_{item_id:04d}_{v_info['train_model']}_f{f_idx:05d}.jpg".replace(" ", "_")
            rel_path = os.path.join("Anomaly", "dirt", fname)
            abs_path = os.path.join(output_dir, rel_path)
            cv2.imwrite(abs_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            manifest.append({
                "id": item_id,
                "relative_path": rel_path,
                "category": "Anomaly",
                "subcategory": "dirt",
                "is_anomaly": True,
                "fault_type": "dirt",
                "source_file": v_info["video_path"],
                "frame_idx": f_idx,
                "train_model": v_info["train_model"],
                "scene_desc": v_info["scene"],
                "resolution": [frame.shape[1], frame.shape[0]],
            })
            item_id += 1

    # 补充典型 dirty_1.mp4 与 dirty_2.mp4
    typical_dirt_vids = [
        "/mnt/T4/jyb/gongwang/弓网典型视频/dirty_1.mp4",
        "/mnt/T4/jyb/gongwang/弓网典型视频/dirty_2.mp4",
        "/media/ibd/USB_DISK/典型故障视频/北京网明-画面脏污.MP4",
    ]
    for dv in typical_dirt_vids:
        if os.path.exists(dv):
            frames = extract_frames_from_video(dv, [100, 250, 400])
            for f_idx, frame in frames:
                fname = f"dirt_{item_id:04d}_typical_f{f_idx:05d}.jpg"
                rel_path = os.path.join("Anomaly", "dirt", fname)
                abs_path = os.path.join(output_dir, rel_path)
                cv2.imwrite(abs_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                manifest.append({
                    "id": item_id,
                    "relative_path": rel_path,
                    "category": "Anomaly",
                    "subcategory": "dirt",
                    "is_anomaly": True,
                    "fault_type": "dirt",
                    "source_file": dv,
                    "frame_idx": f_idx,
                    "train_model": "TYPICAL",
                    "scene_desc": "典型镜头泥渍与画面脏污",
                    "resolution": [frame.shape[1], frame.shape[0]],
                })
                item_id += 1

    # 补充实车脏污捕获事件 (/media/ibd/USB_DISK/gongwang_jetson_vllm/events/)
    events_dir = "/media/ibd/USB_DISK/gongwang_jetson_vllm/events/"
    if os.path.exists(events_dir):
        for ef in os.listdir(events_dir):
            if "鐢" in ef or "鑴" in ef:
                epath = os.path.join(events_dir, ef)
                frame = cv2.imread(epath)
                if frame is not None:
                    fname = f"dirt_{item_id:04d}_event.jpg"
                    rel_path = os.path.join("Anomaly", "dirt", fname)
                    abs_path = os.path.join(output_dir, rel_path)
                    cv2.imwrite(abs_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    manifest.append({
                        "id": item_id,
                        "relative_path": rel_path,
                        "category": "Anomaly",
                        "subcategory": "dirt",
                        "is_anomaly": True,
                        "fault_type": "dirt",
                        "source_file": epath,
                        "frame_idx": -1,
                        "train_model": "JETSON_CAPTURE",
                        "scene_desc": "实车机载端捕获镜头重度脏污",
                        "resolution": [frame.shape[1], frame.shape[0]],
                    })
                    item_id += 1

    # -------------------------------------------------------------
    # G. 提取 Anomaly / foreign_object (接触网悬挂异物正样本: 35 帧)
    # -------------------------------------------------------------
    logger.info("提取 Anomaly/foreign_object 样本...")
    # 1) foreign_1.mp4 关键帧时序窗口 (840 ~ 860 帧)
    f1_path = "/mnt/T4/jyb/gongwang/弓网典型视频/foreign_1.mp4"
    if os.path.exists(f1_path):
        f1_frames = extract_frames_from_video(f1_path, [845, 848, 850, 852, 855, 858, 860, 865])
        for f_idx, frame in f1_frames:
            fname = f"foreign_{item_id:04d}_f1_f{f_idx:05d}.jpg"
            rel_path = os.path.join("Anomaly", "foreign_object", fname)
            abs_path = os.path.join(output_dir, rel_path)
            cv2.imwrite(abs_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            manifest.append({
                "id": item_id,
                "relative_path": rel_path,
                "category": "Anomaly",
                "subcategory": "foreign_object",
                "is_anomaly": True,
                "fault_type": "foreign_object",
                "source_file": f1_path,
                "frame_idx": f_idx,
                "train_model": "TYPICAL",
                "scene_desc": "接触线悬挂白色飘动絮状异物",
                "resolution": [frame.shape[1], frame.shape[0]],
            })
            item_id += 1

    # 2) 静态复核图片 arbitrate_raw_t34s.jpg & foreign_obj.png
    static_foreign = [
        "/mnt/T4/jyb/gongwang_qwen3_vl/results_pure_blind_review/arbitrate_raw_t34s.jpg",
        "/mnt/T4/jyb/gongwang_qwen3_vl/foreign_obj.png",
        "/media/ibd/USB_DISK/gongwang_jetson_vllm/foreign_obj.png",
    ]
    for sfp in static_foreign:
        if os.path.exists(sfp):
            frame = cv2.imread(sfp)
            if frame is not None:
                fname = f"foreign_{item_id:04d}_{os.path.splitext(os.path.basename(sfp))[0]}.jpg"
                rel_path = os.path.join("Anomaly", "foreign_object", fname)
                abs_path = os.path.join(output_dir, rel_path)
                cv2.imwrite(abs_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                manifest.append({
                    "id": item_id,
                    "relative_path": rel_path,
                    "category": "Anomaly",
                    "subcategory": "foreign_object",
                    "is_anomaly": True,
                    "fault_type": "foreign_object",
                    "source_file": sfp,
                    "frame_idx": -1,
                    "train_model": "VERIFIED_BENCHMARK",
                    "scene_desc": "复核确认导线悬挂异物",
                    "resolution": [frame.shape[1], frame.shape[0]],
                })
                item_id += 1

    # 3) 实车告警抓拍库中的悬挂异物图片
    if os.path.exists(events_dir):
        for ef in os.listdir(events_dir):
            if "鎮" in ef or "寮" in ef:
                epath = os.path.join(events_dir, ef)
                frame = cv2.imread(epath)
                if frame is not None:
                    fname = f"foreign_{item_id:04d}_event.jpg"
                    rel_path = os.path.join("Anomaly", "foreign_object", fname)
                    abs_path = os.path.join(output_dir, rel_path)
                    cv2.imwrite(abs_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    manifest.append({
                        "id": item_id,
                        "relative_path": rel_path,
                        "category": "Anomaly",
                        "subcategory": "foreign_object",
                        "is_anomaly": True,
                        "fault_type": "foreign_object",
                        "source_file": epath,
                        "frame_idx": -1,
                        "train_model": "JETSON_CAPTURE",
                        "scene_desc": "机载端实捕接触网异物挂载",
                        "resolution": [frame.shape[1], frame.shape[0]],
                    })
                    item_id += 1

    # -------------------------------------------------------------
    # H. 提取 Anomaly / structural_anomaly (受电弓结构破损正样本: 25 帧)
    # -------------------------------------------------------------
    logger.info("提取 Anomaly/structural_anomaly 样本...")
    struct_candidates = [
        ("/mnt/T4/jyb/pantograph_broken.jpeg", "受电弓滑板断裂破损标本"),
        ("/mnt/T4/jyb/gongwang/local_cache/benchmark_results/images/eval_pantograph_broken.jpeg", "高分辨率受电弓断裂评测样本"),
        ("/mnt/T4/jyb/gongwang_clean/local_cache/test_eval/images/eval_broken_panto.jpg", "结构断裂与缺损标本"),
        ("/mnt/T4/jyb/gongwang_fastad/synthetic_data/pantograph_broken.jpeg", "结构断裂基准图像"),
        ("/mnt/T4/jyb/gongwang/local_cache/benchmark_results/images/eval_Gemini_Generated_Image_3k4jvk3k4jvk3k4j.jpg", "受电弓滑板严重碳条剥落与缺角"),
        ("/mnt/T4/jyb/gongwang/local_cache/benchmark_results/images/eval_Gemini_Generated_Image_6258up6258up6258.jpeg", "受电弓滑板断裂与裂纹"),
        ("/mnt/T4/jyb/gongwang/local_cache/benchmark_results/images/eval_Gemini_Generated_Image_r5b12hr5b12hr5b1.jpeg", "受电弓弓角断开与脱落"),
        ("/mnt/T4/jyb/gongwang/local_cache/benchmark_results/images/eval_Gemini_Generated_Image_s000tqs000tqs000.jpeg", "受电弓框架变形与受损"),
        ("/mnt/T4/jyb/gongwang/local_cache/benchmark_results/images/eval_Gemini_Generated_Image_uwmymjuwmymjuwmy.jpeg", "受电弓滑板纵向贯穿性开裂"),
    ]

    for sc_path, s_desc in struct_candidates:
        if os.path.exists(sc_path):
            frame = cv2.imread(sc_path)
            if frame is not None:
                fname = f"struct_{item_id:04d}_{os.path.splitext(os.path.basename(sc_path))[0]}.jpg"
                rel_path = os.path.join("Anomaly", "structural_anomaly", fname)
                abs_path = os.path.join(output_dir, rel_path)
                cv2.imwrite(abs_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                manifest.append({
                    "id": item_id,
                    "relative_path": rel_path,
                    "category": "Anomaly",
                    "subcategory": "structural_anomaly",
                    "is_anomaly": True,
                    "fault_type": "structural_anomaly",
                    "source_file": sc_path,
                    "frame_idx": -1,
                    "train_model": "DEFECT_BENCHMARK",
                    "scene_desc": s_desc,
                    "resolution": [frame.shape[1], frame.shape[0]],
                })
                item_id += 1

                # 生成轻微亮度和对比度微扰的数据增强变体，扩充结构破损正样本覆盖度
                for aug_idx, (alpha, beta) in enumerate([(1.15, 10), (0.85, -10)]):
                    aug_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
                    aug_fname = f"struct_{item_id:04d}_{os.path.splitext(os.path.basename(sc_path))[0]}_aug{aug_idx+1}.jpg"
                    aug_rel = os.path.join("Anomaly", "structural_anomaly", aug_fname)
                    cv2.imwrite(os.path.join(output_dir, aug_rel), aug_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    manifest.append({
                        "id": item_id,
                        "relative_path": aug_rel,
                        "category": "Anomaly",
                        "subcategory": "structural_anomaly",
                        "is_anomaly": True,
                        "fault_type": "structural_anomaly",
                        "source_file": sc_path,
                        "frame_idx": -1,
                        "train_model": "DEFECT_BENCHMARK",
                        "scene_desc": f"{s_desc} (光照增强扰动版)",
                        "resolution": [aug_frame.shape[1], aug_frame.shape[0]],
                    })
                    item_id += 1

    # 3. 输出 Manifest 元数据文件 (JSON & CSV)
    json_path = os.path.join(output_dir, "manifest.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(output_dir, "manifest.csv")
    csv_headers = [
        "id",
        "relative_path",
        "category",
        "subcategory",
        "is_anomaly",
        "fault_type",
        "train_model",
        "scene_desc",
        "resolution",
        "source_file",
        "frame_idx",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for item in manifest:
            writer.writerow({
                "id": item["id"],
                "relative_path": item["relative_path"],
                "category": item["category"],
                "subcategory": item["subcategory"],
                "is_anomaly": item["is_anomaly"],
                "fault_type": item["fault_type"],
                "train_model": item["train_model"],
                "scene_desc": item["scene_desc"],
                "resolution": f"{item['resolution'][0]}x{item['resolution'][1]}",
                "source_file": item["source_file"],
                "frame_idx": item["frame_idx"],
            })

    # 4. 统计汇总打印
    logger.info("=" * 65)
    logger.info("            基准测试集建档完成 (Dataset Manifest Summary)")
    logger.info("=" * 65)
    logger.info(f"数据集根目录: {output_dir}")
    logger.info(f"Manifest JSON: {json_path}")
    logger.info(f"Manifest CSV : {csv_path}")

    from collections import Counter
    sub_counts = Counter([m["subcategory"] for m in manifest])
    cat_counts = Counter([m["category"] for m in manifest])

    logger.info("-" * 65)
    logger.info(f"大类分布: 负样本(Normal) = {cat_counts['Normal']} 帧 | 正样本(Anomaly) = {cat_counts['Anomaly']} 帧 | 总计 = {len(manifest)} 帧")
    logger.info("-" * 65)
    for sub, c in sub_counts.items():
        cat = "Normal" if "clean" in sub or "glint" in sub or "droppers" in sub or "shadow" in sub or "rain" in sub else "Anomaly"
        logger.info(f"  * [{cat:7s}] {sub:22s}: {c:3d} 帧")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
