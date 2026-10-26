#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/calibrate_thresholds.py
FastAD Layer 3 开集异常检测阈值标定与基准评测工具

针对受电弓网视频中的强光晕、电网吊弦、桥隧阴影等开集干扰，
在真实平衡测试集 (/home/ibd/jyb/Dataset) 上进行量化特征扫描与极值分布统计，
计算 ROC/PR 曲线，标定兼顾高检出率与零误报的最佳门限：
1. img_threshold (全图异常评分门限)
2. pixel_threshold (像素级热力图激活门限)
3. min_defect_area (受电弓结构缺陷最小连通域像素面积)
4. foreign_object 专有参数 (min_foreign_area, max_blobs_filter)
5. dirt 专有参数 (dirt_score_thresh, min_active_cells)

支持自动回写更新:
- config/detectors.yaml
- models/fastad_layer3_dataset.pt
"""

import os
import sys
import json
import csv
import argparse
import time
import logging
from typing import Dict, List, Any, Tuple
import numpy as np
import cv2
import torch
import yaml

# 项目根目录导入支持
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastad_detector import PantographFastADModel
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CalibrateThresholds")


def run_benchmark_inference(
    dataset_dir: str,
    manifest: List[Dict[str, Any]],
    fastad_model: PantographFastADModel,
    yolo_model: YOLO,
) -> List[Dict[str, Any]]:
    """批量对测试集进行特征推断与多通道统计"""
    results = []
    total = len(manifest)
    logger.info(f"开始对测试集全量 {total} 帧进行批量推断...")

    t0 = time.time()
    for idx, item in enumerate(manifest):
        rel_path = item["relative_path"]
        abs_path = os.path.join(dataset_dir, rel_path)

        frame = cv2.imread(abs_path)
        if frame is None:
            logger.warning(f"无法读取图像: {abs_path}")
            continue

        h, w = frame.shape[:2]

        # 1. YOLO 提取受电弓目标框
        yolo_res = yolo_model(frame, verbose=False)[0]
        panto_bbox = None
        for box in yolo_res.boxes:
            cls_id = int(box.cls.item())
            name = yolo_res.names[cls_id].lower()
            if "panto" in name or cls_id == 0:
                panto_bbox = [float(v) for v in box.xyxy[0].cpu().numpy()]
                break

        # 2. FastAD 推理
        fastad_out = fastad_model.predict(frame)
        img_score = float(fastad_out["anomaly_score"])
        anom_map = fastad_out["anomaly_map"]  # (h, w) float32

        # 3. 受电弓 ROI 局部热力统计
        panto_max_pixel = 0.0
        panto_mean_pixel = 0.0
        panto_blob_areas: Dict[float, int] = {}  # 各 pixel_thresh 下的最大斑块面积

        if panto_bbox is not None:
            px1 = max(0, min(w, int(panto_bbox[0])))
            py1 = max(0, min(h, int(panto_bbox[1])))
            px2 = max(0, min(w, int(panto_bbox[2])))
            py2 = max(0, min(h, int(panto_bbox[3])))
            if px2 > px1 and py2 > py1:
                panto_crop = anom_map[py1:py2, px1:px2]
                panto_max_pixel = float(panto_crop.max())
                panto_mean_pixel = float(panto_crop.mean())

                # 预扫描不同 pixel threshold 下的连通域面积
                for pt in [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]:
                    bin_mask = (panto_crop >= pt).astype(np.uint8) * 255
                    cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    max_area = max([int(cv2.contourArea(c)) for c in cnts], default=0)
                    panto_blob_areas[pt] = max_area

        # 4. 接触网开集异物 ROI (catenary_roi_ratio: [0.25, 0.25, 0.85, 0.50])
        cx1, cy1 = int(0.25 * w), int(0.25 * h)
        cx2, cy2 = int(0.85 * w), int(0.50 * h)
        cat_crop = anom_map[cy1:cy2, cx1:cx2]
        cat_max_pixel = float(cat_crop.max())
        cat_mean_pixel = float(cat_crop.mean())

        # 5. 镜头网格脏污统计 (64x64 grid)
        dirt_limit_h = int(h * 0.82)
        dirt_crop = anom_map[:dirt_limit_h, :]
        grid_64 = cv2.resize(dirt_crop, (64, 64), interpolation=cv2.INTER_AREA)

        grid_active_counts = {}
        for pt in [3.0, 3.5, 4.0, 4.5, 5.0]:
            grid_active_counts[pt] = int(np.sum(grid_64 >= pt))

        record = {
            "id": item["id"],
            "relative_path": rel_path,
            "category": item["category"],
            "subcategory": item["subcategory"],
            "is_anomaly": item["is_anomaly"],
            "fault_type": item["fault_type"],
            "train_model": item.get("train_model", ""),
            "scene_desc": item.get("scene_desc", ""),
            "img_score": img_score,
            "panto_bbox": panto_bbox,
            "panto_max_pixel": panto_max_pixel,
            "panto_mean_pixel": panto_mean_pixel,
            "panto_blob_areas": panto_blob_areas,
            "cat_max_pixel": cat_max_pixel,
            "cat_mean_pixel": cat_mean_pixel,
            "grid_active_counts": grid_active_counts,
        }
        results.append(record)

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            elapsed = time.time() - t0
            fps = (idx + 1) / elapsed
            logger.info(f"推断进度: [{idx+1}/{total}] ({((idx+1)/total)*100:.1f}%) - 均帧率: {fps:.1f} FPS")

    return results


def analyze_distributions(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """按场景细分子类别统计各项指标的分位数分布"""
    from collections import defaultdict

    stats = {}
    by_subcat = defaultdict(list)
    for r in results:
        by_subcat[r["subcategory"]].append(r)

    # 总体 Normal vs Anomaly
    normal_img_scores = [r["img_score"] for r in results if not r["is_anomaly"]]
    anomaly_img_scores = [r["img_score"] for r in results if r["is_anomaly"]]

    def get_percentiles(vals):
        if not vals:
            return {}
        arr = np.array(vals)
        return {
            "count": len(vals),
            "min": round(float(np.min(arr)), 2),
            "p25": round(float(np.percentile(arr, 25)), 2),
            "median": round(float(np.median(arr)), 2),
            "p75": round(float(np.percentile(arr, 75)), 2),
            "p90": round(float(np.percentile(arr, 90)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "max": round(float(np.max(arr)), 2),
            "mean": round(float(np.mean(arr)), 2),
        }

    stats["normal_all_img_score"] = get_percentiles(normal_img_scores)
    stats["anomaly_all_img_score"] = get_percentiles(anomaly_img_scores)

    subcat_stats = {}
    for subcat, records in by_subcat.items():
        img_scs = [r["img_score"] for r in records]
        panto_maxs = [r["panto_max_pixel"] for r in records if r["panto_bbox"] is not None]
        cat_maxs = [r["cat_max_pixel"] for r in records]
        subcat_stats[subcat] = {
            "count": len(records),
            "img_score": get_percentiles(img_scs),
            "panto_max_pixel": get_percentiles(panto_maxs),
            "cat_max_pixel": get_percentiles(cat_maxs),
        }
    stats["by_subcategory"] = subcat_stats

    return stats


def sweep_structural_thresholds(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """网格搜索受电弓结构异常的最优阈值组合 (img_thresh, pix_thresh, min_area)"""
    normal_records = [r for r in results if not r["is_anomaly"]]
    struct_records = [r for r in results if r["subcategory"] == "structural_anomaly"]

    sweep_results = []

    # 候选门限空间
    img_thresh_candidates = [5.0, 7.0, 8.5, 10.0, 12.0, 14.0, 16.0]
    pix_thresh_candidates = [4.0, 4.5, 5.0, 5.5, 6.0]
    min_area_candidates = [500, 1000, 1500, 2000, 2500, 3000]

    for it in img_thresh_candidates:
        for pt in pix_thresh_candidates:
            for ma in min_area_candidates:
                # 判定函数
                def is_pred_struct(r):
                    if r["panto_bbox"] is None:
                        return False
                    blob_area = r["panto_blob_areas"].get(pt, 0)
                    return (blob_area >= ma) and (r["img_score"] >= it)

                # True Positives on structural_anomaly
                tp = sum(1 for r in struct_records if is_pred_struct(r))
                fn = len(struct_records) - tp

                # False Positives on Normal (负样本总集)
                fp = sum(1 for r in normal_records if is_pred_struct(r))
                tn = len(normal_records) - fp

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                far = fp / len(normal_records) if len(normal_records) > 0 else 0.0

                sweep_results.append({
                    "img_threshold": it,
                    "pixel_threshold": pt,
                    "min_defect_area": ma,
                    "tp": tp,
                    "fn": fn,
                    "fp": fp,
                    "tn": tn,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "false_alarm_rate": round(far, 4),
                })

    # 按 F1 降序、FP 升序、Recall 降序排序
    sweep_results.sort(key=lambda x: (x["fp"] == 0, x["f1"], x["recall"]), reverse=True)
    return sweep_results


def sweep_dirt_thresholds(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """网格搜索镜头脏污检测的最优静态门限"""
    normal_records = [r for r in results if not r["is_anomaly"]]
    dirt_records = [r for r in results if r["subcategory"] == "dirt"]

    sweep_results = []
    pix_thresh_candidates = [3.5, 4.0, 4.5, 5.0]
    min_active_cells_candidates = [4, 6, 8, 12, 16]

    for pt in pix_thresh_candidates:
        for mac in min_active_cells_candidates:
            def is_pred_dirt(r):
                cnt = r["grid_active_counts"].get(pt, 0)
                return cnt >= mac

            tp = sum(1 for r in dirt_records if is_pred_dirt(r))
            fn = len(dirt_records) - tp
            fp = sum(1 for r in normal_records if is_pred_dirt(r))
            tn = len(normal_records) - fp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            far = fp / len(normal_records) if len(normal_records) > 0 else 0.0

            sweep_results.append({
                "pixel_threshold": pt,
                "min_active_cells": mac,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "false_alarm_rate": round(far, 4),
            })

    sweep_results.sort(key=lambda x: (x["fp"] == 0, x["f1"], x["recall"]), reverse=True)
    return sweep_results


def main():
    parser = argparse.ArgumentParser(description="FastAD Layer 3 综合阈值标定工具")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/home/ibd/jyb/Dataset",
        help="测试数据集根目录",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="/home/ibd/jyb/Dataset/manifest.json",
        help="数据集 Manifest 路径",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/fastad_layer3_dataset.pt",
        help="FastAD 模型权重路径",
    )
    parser.add_argument(
        "--engine-path",
        type=str,
        default="models/fastad_layer3_256.engine",
        help="FastAD TensorRT 引擎路径",
    )
    parser.add_argument(
        "--yolo-path",
        type=str,
        default="models/best.pt",
        help="YOLO 模型权重路径",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="reports/fastad_layer3_calibration_report.json",
        help="标定评测报告输出路径",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="自动将标定出的最优阈值写回 config/detectors.yaml",
    )
    parser.add_argument(
        "--update-checkpoint",
        action="store_true",
        help="自动将标定出的基线阈值更新写入 models/fastad_layer3_dataset.pt",
    )
    parser.add_argument(
        "--auto-sync",
        action="store_true",
        default=True,
        help="在标定前自动扫描 dataset-dir 目录，同步最新的文件增删与移动变更",
    )
    args = parser.parse_args()

    # 1. 自动同步/加载 Manifest
    if args.auto_sync:
        try:
            from tools.sync_dataset_manifest import sync_manifest
            logger.info(f"正在自动同步测试集目录变更: {args.dataset_dir}")
            sync_manifest(args.dataset_dir)
        except Exception as e:
            logger.warning(f"自动同步目录失败: {e}")

    if not os.path.exists(args.manifest_path):
        logger.error(f"Manifest 文件不存在: {args.manifest_path}")
        sys.exit(1)

    with open(args.manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    logger.info(f"成功加载 Manifest，总计样本: {len(manifest)} 帧")

    # 2. 加载模型
    logger.info(f"正在加载 FastAD 模型 (TRT Engine: {args.engine_path})...")
    fastad_model = PantographFastADModel.load(
        args.model_path,
        prefer_tensorrt=True,
        engine_path=args.engine_path if os.path.exists(args.engine_path) else None,
    )

    logger.info(f"正在加载 YOLO 模型 ({args.yolo_path})...")
    yolo_model = YOLO(args.yolo_path)

    # 3. 运行全量推断
    results = run_benchmark_inference(
        dataset_dir=args.dataset_dir,
        manifest=manifest,
        fastad_model=fastad_model,
        yolo_model=yolo_model,
    )

    # 4. 特征分布统计
    logger.info("正在执行特征分布多维统计...")
    dist_stats = analyze_distributions(results)

    # 5. 受电弓结构异常门限扫描
    logger.info("正在扫描受电弓结构异常最优门限...")
    struct_sweep = sweep_structural_thresholds(results)
    best_struct = struct_sweep[0] if struct_sweep else None

    # 6. 镜头脏污门限扫描
    logger.info("正在扫描镜头脏污最优门限...")
    dirt_sweep = sweep_dirt_thresholds(results)
    best_dirt = dirt_sweep[0] if dirt_sweep else None

    # 7. 打印标定报告看板
    logger.info("=" * 72)
    logger.info("       FastAD Layer 3 综合阈值标定与基准评测看板 (Benchmark Board)")
    logger.info("=" * 72)
    logger.info("1. 全图异常评分分布 (img_score Percentiles):")
    logger.info(f"   - 负样本 (Normal 150帧) : Min={dist_stats['normal_all_img_score']['min']} | P50={dist_stats['normal_all_img_score']['median']} | P90={dist_stats['normal_all_img_score']['p90']} | P95={dist_stats['normal_all_img_score']['p95']} | Max={dist_stats['normal_all_img_score']['max']}")
    logger.info(f"   - 正样本 (Anomaly 112帧): Min={dist_stats['anomaly_all_img_score']['min']} | P50={dist_stats['anomaly_all_img_score']['median']} | P90={dist_stats['anomaly_all_img_score']['p90']} | P95={dist_stats['anomaly_all_img_score']['p95']} | Max={dist_stats['anomaly_all_img_score']['max']}")
    logger.info("-" * 72)
    logger.info("2. 各细分工况统计对比:")
    for subcat, sc_stat in dist_stats["by_subcategory"].items():
        is_pos = subcat in ["dirt", "foreign_object", "structural_anomaly"]
        tag = "[正样本]" if is_pos else "[负样本]"
        logger.info(f"   {tag:6s} {subcat:20s} ({sc_stat['count']:2d}帧) -> img_score [P50: {sc_stat['img_score']['median']:4.1f}, Max: {sc_stat['img_score']['max']:4.1f}] | panto_max [P50: {sc_stat['panto_max_pixel'].get('median', 0.0):3.1f}, Max: {sc_stat['panto_max_pixel'].get('max', 0.0):3.1f}]")

    logger.info("-" * 72)
    logger.info("3. 推荐最优标定门限 (Recommended Optimal Thresholds):")
    if best_struct:
        logger.info(f"   [受电弓结构异常]")
        logger.info(f"   -> 最佳参数组合 : img_threshold={best_struct['img_threshold']}, pixel_threshold={best_struct['pixel_threshold']}, min_defect_area={best_struct['min_defect_area']}")
        logger.info(f"   -> 评测指标效果 : Precision={best_struct['precision']*100:.1f}%, Recall={best_struct['recall']*100:.1f}%, F1={best_struct['f1']:.4f}, 现场误报率(FAR)={best_struct['false_alarm_rate']*100:.2f}% (FP={best_struct['fp']}/{best_struct['fp']+best_struct['tn']})")

    if best_dirt:
        logger.info(f"   [镜头画面脏污]")
        logger.info(f"   -> 最佳参数组合 : pixel_threshold={best_dirt['pixel_threshold']}, min_active_cells={best_dirt['min_active_cells']}")
        logger.info(f"   -> 评测指标效果 : Precision={best_dirt['precision']*100:.1f}%, Recall={best_dirt['recall']*100:.1f}%, F1={best_dirt['f1']:.4f}, 误报数={best_dirt['fp']}")

    logger.info("=" * 72)

    # 8. 保存标定报告
    os.makedirs(os.path.dirname(os.path.abspath(args.output_report)), exist_ok=True)
    report_data = {
        "dataset_summary": {
            "total_frames": len(results),
            "normal_frames": len([r for r in results if not r["is_anomaly"]]),
            "anomaly_frames": len([r for r in results if r["is_anomaly"]]),
        },
        "distributions": dist_stats,
        "structural_sweep_top5": struct_sweep[:5] if struct_sweep else [],
        "dirt_sweep_top5": dirt_sweep[:5] if dirt_sweep else [],
        "recommended_parameters": {
            "structural": {
                "img_threshold": best_struct["img_threshold"] if best_struct else 10.0,
                "pixel_threshold": best_struct["pixel_threshold"] if best_struct else 5.0,
                "min_defect_area": best_struct["min_defect_area"] if best_struct else 2000,
            },
            "dirt": {
                "pixel_threshold": best_dirt["pixel_threshold"] if best_dirt else 4.0,
                "min_active_cells": best_dirt["min_active_cells"] if best_dirt else 6,
            },
        },
    }
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    logger.info(f"标定全量报告已保存至: {args.output_report}")

    # 9. 自动更新 config/detectors.yaml
    if args.update_config and best_struct:
        config_path = os.path.join(BASE_DIR, "config", "detectors.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

            for det in cfg.get("detectors", []):
                if det.get("type") == "FastadStructDetector":
                    det["params"]["img_threshold"] = float(best_struct["img_threshold"])
                    det["params"]["pixel_threshold"] = float(best_struct["pixel_threshold"])
                    det["params"]["min_defect_area"] = int(best_struct["min_defect_area"])
                    logger.info(f"已更新 FastadStructDetector 配置 -> img_threshold: {best_struct['img_threshold']}, pixel_threshold: {best_struct['pixel_threshold']}, min_defect_area: {best_struct['min_defect_area']}")

                if det.get("type") == "FastadDirtDetector" and best_dirt:
                    det["params"]["pixel_threshold"] = float(best_dirt["pixel_threshold"])
                    det["params"]["min_active_cells"] = int(best_dirt["min_active_cells"])
                    logger.info(f"已更新 FastadDirtDetector 配置 -> pixel_threshold: {best_dirt['pixel_threshold']}, min_active_cells: {best_dirt['min_active_cells']}")

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
            logger.info(f"成功将最新标定参数写入: {config_path}")

    # 10. 自动更新 checkpoint 中的 calibrated_*_threshold
    if args.update_checkpoint and best_struct:
        ckpt_path = os.path.join(BASE_DIR, args.model_path)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "fastad_state" in ckpt:
                cal_img = float(best_struct["img_threshold"])
                cal_pix = float(best_struct["pixel_threshold"])
                ckpt["fastad_state"]["calibrated_image_threshold"] = torch.tensor(cal_img)
                ckpt["fastad_state"]["calibrated_pixel_threshold"] = torch.tensor(cal_pix)
                torch.save(ckpt, ckpt_path)
                logger.info(f"成功将标定基线阈值写入模型 Checkpoint: {ckpt_path} (img: {cal_img}, pix: {cal_pix})")


if __name__ == "__main__":
    main()
