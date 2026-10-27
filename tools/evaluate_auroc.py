#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/evaluate_auroc.py
FastAD Layer 3 模型多维度 AUROC 与 PR-AUC 基准评测工具

功能：
1. 评估当前 FastAD Layer 3 模型在 /home/ibd/jyb/Dataset/ 测试集上的判别能力上限；
2. 计算多维度的 AUROC (Area Under ROC) 与 PR-AUC (Average Precision)：
   - 全局全图异常分 AUROC (Global Image-level AUROC)
   - 画面脏污专项 AUROC & PR-AUC (镜头网格激活度)
   - 受电弓结构异常专项 AUROC & PR-AUC (受电弓 ROI 内部极大值与连通域面积)
   - 悬挂异物专项 AUROC & PR-AUC (接触网 ROI 内部极大值)
3. 基于 ROC 曲线与 PR 曲线，推导不同工业工作点（Operating Points）：
   - 约登指数最优工作点 (Youden's J: Max Recall + Specificity)
   - 严格低误报工作点 (FPR <= 1% / FPR <= 5%)
   - F1-Score 最大化工作点
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, List, Any, Tuple
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, average_precision_score

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastad_detector import PantographFastADModel
from ultralytics import YOLO
from tools.sync_dataset_manifest import sync_manifest

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EvaluateAUROC")


def calculate_metrics_and_operating_points(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    name: str = "",
) -> Dict[str, Any]:
    """计算单个任务的 AUROC、PR-AUC 以及各类工业最佳切点"""
    num_pos = int(np.sum(y_true == 1))
    num_neg = int(np.sum(y_true == 0))

    if num_pos == 0 or num_neg == 0:
        return {
            "name": name,
            "num_pos": num_pos,
            "num_neg": num_neg,
            "auroc": float("nan"),
            "pr_auc": float("nan"),
            "youden_cutoff": None,
            "fpr1_cutoff": None,
            "fpr5_cutoff": None,
            "best_f1_cutoff": None,
        }

    # 1. AUROC
    try:
        auroc_val = float(roc_auc_score(y_true, y_scores))
    except Exception:
        auroc_val = float("nan")

    # 2. PR-AUC (Average Precision)
    try:
        pr_auc_val = float(average_precision_score(y_true, y_scores))
    except Exception:
        pr_auc_val = float("nan")

    # 3. ROC 曲线与切点
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # 约登指数 Youden's J = TPR - FPR
    j_scores = tpr - fpr
    best_j_idx = int(np.argmax(j_scores))
    youden_thresh = float(thresholds[best_j_idx])
    youden_tpr = float(tpr[best_j_idx])
    youden_fpr = float(fpr[best_j_idx])

    # 严格低误报切点: FPR <= 1% (若无则取最接近 0 的点)
    valid_fpr1 = np.where(fpr <= 0.0101)[0]
    idx_fpr1 = valid_fpr1[-1] if len(valid_fpr1) > 0 else 0
    cutoff_fpr1 = {
        "threshold": float(thresholds[idx_fpr1]),
        "recall": float(tpr[idx_fpr1]),
        "fpr": float(fpr[idx_fpr1]),
    }

    # 严格低误报切点: FPR <= 5%
    valid_fpr5 = np.where(fpr <= 0.0501)[0]
    idx_fpr5 = valid_fpr5[-1] if len(valid_fpr5) > 0 else 0
    cutoff_fpr5 = {
        "threshold": float(thresholds[idx_fpr5]),
        "recall": float(tpr[idx_fpr5]),
        "fpr": float(fpr[idx_fpr5]),
    }

    # 4. PR 曲线与 Best F1 切点
    precisions, recalls, pr_threshs = precision_recall_curve(y_true, y_scores)
    # 计算各个阈值下的 F1
    f1_scores = np.zeros_like(pr_threshs)
    for i in range(len(pr_threshs)):
        p = precisions[i]
        r = recalls[i]
        f1_scores[i] = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

    if len(f1_scores) > 0:
        best_f1_idx = int(np.argmax(f1_scores))
        best_f1_thresh = float(pr_threshs[best_f1_idx])
        best_f1_p = float(precisions[best_f1_idx])
        best_f1_r = float(recalls[best_f1_idx])
        best_f1_val = float(f1_scores[best_f1_idx])
    else:
        best_f1_thresh, best_f1_p, best_f1_r, best_f1_val = 0.0, 0.0, 0.0, 0.0

    return {
        "name": name,
        "num_pos": num_pos,
        "num_neg": num_neg,
        "auroc": auroc_val,
        "pr_auc": pr_auc_val,
        "youden_cutoff": {
            "threshold": youden_thresh,
            "recall": youden_tpr,
            "fpr": youden_fpr,
            "j_score": float(j_scores[best_j_idx]),
        },
        "fpr1_cutoff": cutoff_fpr1,
        "fpr5_cutoff": cutoff_fpr5,
        "best_f1_cutoff": {
            "threshold": best_f1_thresh,
            "precision": best_f1_p,
            "recall": best_f1_r,
            "f1": best_f1_val,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="FastAD Layer 3 AUROC 与 PR-AUC 评估工具")
    parser.add_argument("--dataset-dir", type=str, default="/home/ibd/jyb/Dataset", help="数据集根目录")
    parser.add_argument("--manifest-path", type=str, default="/home/ibd/jyb/Dataset/manifest.json", help="Manifest路径")
    parser.add_argument("--model-path", type=str, default="models/fastad_layer3_dataset.pt", help="模型权重路径")
    parser.add_argument("--engine-path", type=str, default="models/fastad_layer3_256.engine", help="TensorRT引擎路径")
    parser.add_argument("--yolo-path", type=str, default="models/best.pt", help="YOLO模型路径")
    parser.add_argument("--auto-sync", action="store_true", default=True, help="自动同步数据集目录")
    args = parser.parse_args()

    # 1. 自动同步
    if args.auto_sync:
        sync_manifest(args.dataset_dir)

    with open(args.manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    logger.info(f"成功加载 Manifest，总计样本: {len(manifest)} 帧")

    # 2. 加载模型
    logger.info(f"正在加载 FastAD 模型 (TRT: {args.engine_path})...")
    fastad_model = PantographFastADModel.load(
        args.model_path,
        prefer_tensorrt=True,
        engine_path=args.engine_path if os.path.exists(args.engine_path) else None,
    )
    logger.info("正在加载 YOLO 模型...")
    yolo_model = YOLO(args.yolo_path)

    # 3. 批量推断与特征收集
    total = len(manifest)
    logger.info(f"正在对当前数据集 {total} 帧进行批量推断提取评分...")

    records = []
    t0 = time.time()
    for idx, item in enumerate(manifest):
        rel_path = item["relative_path"]
        abs_path = os.path.join(args.dataset_dir, rel_path)
        frame = cv2.imread(abs_path)
        if frame is None:
            continue

        h, w = frame.shape[:2]

        # YOLO 受电弓定位
        yolo_res = yolo_model(frame, verbose=False)[0]
        panto_bbox = None
        for box in yolo_res.boxes:
            cls_id = int(box.cls.item())
            name = yolo_res.names[cls_id].lower()
            if "panto" in name or cls_id == 0:
                panto_bbox = [float(v) for v in box.xyxy[0].cpu().numpy()]
                break

        # FastAD 推理
        fastad_out = fastad_model.predict(frame)
        img_score = float(fastad_out["anomaly_score"])
        anom_map = fastad_out["anomaly_map"]

        # 受电弓 ROI 局部特征
        panto_max = 0.0
        panto_mean = 0.0
        panto_blob_35 = 0
        if panto_bbox is not None:
            px1 = max(0, min(w, int(panto_bbox[0])))
            py1 = max(0, min(h, int(panto_bbox[1])))
            px2 = max(0, min(w, int(panto_bbox[2])))
            py2 = max(0, min(h, int(panto_bbox[3])))
            if px2 > px1 and py2 > py1:
                panto_crop = anom_map[py1:py2, px1:px2]
                panto_max = float(panto_crop.max())
                panto_mean = float(panto_crop.mean())

                bin35 = (panto_crop >= 3.5).astype(np.uint8) * 255
                cnts, _ = cv2.findContours(bin35, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                panto_blob_35 = max([int(cv2.contourArea(c)) for c in cnts], default=0)

        # 接触网 ROI (异物)
        cx1, cy1, cx2, cy2 = int(0.25 * w), int(0.25 * h), int(0.85 * w), int(0.50 * h)
        cat_crop = anom_map[cy1:cy2, cx1:cx2]
        cat_max = float(cat_crop.max())

        # 镜头脏污特征 (64x64 网格)
        dirt_h = int(h * 0.82)
        dirt_crop = anom_map[:dirt_h, :]
        grid_64 = cv2.resize(dirt_crop, (64, 64), interpolation=cv2.INTER_AREA)
        dirt_active_cells = int(np.sum(grid_64 >= 4.0))
        dirt_grid_mean = float(np.mean(grid_64))

        records.append({
            "id": item["id"],
            "relative_path": rel_path,
            "category": item["category"],
            "subcategory": item["subcategory"],
            "is_anomaly": item["is_anomaly"],
            "fault_type": item["fault_type"],
            "img_score": img_score,
            "panto_max": panto_max,
            "panto_mean": panto_mean,
            "panto_blob_35": panto_blob_35,
            "cat_max": cat_max,
            "dirt_active_cells": dirt_active_cells,
            "dirt_grid_mean": dirt_grid_mean,
        })

    elapsed = time.time() - t0
    logger.info(f"特征提取完毕，共 {len(records)} 帧，耗时: {elapsed:.2f}s (均帧率: {len(records)/elapsed:.1f} FPS)")

    # 4. 构建不同任务的 (y_true, y_score)
    # 全局正常样本索引
    neg_mask = np.array([not r["is_anomaly"] for r in records])

    # 任务 A: 全局全图异常分 (全集正样本 vs 负样本)
    y_global_true = np.array([1 if r["is_anomaly"] else 0 for r in records])
    y_global_score = np.array([r["img_score"] for r in records])
    res_global = calculate_metrics_and_operating_points(y_global_true, y_global_score, "全图全局评分 (img_score)")

    # 任务 B: 画面脏污 (Dirt: 脏污正样本 vs 所有正常负样本)
    dirt_indices = [i for i, r in enumerate(records) if r["subcategory"] == "dirt" or not r["is_anomaly"]]
    y_dirt_true = np.array([1 if records[i]["subcategory"] == "dirt" else 0 for i in dirt_indices])
    y_dirt_score_cells = np.array([records[i]["dirt_active_cells"] for i in dirt_indices])
    y_dirt_score_mean = np.array([records[i]["dirt_grid_mean"] for i in dirt_indices])
    res_dirt_cells = calculate_metrics_and_operating_points(y_dirt_true, y_dirt_score_cells, "画面脏污 (网格激活数 cells@4.0)")
    res_dirt_mean = calculate_metrics_and_operating_points(y_dirt_true, y_dirt_score_mean, "画面脏污 (镜头网格均值分 mean)")

    # 任务 C: 受电弓结构异常 (Structural Anomaly: 结构正样本 vs 所有正常负样本)
    struct_indices = [i for i, r in enumerate(records) if r["subcategory"] == "structural_anomaly" or not r["is_anomaly"]]
    y_struct_true = np.array([1 if records[i]["subcategory"] == "structural_anomaly" else 0 for i in struct_indices])
    y_struct_pmax = np.array([records[i]["panto_max"] for i in struct_indices])
    y_struct_blob = np.array([records[i]["panto_blob_35"] for i in struct_indices])
    res_struct_pmax = calculate_metrics_and_operating_points(y_struct_true, y_struct_pmax, "受电弓结构异常 (局部峰值 panto_max)")
    res_struct_blob = calculate_metrics_and_operating_points(y_struct_true, y_struct_blob, "受电弓结构异常 (连通斑块面积 blob@3.5)")

    # 任务 D: 悬挂异物 (Foreign Object: 异物正样本 vs 所有正常负样本)
    foreign_indices = [i for i, r in enumerate(records) if r["subcategory"] == "foreign_object" or not r["is_anomaly"]]
    y_foreign_true = np.array([1 if records[i]["subcategory"] == "foreign_object" else 0 for i in foreign_indices])
    y_foreign_catmax = np.array([records[i]["cat_max"] for i in foreign_indices])
    res_foreign = calculate_metrics_and_operating_points(y_foreign_true, y_foreign_catmax, "悬挂异物 (接触网局部峰值 cat_max)")

    # 5. 打印规范的 AUROC & PR-AUC 看板报告
    print("\n" + "=" * 90)
    print("        FastAD Layer 3 多维度 AUROC / PR-AUC 与工业工作点评测大表")
    print("=" * 90)
    print(f"{'评测维度 / 任务指标':<35} | {'正/负样本':<10} | {'AUROC':<9} | {'PR-AUC':<9} | {'最佳F1切点(Thr/P/R/F1)':<24}")
    print("-" * 90)

    for res in [res_global, res_dirt_cells, res_dirt_mean, res_struct_pmax, res_struct_blob, res_foreign]:
        sample_str = f"{res['num_pos']}/{res['num_neg']}"
        auroc_str = f"{res['auroc']:.4f}" if not np.isnan(res['auroc']) else "N/A"
        prauc_str = f"{res['pr_auc']:.4f}" if not np.isnan(res['pr_auc']) else "N/A"
        f1_info = res["best_f1_cutoff"]
        if f1_info:
            f1_str = f"T={f1_info['threshold']:.2f} (P:{f1_info['precision']*100:.0f}% R:{f1_info['recall']*100:.0f}% F1:{f1_info['f1']:.2f})"
        else:
            f1_str = "N/A"
        print(f"{res['name']:<35} | {sample_str:<10} | {auroc_str:<9} | {prauc_str:<9} | {f1_str:<24}")

    print("=" * 90)

    print("\n" + "=" * 90)
    print("             工业级严格工作点对比 (Operating Points Analysis)")
    print("=" * 90)
    print(f"{'评测任务':<35} | {'约登J点 (平衡优选)':<26} | {'严格低误报 (FPR<=1%)':<24}")
    print("-" * 90)

    for res in [res_dirt_cells, res_dirt_mean, res_struct_blob, res_foreign]:
        yd = res["youden_cutoff"]
        f1 = res["fpr1_cutoff"]
        yd_str = f"Thr>={yd['threshold']:.2f} (R:{yd['recall']*100:.1f}%, FPR:{yd['fpr']*100:.1f}%)" if yd else "N/A"
        f1_str = f"Thr>={f1['threshold']:.2f} (R:{f1['recall']*100:.1f}%, FPR:{f1['fpr']*100:.1f}%)" if f1 else "N/A"
        print(f"{res['name']:<35} | {yd_str:<26} | {f1_str:<24}")

    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
