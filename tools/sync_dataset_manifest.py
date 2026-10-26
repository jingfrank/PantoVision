#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/sync_dataset_manifest.py
数据集目录实时扫描与 Manifest 同步工具

当人工对 /home/ibd/jyb/Dataset 目录进行删改、移动、重新分类后，
运行此脚本即可自动遍历最新目录结构，同步更新 manifest.json 与 manifest.csv。

目录结构识别规范：
- Normal/ 及其任意子文件夹下的图像 -> 统一标记为 category="Normal", is_anomaly=False
- Anomaly/<fault_type>/ 下的图像   -> 统一标记为 category="Anomaly", is_anomaly=True, fault_type=<fault_type>
"""

import os
import sys
import json
import csv
import argparse
import cv2
from typing import Dict, List, Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sync_manifest(dataset_dir: str):
    dataset_dir = os.path.abspath(dataset_dir)
    if not os.path.exists(dataset_dir):
        print(f"[ERROR] 目录不存在: {dataset_dir}")
        return

    manifest: List[Dict[str, Any]] = []
    item_id = 1

    normal_dir = os.path.join(dataset_dir, "Normal")
    anomaly_dir = os.path.join(dataset_dir, "Anomaly")

    # 1. 扫描 Normal 目录
    if os.path.exists(normal_dir):
        for root, dirs, files in os.walk(normal_dir):
            for file in sorted(files):
                ext = os.path.splitext(file)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, dataset_dir)
                    # 推断子类
                    sub_rel = os.path.relpath(root, normal_dir)
                    subcat = "normal_clean" if sub_rel == "." else sub_rel.replace(os.sep, "_")

                    try:
                        im = cv2.imread(abs_path)
                        h, w = im.shape[:2] if im is not None else [0, 0]
                    except Exception:
                        h, w = 0, 0

                    manifest.append({
                        "id": item_id,
                        "relative_path": rel_path,
                        "category": "Normal",
                        "subcategory": subcat,
                        "is_anomaly": False,
                        "fault_type": "normal",
                        "resolution": [w, h],
                        "filename": file,
                    })
                    item_id += 1

    # 2. 扫描 Anomaly 目录
    if os.path.exists(anomaly_dir):
        for item in sorted(os.listdir(anomaly_dir)):
            sub_path = os.path.join(anomaly_dir, item)
            if os.path.isdir(sub_path):
                fault_type = item
                for root, dirs, files in os.walk(sub_path):
                    for file in sorted(files):
                        ext = os.path.splitext(file)[1].lower()
                        if ext in IMAGE_EXTENSIONS:
                            abs_path = os.path.join(root, file)
                            rel_path = os.path.relpath(abs_path, dataset_dir)

                            try:
                                im = cv2.imread(abs_path)
                                h, w = im.shape[:2] if im is not None else [0, 0]
                            except Exception:
                                h, w = 0, 0

                            manifest.append({
                                "id": item_id,
                                "relative_path": rel_path,
                                "category": "Anomaly",
                                "subcategory": fault_type,
                                "is_anomaly": True,
                                "fault_type": fault_type,
                                "resolution": [w, h],
                                "filename": file,
                            })
                            item_id += 1

    # 3. 输出 manifest.json
    json_path = os.path.join(dataset_dir, "manifest.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 4. 输出 manifest.csv
    csv_path = os.path.join(dataset_dir, "manifest.csv")
    csv_headers = ["id", "relative_path", "category", "subcategory", "is_anomaly", "fault_type", "resolution", "filename"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for m in manifest:
            writer.writerow({
                "id": m["id"],
                "relative_path": m["relative_path"],
                "category": m["category"],
                "subcategory": m["subcategory"],
                "is_anomaly": m["is_anomaly"],
                "fault_type": m["fault_type"],
                "resolution": f"{m['resolution'][0]}x{m['resolution'][1]}",
                "filename": m["filename"],
            })

    # 5. 打印统计信息
    normal_cnt = sum(1 for m in manifest if not m["is_anomaly"])
    anomaly_cnt = sum(1 for m in manifest if m["is_anomaly"])

    print("=" * 60)
    print("      数据集目录结构同步完成 (Manifest Synced)")
    print("=" * 60)
    print(f"数据集根目录: {dataset_dir}")
    print(f"总样本数: {len(manifest)} (Normal 负样本: {normal_cnt} 帧, Anomaly 正样本: {anomaly_cnt} 帧)")
    print(f"索引文件已更新: \n  - {json_path}\n  - {csv_path}")

    from collections import Counter
    sub_counts = Counter([m["subcategory"] for m in manifest])
    print("-" * 60)
    for sub, count in sub_counts.items():
        print(f"  * {sub:25s}: {count:4d} 帧")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="扫描并同步 Dataset 目录为最新 Manifest")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/home/ibd/jyb/Dataset",
        help="数据集根目录",
    )
    args = parser.parse_args()
    sync_manifest(args.dataset_dir)
