#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_false_alarm_manifest.py — 误报视频数据集 Manifest 自动建档工具

扫描 <input_dir>/类别/车型/视频.MP4 三层目录结构（或任意层级视频文件），
生成标准化数据集清单 (JSON 和 CSV 格式)。

用法:
    python tools/build_false_alarm_manifest.py \
        --input-dir "/media/ibd/USB_DISK/典型故障视频/误报视频" \
        --output "local_cache/manifest_false_alarms.json"
"""

import os
import sys
import json
import csv
import argparse
import pathlib
from typing import Dict, List, Any

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".flv"}

TRAIN_FAMILY_MAP = {
    "CR400AF": "CR400AF",
    "CR400-AF": "CR400AF",
    "CR400BF": "CR400BF",
    "CR400-BF": "CR400BF",
    "CR400FS": "CR400FS",
    "CR400-FS": "CR400FS",
    "CRH380A": "CRH380A",
    "CRH380B": "CRH380B",
    "CRH380D": "CRH380D",
}


def infer_train_family(model_name: str) -> str:
    """根据车型或文件夹名称推断动车组家族"""
    m_upper = model_name.upper().replace("_", "-")
    for key, val in TRAIN_FAMILY_MAP.items():
        if key.upper().replace("_", "-") in m_upper:
            return val
    return "UNKNOWN"


def build_manifest(input_dir: str, output_path: str) -> List[Dict[str, Any]]:
    """递归遍历指定目录，构建视频元数据清单"""
    root = pathlib.Path(input_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    records: List[Dict[str, Any]] = []
    vid_id = 1

    # 递归查找所有匹配的视频文件
    all_video_files = sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    )

    for p in all_video_files:
        rel_parts = p.relative_to(root).parts
        # 根据相对路径深度解析类别与车型
        if len(rel_parts) >= 3:
            category = rel_parts[0]
            train_model = rel_parts[1]
        elif len(rel_parts) == 2:
            category = rel_parts[0]
            train_model = "UNKNOWN"
        else:
            category = "DEFAULT"
            train_model = "UNKNOWN"

        train_family = infer_train_family(train_model)
        if train_family == "UNKNOWN" and category != "DEFAULT":
            train_family = infer_train_family(category)

        rec = {
            "video_id": f"FA_{vid_id:04d}",
            "file_path": str(p),
            "filename": p.name,
            "category": category,
            "train_model": train_model,
            "train_family": train_family,
            "ground_truth": "NORMAL",
            "review_status": "AUTO_ASSIGNED",
            "our_alarm": None,
            "our_fault_types": [],
            "notes": f"客户误报样本 - {category}",
        }
        records.append(rec)
        vid_id += 1

    if not records:
        print(f"[WARN] 目录 {input_dir} 下未找到任何视频文件！")
        return []

    # 写入 JSON
    out_json = pathlib.Path(output_path).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # 写入 CSV
    out_csv = out_json.with_suffix(".csv")
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"[OK] 成功建档 {len(records)} 个视频:")
    print(f"     JSON清单: {out_json}")
    print(f"     CSV清单:  {out_csv}")

    # 统计分类与车型分布
    cat_counts: Dict[str, int] = {}
    family_counts: Dict[str, int] = {}
    for r in records:
        c = r["category"]
        fam = r["train_family"]
        cat_counts[c] = cat_counts.get(c, 0) + 1
        family_counts[fam] = family_counts.get(fam, 0) + 1

    print("\n--- 类别分布明细 ---")
    for c, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {c:<30}: {cnt} 个视频")

    print("\n--- 车型家族分布 ---")
    for fam, cnt in sorted(family_counts.items(), key=lambda x: -x[1]):
        print(f"  {fam:<15}: {cnt} 个视频")

    return records


def main():
    parser = argparse.ArgumentParser(description="误报视频数据集 Manifest 自动建档工具")
    parser.add_argument("--input-dir", required=True, type=str, help="误报视频根目录路径")
    parser.add_argument("--output", required=True, type=str, help="输出 Manifest JSON 文件路径 (同时生成同名 .csv)")
    args = parser.parse_args()

    build_manifest(args.input_dir, args.output)


if __name__ == "__main__":
    main()
