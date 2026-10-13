#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_false_alarm_manifest.py — 误报视频数据集 Manifest 自动建档工具 (重构规范版)

扫描 <input_dir>/客户报错类型/车型/视频.MP4 三层目录结构（或任意层级视频文件），
生成标准化数据集清单 (JSON 和 CSV 格式)。

核心升级：
1. 将原 category 字段正名为 customer_reported_fault（客户原算法报警结果/报错标签），消除语义歧义；
2. ground_truth 初始化为 PENDING_VERIFY（待验证状态），严谨区分待验证样本与已确认真值；
3. 增加 actual_fault_type、review_status、verified_by、verification_basis 等车规级复核字段；
4. CSV 格式优化（支持 Excel 直接打开且兼容列表字段解析）。

用法:
    python tools/build_false_alarm_manifest.py \
        --input-dir "/media/ibd/USB_DISK/典型故障视频/误报视频" \
        --output "/media/ibd/USB_DISK/manifest_false_alarms.json"
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
    """递归遍历指定目录，构建标准化视频元数据清单"""
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
        # 根据相对路径深度解析客户报错标签与车型
        if len(rel_parts) >= 3:
            customer_fault = rel_parts[0]
            train_model = rel_parts[1]
        elif len(rel_parts) == 2:
            customer_fault = rel_parts[0]
            train_model = "UNKNOWN"
        else:
            customer_fault = "DEFAULT"
            train_model = "UNKNOWN"

        train_family = infer_train_family(train_model)
        if train_family == "UNKNOWN" and customer_fault != "DEFAULT":
            train_family = infer_train_family(customer_fault)

        rec = {
            "video_id": f"FA_{vid_id:04d}",
            "file_path": str(p),
            "filename": p.name,
            "customer_reported_fault": customer_fault,  # 客户原系统报警结果 / 报错标签
            "category": customer_fault,                 # 兼容旧代码字段别名
            "train_model": train_model,
            "train_family": train_family,
            "ground_truth": "PENDING_VERIFY",           # [PENDING_VERIFY / NORMAL / TRUE_FAULT / ENVIRONMENT_LIMIT]
            "actual_fault_type": "",                    # 若确认为真故障: [structural / foreign_object / dirt / spark / tilt / float]
            "review_status": "UNVERIFIED",              # [UNVERIFIED / AUTO_SCREENED / MANUALLY_VERIFIED]
            "verified_by": "",                          # 验证人 (人工复核后填写)
            "verification_basis": "",                   # 判定依据说明 (如: "碳板反光无形变", "导线挂细绳")
            "our_alarm": None,                          # 我方算法是否触发告警 [True / False]
            "our_fault_types": [],                      # 我方检出故障项列表
            "notes": f"客户原算法报错标签: {customer_fault}",
        }
        records.append(rec)
        vid_id += 1

    if not records:
        print(f"[WARN] 目录 {input_dir} 下未找到任何视频文件！")
        return []

    # 1. 写入标准化 JSON
    out_json = pathlib.Path(output_path).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # 2. 写入标准化 CSV (方便在 Excel 中直接打开与人工标记)
    out_csv = out_json.with_suffix(".csv")
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = list(records[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            r_copy = dict(r)
            if isinstance(r_copy.get("our_fault_types"), list):
                r_copy["our_fault_types"] = ";".join(r_copy["our_fault_types"])
            writer.writerow(r_copy)

    print(f"\n========================================================")
    print(f" [OK] 成功重构建档 {len(records)} 个视频清单:")
    print(f"   ▶ JSON清单: {out_json}")
    print(f"   ▶ CSV清单:  {out_csv} (支持 Excel 直接打开与编辑)")
    print(f"========================================================")

    # 统计客户报错分类与车型分布
    fault_counts: Dict[str, int] = {}
    family_counts: Dict[str, int] = {}
    for r in records:
        cf = r["customer_reported_fault"]
        fam = r["train_family"]
        fault_counts[cf] = fault_counts.get(cf, 0) + 1
        family_counts[fam] = family_counts.get(fam, 0) + 1

    print("\n--- 客户原算法报错标签分布明细 ---")
    for cf, cnt in sorted(fault_counts.items(), key=lambda x: -x[1]):
        print(f"  {cf:<30}: {cnt} 个视频")

    print("\n--- 车型家族分布明细 ---")
    for fam, cnt in sorted(family_counts.items(), key=lambda x: -x[1]):
        print(f"  {fam:<15}: {cnt} 个视频")

    return records


def main():
    parser = argparse.ArgumentParser(description="误报视频数据集 Manifest 自动建档工具 (重构规范版)")
    parser.add_argument("--input-dir", required=True, type=str, help="误报视频根目录路径")
    parser.add_argument("--output", required=True, type=str, help="输出 Manifest JSON 文件路径 (同时生成同名 .csv)")
    args = parser.parse_args()

    build_manifest(args.input_dir, args.output)


if __name__ == "__main__":
    main()
