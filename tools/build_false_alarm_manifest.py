#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_false_alarm_manifest.py — 误报视频数据集 Manifest 自动建档与标准表格生成工具

功能说明：
1. 递归扫描指定目录下的工况视频（如 <input_dir>/客户报错类型/车型/视频.MP4）；
2. 提取核心元数据：编号 (FA_xxxx)、文件路径、文件名、车型、车型家族、客户原报错标签；
3. ground_truth 设置为 Excel 原生数据有效性下拉选择框 (DataValidation)，方便人工离线复核直接点选；
4. 保留 our_alarm (我方算法是否报警) 与 our_fault_types (我方检出故障类型)，供离线基准评测引擎回填比对；
5. 自动同步输出三种格式文件：
   - .xlsx: 包含美化表头、冻结首行、自动筛选以及 ground_truth 下拉菜单
   - .csv:  带 UTF-8 BOM，方便通用脚本与文本工具快速查看
   - .json: 标准元数据，供自动化调度与测试套件直接反序列化
6. 支持保护性增量扫描：若历史清单中已有人工填写的真值或评测结果，自动保留，防止误覆盖。

下拉真值选项定义：
- 待验证 (默认初始状态)
- 正常(客户误报)
- 受电弓结构异常
- 悬挂异物
- 大火花
- 画面脏污
- 受电弓倾斜
- 异常飘弓
- 环境受限/物理失真

用法示例:
    python tools/build_false_alarm_manifest.py \\
        --input-dir "/media/ibd/USB_DISK/典型故障视频/误报视频" \\
        --output "/media/ibd/USB_DISK/manifest_false_alarms.xlsx"
"""

import os
import sys
import json
import csv
import argparse
import pathlib
from typing import Dict, List, Any, Optional

try:
    import openpyxl
    from openpyxl.worksheet.datavalidation import DataValidation
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".flv"}

# 车型家族推断映射字典
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

# 人工复核真值下拉选择标准枚举
GROUND_TRUTH_OPTIONS = [
    "待验证",
    "正常(客户误报)",
    "受电弓结构异常",
    "悬挂异物",
    "大火花",
    "画面脏污",
    "受电弓倾斜",
    "异常飘弓",
    "环境受限/物理失真",
]

# 统一字段定义与列顺序
MANIFEST_HEADERS = [
    "video_id",
    "file_path",
    "filename",
    "train_model",
    "train_family",
    "customer_reported_fault",
    "ground_truth",
    "our_alarm",
    "our_fault_types",
    "notes",
]


def infer_train_family(model_name: str) -> str:
    """根据车型或路径目录名推断动车组家族"""
    m_upper = model_name.upper().replace("_", "-")
    for key, val in TRAIN_FAMILY_MAP.items():
        if key.upper().replace("_", "-") in m_upper:
            return val
    return "UNKNOWN"


def load_existing_annotations(base_path: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    """
    检查并读取既有的清单文件（优先 JSON），防止重新建档时覆盖既有人工标注
    """
    history_map: Dict[str, Dict[str, Any]] = {}
    json_path = base_path.with_suffix(".json")
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                items = json.load(f)
                for it in items:
                    key = it.get("file_path") or it.get("filename")
                    if key:
                        history_map[key] = it
                        if it.get("filename"):
                            history_map[it["filename"]] = it
            return history_map
        except Exception as e:
            print(f"[WARN] 读取既有 JSON 失败: {e}")

    csv_path = base_path.with_suffix(".csv")
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for it in reader:
                    key = it.get("file_path") or it.get("filename")
                    if key:
                        history_map[key] = it
                        if it.get("filename"):
                            history_map[it["filename"]] = it
            return history_map
        except Exception as e:
            print(f"[WARN] 读取既有 CSV 失败: {e}")

    return history_map


def normalize_historical_gt(val: Any) -> str:
    """标准化既有历史真值选项（兼容早期英文标记）"""
    if not val or val == "PENDING_VERIFY":
        return "待验证"
    if val == "NORMAL":
        return "正常(客户误报)"
    if val == "ENVIRONMENT_LIMIT":
        return "环境受限/物理失真"
    if str(val) in GROUND_TRUTH_OPTIONS:
        return str(val)
    return "待验证"


def export_to_excel(records: List[Dict[str, Any]], xlsx_path: pathlib.Path):
    """
    使用 openpyxl 导出具有数据有效性下拉选择框的美化 Excel 文件
    """
    if not OPENPYXL_AVAILABLE:
        print("[ERROR] 未检测到 openpyxl 库，跳过 .xlsx 导出！请安装 openpyxl。")
        return

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "误报视频复核清单"

    # 1. 写入表头
    ws.append(MANIFEST_HEADERS)

    # 样式定义
    header_font = Font(name="微软雅黑", size=11, bold=True, color="FFFFFF")
    header_fill_default = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")  # 经典科技蓝
    header_fill_gt = PatternFill(start_color="C55A11", end_color="C55A11", fill_type="solid")       # 暖橙色高亮真值列

    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )

    for col_idx, h in enumerate(MANIFEST_HEADERS, 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill_gt if h == "ground_truth" else header_fill_default
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border

    ws.row_dimensions[1].height = 28

    # 2. 写入数据行
    gt_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")  # 柔和浅黄提醒可编辑
    gt_font = Font(name="微软雅黑", size=10, bold=True, color="806000")
    reg_font = Font(name="微软雅黑", size=10)
    center_cols = {"video_id", "train_model", "train_family", "ground_truth", "our_alarm"}

    for r_idx, r in enumerate(records, 2):
        row_vals = [
            r.get("video_id", ""),
            r.get("file_path", ""),
            r.get("filename", ""),
            r.get("train_model", ""),
            r.get("train_family", ""),
            r.get("customer_reported_fault", ""),
            r.get("ground_truth", "待验证"),
            "" if r.get("our_alarm") is None else str(r.get("our_alarm")),
            ";".join(r.get("our_fault_types")) if isinstance(r.get("our_fault_types"), list) else str(r.get("our_fault_types", "")),
            r.get("notes", ""),
        ]
        ws.append(row_vals)
        ws.row_dimensions[r_idx].height = 20

        for col_idx, h in enumerate(MANIFEST_HEADERS, 1):
            cell = ws.cell(row=r_idx, column=col_idx)
            cell.border = thin_border
            align_h = "center" if h in center_cols else "left"
            cell.alignment = Alignment(horizontal=align_h, vertical="center")

            if h == "ground_truth":
                cell.fill = gt_fill
                cell.font = gt_font
            else:
                cell.font = reg_font

    # 3. 添加 ground_truth 列的数据有效性下拉菜单 (DataValidation)
    gt_col_letter = "G"
    max_row = max(1000, len(records) + 200)
    formula_str = f'"{",".join(GROUND_TRUTH_OPTIONS)}"'
    dv = DataValidation(type="list", formula1=formula_str, allow_blank=True)
    dv.promptTitle = "真值选择"
    dv.prompt = "请点击下拉箭头选择人工复核真值"
    dv.errorTitle = "输入无效"
    dv.error = "请从预设的下拉选项中选择规范的故障真值分类！"

    ws.add_data_validation(dv)
    dv.add(f"{gt_col_letter}2:{gt_col_letter}{max_row}")

    # 4. 冻结首行与开启自动筛选
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(MANIFEST_HEADERS))}{len(records) + 1}"

    # 5. 列宽优化
    col_widths = {
        "video_id": 12,
        "file_path": 42,
        "filename": 35,
        "train_model": 20,
        "train_family": 14,
        "customer_reported_fault": 28,
        "ground_truth": 22,
        "our_alarm": 14,
        "our_fault_types": 24,
        "notes": 35,
    }
    for col_idx, h in enumerate(MANIFEST_HEADERS, 1):
        c_letter = get_column_letter(col_idx)
        ws.column_dimensions[c_letter].width = col_widths.get(h, 18)

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(xlsx_path)


def export_to_csv(records: List[Dict[str, Any]], csv_path: pathlib.Path):
    """导出标准 UTF-8 BOM CSV 文件 (Excel 双击直接打开不乱码)"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_HEADERS)
        writer.writeheader()
        for r in records:
            r_copy = {k: r.get(k, "") for k in MANIFEST_HEADERS}
            if r_copy["our_alarm"] is None:
                r_copy["our_alarm"] = ""
            if isinstance(r_copy["our_fault_types"], list):
                r_copy["our_fault_types"] = ";".join(r_copy["our_fault_types"])
            writer.writerow(r_copy)


def export_to_json(records: List[Dict[str, Any]], json_path: pathlib.Path):
    """导出标准 JSON 文件 (方便系统测试套件直接读取)"""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def build_manifest(input_dir: str, output_path: str, force_reset: bool = False) -> List[Dict[str, Any]]:
    """递归遍历指定目录，构建/更新标准化视频清单"""
    root = pathlib.Path(input_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    out_file = pathlib.Path(output_path).resolve()
    base_stem = out_file.parent / out_file.stem

    existing_history = {} if force_reset else load_existing_annotations(base_stem)
    if existing_history:
        print(f"[INFO] 已检测到既有清单，成功加载 {len(existing_history)} 条历史标注/评测记录进行增量合并")

    records: List[Dict[str, Any]] = []
    vid_id = 1

    # 递归查找所有匹配的视频文件
    all_video_files = sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    )

    for p in all_video_files:
        rel_parts = p.relative_to(root).parts
        # 解析路径深度获取客户报错标签与车型
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

        # 检查历史记录中是否已有标注或测试数据
        hist = existing_history.get(str(p)) or existing_history.get(p.name)

        if hist:
            gt_val = normalize_historical_gt(hist.get("ground_truth"))
            alarm_val = hist.get("our_alarm")
            if isinstance(alarm_val, str):
                if alarm_val.lower() == "true":
                    alarm_val = True
                elif alarm_val.lower() == "false":
                    alarm_val = False
                elif alarm_val == "":
                    alarm_val = None
            f_types_val = hist.get("our_fault_types", [])
            if isinstance(f_types_val, str):
                f_types_val = [x.strip() for x in f_types_val.split(";") if x.strip()]
            notes_val = hist.get("notes") or f"客户原算法报错标签: {customer_fault}"
        else:
            gt_val = "待验证"
            alarm_val = None
            f_types_val = []
            notes_val = f"客户原算法报错标签: {customer_fault}"

        rec = {
            "video_id": f"FA_{vid_id:04d}",
            "file_path": str(p),
            "filename": p.name,
            "train_model": train_model,
            "train_family": train_family,
            "customer_reported_fault": customer_fault,
            "ground_truth": gt_val,
            "our_alarm": alarm_val,
            "our_fault_types": f_types_val,
            "notes": notes_val,
        }
        records.append(rec)
        vid_id += 1

    if not records:
        print(f"[WARN] 目录 {input_dir} 下未找到任何视频文件！")
        return []

    # 同步生成三种格式
    out_xlsx = base_stem.with_suffix(".xlsx")
    out_csv = base_stem.with_suffix(".csv")
    out_json = base_stem.with_suffix(".json")

    export_to_excel(records, out_xlsx)
    export_to_csv(records, out_csv)
    export_to_json(records, out_json)

    print(f"\n========================================================")
    print(f" [OK] 成功生成/同步 {len(records)} 个误报视频清单:")
    print(f"   ▶ Excel表格: {out_xlsx} (含 ground_truth 原生下拉框)")
    print(f"   ▶ CSV清单:   {out_csv} (UTF-8 BOM)")
    print(f"   ▶ JSON清单:  {out_json} (程序自动化读取)")
    print(f"========================================================")

    # 统计客户报错分类与车型分布
    fault_counts: Dict[str, int] = {}
    family_counts: Dict[str, int] = {}
    gt_counts: Dict[str, int] = {}
    for r in records:
        cf = r["customer_reported_fault"]
        fam = r["train_family"]
        gt = r["ground_truth"]
        fault_counts[cf] = fault_counts.get(cf, 0) + 1
        family_counts[fam] = family_counts.get(fam, 0) + 1
        gt_counts[gt] = gt_counts.get(gt, 0) + 1

    print("\n--- 客户原算法报错标签分布 ---")
    for cf, cnt in sorted(fault_counts.items(), key=lambda x: -x[1]):
        print(f"  {cf:<30}: {cnt} 个视频")

    print("\n--- 车型家族分布 ---")
    for fam, cnt in sorted(family_counts.items(), key=lambda x: -x[1]):
        print(f"  {fam:<15}: {cnt} 个视频")

    print("\n--- 当前人工复核真值 (ground_truth) 分布 ---")
    for gt, cnt in sorted(gt_counts.items(), key=lambda x: -x[1]):
        print(f"  {gt:<20}: {cnt} 个视频")

    print("\n💡 操作指引: 使用 WPS 或 Microsoft Excel 打开 manifest_false_alarms.xlsx，")
    print("   点击【ground_truth】列即可直接通过下拉菜单选择真实故障分类。")

    return records


def main():
    parser = argparse.ArgumentParser(description="误报视频数据集 Manifest 自动建档与表格生成工具 (升级版)")
    parser.add_argument("--input-dir", required=True, type=str, help="误报视频根目录路径")
    parser.add_argument("--output", required=True, type=str, help="输出 Manifest 路径 (支持 .xlsx / .csv / .json)")
    parser.add_argument("--force", action="store_true", help="强制重置所有真值和评测状态为初始值")
    args = parser.parse_args()

    build_manifest(args.input_dir, args.output, force_reset=args.force)


if __name__ == "__main__":
    main()
