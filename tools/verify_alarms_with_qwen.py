#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_alarms_with_qwen.py — 使用 Qwen3-VL-4B 对 benchmark_results_v2 告警图片进行深度核验

功能说明：
1. 全量扫描 /mnt/T4/jyb/gongwang_clean/local_cache/benchmark_results_v2 路径下的告警抓拍图片；
2. 重点核验【大火花】(192张) 与【受电弓结构异常】(112张)，并辅以脏污与异物对比样本；
3. 使用 Qwen3-VL-4B 真实多模态大模型进行多图 Batch 并发独立研判；
4. 甄别出真实故障与算法误报（如强光照反射、站台照明、正常机械接头等）；
5. 实时追加输出 CSV 清单与结构化统计报表。
"""

import os
import sys
import glob
import time
import json
import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import torch
from PIL import Image

try:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
except ImportError:
    print("[ERROR] 请确保在已安装 transformers 的环境中运行！")
    sys.exit(1)

DEFAULT_MODEL_PATH = "/mnt/T4/jyb/gongwang/models/Qwen3-VL-4B-Instruct"
DEFAULT_INPUT_DIR = "/mnt/T4/jyb/gongwang_clean/local_cache/benchmark_results_v2"
DEFAULT_OUTPUT_CSV = "/mnt/T4/jyb/gongwang_clean/local_cache/qwen_verification_results.csv"
DEFAULT_OUTPUT_JSON = "/mnt/T4/jyb/gongwang_clean/local_cache/qwen_verification_results.json"

PROMPT = """你是一个专业的铁路受电弓与接触网故障监测及智能复核安全专家。
请仔细观察图像中受电弓滑板、弓角、支架结构与接触网接触状态。
【注意】：图像上可能叠加有工控系统的初检标记框或顶部告警文字。请你作为独立复核专家，严格基于受电弓本身的物理形态进行真实性研判，甄别出算法可能存在的误报：

1. 【大火花(spark)】:
   - 真实火花判定标准：受电弓碳滑板与接触线之间出现强烈的放电电弧、大面积刺眼白光或蓝色爆燃辉光。
   - 排除误报约束：外界站台照明大灯、夜间路灯、反光镜、车顶金属高反光拉丝绝不是火花！正常运行微弱蓝白小星火不算大火花。

2. 【受电弓结构异常(structural_anomaly)】:
   - 真实结构异常标准：弓角(羊角)明显折断缺失、碳滑板严重剥落碎裂变形、上导杆断脱、金属支架框架扭曲断裂。
   - 排除误报约束：受电弓正常的金属螺栓、固定线夹、卡扣、接头属于标准机械部件，绝不是异常或断裂！因阳光斜射产生的局部阴影或反光亮斑绝不是结构异常！

请严格按以下 JSON 格式输出判定结果，不要输出任何多余废话：
```json
{
  "reasoning": "50字以内的专业视觉研判依据",
  "spark": 0或1,
  "structural_anomaly": 0或1
}
```
"""


def parse_alarm_type(filename: str) -> str:
    """从文件名解析原始初检告警类别"""
    base = os.path.basename(filename).replace(".jpg", "").replace(".jpeg", "")
    parts = base.split("_")
    if len(parts) >= 4:
        return "_".join(parts[3:])
    return "unknown"


def parse_qwen_output(out_text: str):
    """解析 Qwen 输出的 JSON"""
    json_match = re.search(r'\{[^{}]*"reasoning"[^{}]*\}', out_text, re.DOTALL)
    if not json_match:
        json_match = re.search(r'\{[^{}]*"spark"[^{}]*\}', out_text, re.DOTALL)

    reasoning = ""
    spark = 0
    structural_anomaly = 0

    if json_match:
        try:
            data = json.loads(json_match.group(0))
            reasoning = str(data.get("reasoning", "")).strip()
            spark = 1 if int(data.get("spark", 0)) > 0 else 0
            structural_anomaly = 1 if int(data.get("structural_anomaly", 0)) > 0 else 0
            return reasoning, spark, structural_anomaly
        except Exception:
            pass

    # 兜底正则提取
    if re.search(r'"spark"\s*:\s*1', out_text):
        spark = 1
    if re.search(r'"structural_anomaly"\s*:\s*1', out_text):
        structural_anomaly = 1

    reason_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', out_text)
    if reason_match:
        reasoning = reason_match.group(1)
    else:
        reasoning = out_text[:120].replace("\n", " ")

    return reasoning, spark, structural_anomaly


def determine_verdict(orig_type: str, spark: int, struct: int) -> str:
    """综合研判初检与复核结论"""
    if orig_type == "big_spark":
        if spark == 1:
            return "确认大火花 (True Spark Alarm)"
        else:
            return "大火花虚警已剔除 (Suppressed Spark False Alarm)"
    elif orig_type == "structure_abnormal":
        if struct == 1:
            return "确认结构异常 (True Struct Alarm)"
        else:
            return "结构异常虚警已剔除 (Suppressed Struct False Alarm)"
    else:
        # 对照组样本 (如 dirty, foreign_object)
        detected = []
        if spark == 1:
            detected.append("大火花")
        if struct == 1:
            detected.append("结构异常")
        if detected:
            return f"交叉检出: {';'.join(detected)}"
        return f"对照组核实无火花/结构异常 ({orig_type})"


def main():
    parser = argparse.ArgumentParser(description="使用 Qwen3-VL 对告警图片进行深度核验")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=0, help="限制测试总数 (0 为全量测试)")
    args = parser.parse_args()

    print("=" * 80)
    print("🚀 启动 Qwen3-VL-4B 受电弓故障报警核验评测引擎")
    print(f"  • 模型权重目录: {args.model_path}")
    print(f"  • 告警图片目录: {args.input_dir}")
    print(f"  • 输出 CSV 路径: {args.output_csv}")
    print("=" * 80)

    # 1. 扫描与筛选图片
    all_imgs = sorted(glob.glob(os.path.join(args.input_dir, "*", "*.jpg")))
    spark_imgs = [p for p in all_imgs if parse_alarm_type(p) == "big_spark"]
    struct_imgs = [p for p in all_imgs if parse_alarm_type(p) == "structure_abnormal"]
    dirty_imgs = [p for p in all_imgs if parse_alarm_type(p) == "dirty"][:20]
    foreign_imgs = [p for p in all_imgs if parse_alarm_type(p) == "foreign_object"][:10]
    tilt_imgs = [p for p in all_imgs if parse_alarm_type(p) == "tilt"]

    selected_imgs = spark_imgs + struct_imgs + dirty_imgs + foreign_imgs + tilt_imgs
    if args.max_images > 0:
        selected_imgs = selected_imgs[:args.max_images]

    print(f"📊 待核验图片总数: {len(selected_imgs)} 张")
    print(f"   ├─ 大火花 (big_spark) 告警抓拍: {len(spark_imgs)} 张 (全量覆盖)")
    print(f"   ├─ 结构异常 (structure_abnormal) 告警抓拍: {len(struct_imgs)} 张 (全量覆盖)")
    print(f"   ├─ 画面脏污 (dirty) 对照抽样: {len(dirty_imgs)} 张")
    print(f"   ├─ 悬挂异物 (foreign_object) 对照抽样: {len(foreign_imgs)} 张")
    print(f"   └─ 受电弓倾斜 (tilt) 对照抓拍: {len(tilt_imgs)} 张")

    # 2. 加载 Qwen3-VL-4B 模型
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.80, device=0)

    print("\n⏳ 正在向 GPU (TITAN RTX) 加载 Qwen3-VL-4B-Instruct (FP16)...")
    t_load = time.time()
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.float16,
        device_map="cuda:0",
        low_cpu_mem_usage=True
    ).eval()
    print(f"✅ 模型加载成功！耗时: {time.time() - t_load:.2f}s\n")

    # 3. 初始化 CSV 实时写入
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    csv_file = open(args.output_csv, "w", encoding="utf-8-sig", newline="")
    csv_fields = [
        "index", "image_path", "filename", "video_stem", "original_alarm",
        "qwen_spark", "qwen_structural_anomaly", "verdict", "reasoning", "latency_s"
    ]
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()
    csv_file.flush()

    # 4. 批处理推理
    all_records = []
    total_imgs = len(selected_imgs)
    batch_size = args.batch_size
    t_start_all = time.time()

    for b_idx in range(0, total_imgs, batch_size):
        b_paths = selected_imgs[b_idx: b_idx + batch_size]
        pil_imgs = []
        valid_paths = []

        for p in b_paths:
            try:
                img = Image.open(p).convert("RGB")
                pil_imgs.append(img)
                valid_paths.append(p)
            except Exception as e:
                print(f"[WARN] 无法打开图片 {p}: {e}")

        if not pil_imgs:
            continue

        messages = [
            [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
            for img in pil_imgs
        ]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]

        t0 = time.time()
        inputs = processor(text=texts, images=pil_imgs, padding=True, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=96, do_sample=False)
        outputs = processor.batch_decode(gen_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        dt_batch = time.time() - t0
        dt_per_img = dt_batch / len(pil_imgs)

        for i, (p, out_str) in enumerate(zip(valid_paths, outputs)):
            idx_num = b_idx + i + 1
            orig_alarm = parse_alarm_type(p)
            reasoning, spk, struct = parse_qwen_output(out_str)
            verdict = determine_verdict(orig_alarm, spk, struct)
            video_stem = os.path.basename(os.path.dirname(p))

            rec = {
                "index": idx_num,
                "image_path": p,
                "filename": os.path.basename(p),
                "video_stem": video_stem,
                "original_alarm": orig_alarm,
                "qwen_spark": spk,
                "qwen_structural_anomaly": struct,
                "verdict": verdict,
                "reasoning": reasoning,
                "latency_s": round(dt_per_img, 2),
            }
            all_records.append(rec)
            csv_writer.writerow(rec)

        csv_file.flush()
        print(f"[{b_idx + len(pil_imgs)}/{total_imgs}] 批推理耗时: {dt_batch:.2f}s ({dt_per_img:.2f}s/张) | 样例结论: {all_records[-1]['verdict']}")

    csv_file.close()
    total_time = time.time() - t_start_all

    # 5. 保存 JSON 结果
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    # 6. 统计汇总大盘
    spark_recs = [r for r in all_records if r["original_alarm"] == "big_spark"]
    struct_recs = [r for r in all_records if r["original_alarm"] == "structure_abnormal"]
    dirty_recs = [r for r in all_records if r["original_alarm"] == "dirty"]
    foreign_recs = [r for r in all_records if r["original_alarm"] == "foreign_object"]

    spark_conf = sum(1 for r in spark_recs if r["qwen_spark"] == 1)
    spark_supp = len(spark_recs) - spark_conf

    struct_conf = sum(1 for r in struct_recs if r["qwen_structural_anomaly"] == 1)
    struct_supp = len(struct_recs) - struct_conf

    print("\n" + "=" * 80)
    print("📈 Qwen3-VL-4B 核验结果统计汇总大盘")
    print("=" * 80)
    print(f"总计核验图片: {len(all_records)} 张 | 总耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    print("-" * 80)
    print(f"【大火花 (big_spark) 告警核验结果】 (共 {len(spark_recs)} 张):")
    print(f"  • 大模型确认为真实火花: {spark_conf} 张 ({(spark_conf/len(spark_recs)*100 if spark_recs else 0):.1f}%)")
    print(f"  • 大模型剔除为外界强光/反光虚警: {spark_supp} 张 ({(spark_supp/len(spark_recs)*100 if spark_recs else 0):.1f}%)")
    print("-" * 80)
    print(f"【受电弓结构异常 (structure_abnormal) 告警核验结果】 (共 {len(struct_recs)} 张):")
    print(f"  • 大模型确认为结构异常: {struct_conf} 张 ({(struct_conf/len(struct_recs)*100 if struct_recs else 0):.1f}%)")
    print(f"  • 大模型剔除为阴影/正常机械部件虚警: {struct_supp} 张 ({(struct_supp/len(struct_recs)*100 if struct_recs else 0):.1f}%)")
    print("-" * 80)
    print(f"【对照组样本核验 (dirty: {len(dirty_recs)}张, foreign: {len(foreign_recs)}张)】:")
    cross_spark = sum(1 for r in (dirty_recs + foreign_recs) if r["qwen_spark"] == 1)
    cross_struct = sum(1 for r in (dirty_recs + foreign_recs) if r["qwen_structural_anomaly"] == 1)
    print(f"  • 对照组被 Qwen 交叉报为火花: {cross_spark} 张")
    print(f"  • 对照组被 Qwen 交叉报为结构异常: {cross_struct} 张")
    print(f"  • 对照组保持正常无误判率: {((len(dirty_recs)+len(foreign_recs)-cross_spark-cross_struct)/(len(dirty_recs)+len(foreign_recs))*100 if (dirty_recs+foreign_recs) else 0):.1f}%")
    print("=" * 80)
    print(f"📄 逐图片详细核验报表已实时输出至: {args.output_csv}")
    print(f"💾 JSON 全量元数据已保存至: {args.output_json}")


if __name__ == "__main__":
    main()
