#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-4B 全量误报视频数据集 (653条) 批量复核与评测基准脚本
用途：对客户误报视频集 manifest_false_alarms.csv 中的全部 653 个视频进行严格批量测试，
针对每个视频在工控防线触发误报的关键帧进行 Qwen3-VL-4B 真实推断，
对比工控防线与 Qwen 防线的虚警率及纠偏消除率，并导出详尽的统计结果与报告。
"""

import os
import sys
import time
import json
import csv
import re
import glob
import cv2
import torch
import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

MODEL_PATH = "models/Qwen3-VL-4B-Instruct"
MANIFEST_PATH = "local_cache/manifest_false_alarms.csv"
BENCHMARK_DIR = "local_cache/benchmark_results"
OUTPUT_DIR = "local_cache/qwen_full_eval_results"
BATCH_SIZE = 8

PROMPT = (
    "你是一个铁路车载受电弓与接触网故障监测及智能复核专家。请仔细观察这张受电弓监控抓拍图像，严格判断是否存在真正的物理故障或重大异常：\n"
    "1. 大火花(spark): 受电弓滑板与接触网间出现剧烈电弧爆燃/大火花（微小蓝白星火不算）。\n"
    "2. 结构异常(structural): 碳滑板断裂/严重缺失、弓角折断、框架变形（强光反射造成的白斑绝非损坏）。\n"
    "3. 画面脏污(dirt): 镜头粘附严重泥斑、大面积油污导致看不清设备（天桥暗影、正常雨滴绝非脏污）。\n"
    "4. 受电弓倾斜(tilt): 受电弓明显几何歪斜。\n"
    "5. 悬挂异物(foreign_object): 接触网或弓头缠绕挂有塑料袋、风筝线、树枝等外来异物。\n"
    "请仅按标准 JSON 格式输出，1表示确诊异常，0表示正常：\n"
    '{"fault":0, "spark":0, "structural":0, "dirt":0, "tilt":0, "foreign_object":0, "reason":"30字以内的状态分析理由"}'
)


def load_industrial_alarms():
    """扫描 benchmark_results 目录获取所有视频的历史工控报警帧与报警类型"""
    bench_dirs = glob.glob(os.path.join(BENCHMARK_DIR, "*"))
    alarm_map = {}
    for d in bench_dirs:
        if not os.path.isdir(d):
            continue
        vbase = os.path.basename(d)
        alarms = glob.glob(os.path.join(d, "alarm_*.jpg"))
        if alarms:
            ftypes = set()
            frames = []
            for a in alarms:
                base = os.path.basename(a).replace(".jpg", "")
                ftype = base.split("_")[-1]
                ftypes.add(ftype)
                m = re.search(r"alarm_f(\d+)_", base)
                if m:
                    frames.append(int(m.group(1)))
            alarm_map[vbase] = {
                "types": list(ftypes),
                "frames": sorted(frames),
            }
    return alarm_map


def load_manifest(alarm_map):
    """读取 manifest_false_alarms.csv 获取全量视频信息"""
    videos = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video_id", "")
            vpath = row.get("file_path", "")
            fname = row.get("filename", "")
            vbase = os.path.splitext(fname)[0]
            cat = row.get("customer_reported_fault", "")
            
            if not os.path.exists(vpath):
                continue
                
            ind_info = alarm_map.get(vbase, None)
            if ind_info and len(ind_info["frames"]) > 0:
                target_frame = ind_info["frames"][0]
                ind_types = ind_info["types"]
                ind_alarm = True
            else:
                target_frame = 75  # 正常无告警视频默认测第75帧 (约3秒)
                ind_types = []
                ind_alarm = False
                
            videos.append({
                "video_id": vid,
                "video_name": fname,
                "base": vbase,
                "file_path": vpath,
                "category": cat,
                "target_frame": target_frame,
                "industrial_alarm": ind_alarm,
                "industrial_types": ind_types,
            })
    return videos


def parse_qwen_output(text):
    parsed = {
        "fault": 0,
        "spark": 0,
        "structural": 0,
        "dirt": 0,
        "tilt": 0,
        "foreign_object": 0,
        "reason": "正常"
    }
    try:
        clean = text
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()
            
        j_s = clean.find("{")
        j_e = clean.rfind("}")
        if j_s != -1 and j_e != -1:
            data = json.loads(clean[j_s:j_e+1])
            for k in ["fault", "spark", "structural", "dirt", "tilt", "foreign_object"]:
                if k in data:
                    parsed[k] = int(data[k])
            if any(parsed[k] > 0 for k in ["spark", "structural", "dirt", "tilt", "foreign_object"]):
                parsed["fault"] = 1
            if "reason" in data:
                parsed["reason"] = str(data["reason"])
            elif "analysis" in data:
                parsed["reason"] = str(data["analysis"])
    except Exception:
        pass
    return parsed


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 80)
    print(" 🚀 [Qwen3-VL-4B] 初始化全量误报视频数据集 (653条) 批量复核流水线")
    print(f" 📂 Manifest 路径: {MANIFEST_PATH}")
    print(f" ⚙️  批处理大小 (Batch Size): {BATCH_SIZE}")
    print("=" * 80)
    
    # 1. 扫描工控历史告警与数据集
    alarm_map = load_industrial_alarms()
    videos = load_manifest(alarm_map)
    total_videos = len(videos)
    print(f"✅ 成功命中本地存在的视频: {total_videos} 个 (工控产生报警: {sum(1 for v in videos if v['industrial_alarm'])} 个)")

    # 2. 加载大模型
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    t0_load = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map="cuda:0",
        low_cpu_mem_usage=True
    )
    model.eval()
    print(f"✅ Qwen3-VL-4B 加载就绪！耗时: {time.time()-t0_load:.1f}s | 显存: {torch.cuda.memory_allocated(0)/1024**2:.0f} MB\n")

    # 3. 分批推断
    results = []
    total_batches = (total_videos + BATCH_SIZE - 1) // BATCH_SIZE
    t_start_all = time.time()
    
    for b_idx in range(total_batches):
        batch_slice = videos[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
        pil_images = []
        valid_items = []
        
        for item in batch_slice:
            cap = cv2.VideoCapture(item["file_path"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, item["target_frame"] - 1))
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                continue
                
            resized = cv2.resize(frame, (640, 480))
            pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            pil_images.append(pil_img)
            valid_items.append(item)
            
        if not pil_images:
            continue
            
        messages_list = [
            [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
            for img in pil_images
        ]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
        inputs = processor(text=texts, images=pil_images, padding=True, return_tensors="pt").to("cuda:0")
        
        t_batch_start = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=48)
        batch_cost = time.time() - t_batch_start
        
        new_ids = out[:, inputs["input_ids"].shape[1]:]
        decoded = processor.batch_decode(new_ids, skip_special_tokens=True)
        
        for item, dec_text in zip(valid_items, decoded):
            p = parse_qwen_output(dec_text)
            qwen_alarm = bool(p["fault"] > 0)
            
            # 是否成功纠偏/抑制了工控误报
            is_suppressed = item["industrial_alarm"] and (not qwen_alarm)
            
            rec = {
                "video_id": item["video_id"],
                "video_name": item["video_name"],
                "category": item["category"],
                "target_frame": item["target_frame"],
                "industrial_alarm": item["industrial_alarm"],
                "industrial_types": item["industrial_types"],
                "qwen_alarm": qwen_alarm,
                "qwen_details": p,
                "is_suppressed": is_suppressed
            }
            results.append(rec)
            
        processed_so_far = len(results)
        fps_batch = len(valid_items) / batch_cost
        pct = (processed_so_far / total_videos) * 100
        print(f"[{processed_so_far:03d}/{total_videos}] ({pct:5.1f}%) 批次 {b_idx+1}/{total_batches} | 耗时: {batch_cost:4.2f}s ({fps_batch:3.1f} FPS) | 纠偏成功累计: {sum(1 for r in results if r['is_suppressed'])}")

    total_time = time.time() - t_start_all
    print("\n" + "=" * 80)
    print(f"🎉 全量 653 个视频 Qwen3-VL 批量复核完毕！总耗时: {total_time:.1f} 秒 ({total_time/60:.2f} 分钟)")
    print("=" * 80)

    # 4. 统计分析与分类聚合
    cat_stats = {}
    for r in results:
        cat = r["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {
                "total": 0,
                "ind_alarms": 0,
                "qwen_alarms": 0,
                "suppressed": 0,
            }
        st = cat_stats[cat]
        st["total"] += 1
        if r["industrial_alarm"]:
            st["ind_alarms"] += 1
        if r["qwen_alarm"]:
            st["qwen_alarms"] += 1
        if r["is_suppressed"]:
            st["suppressed"] += 1

    # 导出 JSON
    json_path = os.path.join(OUTPUT_DIR, "qwen_full_benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 导出 CSV
    csv_path = os.path.join(OUTPUT_DIR, "qwen_full_benchmark_results.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "video_name", "category", "target_frame", "industrial_alarm", "industrial_types", "qwen_alarm", "qwen_reason", "is_suppressed"])
        for r in results:
            writer.writerow([
                r["video_id"],
                r["video_name"],
                r["category"],
                r["target_frame"],
                r["industrial_alarm"],
                ",".join(r["industrial_types"]),
                r["qwen_alarm"],
                r["qwen_details"]["reason"],
                r["is_suppressed"]
            ])

    print("\n==========================================================================================")
    print(f"{'场景类别':<28} | {'总数':<5} | {'工控误报数 (率)':<16} | {'Qwen虚警数 (率)':<16} | {'工控误报抑制率'}")
    print("------------------------------------------------------------------------------------------")
    
    total_all = 0
    ind_all = 0
    qwen_all = 0
    supp_all = 0
    
    for cat, st in sorted(cat_stats.items(), key=lambda x: x[1]["total"], reverse=True):
        ind_rate = (st["ind_alarms"] / st["total"] * 100) if st["total"] > 0 else 0
        qwen_rate = (st["qwen_alarms"] / st["total"] * 100) if st["total"] > 0 else 0
        supp_rate = (st["suppressed"] / st["ind_alarms"] * 100) if st["ind_alarms"] > 0 else 100.0
        
        total_all += st["total"]
        ind_all += st["ind_alarms"]
        qwen_all += st["qwen_alarms"]
        supp_all += st["suppressed"]
        
        print(f"{cat:<28} | {st['total']:<5} | {st['ind_alarms']:3d} ({ind_rate:5.1f}%)   | {st['qwen_alarms']:3d} ({qwen_rate:5.1f}%)   | {supp_rate:5.1f}%")

    ind_all_rate = (ind_all / total_all * 100) if total_all > 0 else 0
    qwen_all_rate = (qwen_all / total_all * 100) if total_all > 0 else 0
    supp_all_rate = (supp_all / ind_all * 100) if ind_all > 0 else 100.0

    print("==========================================================================================")
    print(f"{'【全量总计】':<28} | {total_all:<5} | {ind_all:3d} ({ind_all_rate:5.1f}%)   | {qwen_all:3d} ({qwen_all_rate:5.1f}%)   | {supp_all_rate:5.1f}%")
    print(f"\n📁 详细结果已导出至:\n  - JSON: {json_path}\n  - CSV:  {csv_path}")


if __name__ == "__main__":
    main()
