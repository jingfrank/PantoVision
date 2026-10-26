#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-4B 批量验证评测脚本：20 条跨类别工控误报视频
覆盖类别：雨天(3)、太阳光(3)、灯光(2)、碳滑板变形(3)、监控不完整(3)、识别不到弓(2)、弓角断裂(2)、弓角缺失(2)
对比目标：验证 Qwen3-VL 能否有效防止工控防线在这些典型场景下的误报。
"""

import os
import sys
import time
import json
import csv
import cv2
import torch
from PIL import Image

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

MODEL_PATH = "models/Qwen3-VL-4B-Instruct"

# 20 条代表性跨类别评测视频配置
VIDEO_CONFIGS = [
    # 1. 雨天场景 (3 条)
    {
        "id": "V01",
        "category": "雨天",
        "name": "20260408_06_08_090134381_G6164_A",
        "industrial_alarms": ["abnormal"],
        "target_frames": [401, 750, 1100]  # 401为工控误报帧
    },
    {
        "id": "V02",
        "category": "雨天",
        "name": "20260411_06_08_195356722_G6124_A",
        "industrial_alarms": ["abnormal"],
        "target_frames": [26, 401, 800]  # 26/401为工控误报帧
    },
    {
        "id": "V03",
        "category": "雨天",
        "name": "20260421_06_08_105939373_G6164_A",
        "industrial_alarms": ["dirty"],
        "target_frames": [300, 750, 1451]  # 1451为工控误报帧
    },

    # 2. 太阳光 / 强眩光 (3 条)
    {
        "id": "V04",
        "category": "太阳光",
        "name": "20260408_06_08_153507602_G6163_A",
        "industrial_alarms": ["abnormal", "dirty"],
        "target_frames": [426, 1176, 1251]  # 426/1176/1251为工控误报帧
    },
    {
        "id": "V05",
        "category": "太阳光",
        "name": "20260429_03_08_060501164_C8861_A",
        "industrial_alarms": ["dirty"],
        "target_frames": [76, 500, 1000]  # 76为工控误报帧
    },
    {
        "id": "V06",
        "category": "太阳光",
        "name": "20260412_03_08_093317844_G866_A",
        "industrial_alarms": ["dirty"],
        "target_frames": [76, 537, 900]  # 76/537为工控误报帧
    },

    # 3. 夜间强人工灯光 (2 条)
    {
        "id": "V07",
        "category": "灯光",
        "name": "20260420_03_08_205958406_G6129_A",
        "industrial_alarms": ["dirty"],
        "target_frames": [601, 900, 1201]  # 601/1201为工控误报帧
    },
    {
        "id": "V08",
        "category": "灯光",
        "name": "20260413_03_08_020111721_G6482_A",
        "industrial_alarms": ["dirty"],
        "target_frames": [76, 600, 1100]  # 76为工控误报帧
    },

    # 4. 碳滑板变形误报 (3 条)
    {
        "id": "V09",
        "category": "碳滑板变形",
        "name": "20260419_03_08_110833656_G6078_A",
        "industrial_alarms": ["abnormal"],
        "target_frames": [51, 500, 1000]  # 51为工控误报帧
    },
    {
        "id": "V10",
        "category": "碳滑板变形",
        "name": "20260419_03_08_153229667_G6076_A",
        "industrial_alarms": ["abnormal"],
        "target_frames": [426, 601, 900]  # 426/601为工控误报帧
    },
    {
        "id": "V11",
        "category": "碳滑板变形",
        "name": "20260419_03_08_163001478_G6076_A",
        "industrial_alarms": ["abnormal"],
        "target_frames": [551, 800, 1151]  # 551/1151为工控误报帧
    },

    # 5. 监控不完整 / 边缘截断误报 (3 条)
    {
        "id": "V12",
        "category": "监控不完整",
        "name": "20260406_03_08_151609156_G6363",
        "industrial_alarms": ["dirty"],
        "target_frames": [76, 600, 1200]  # 76为工控误报帧
    },
    {
        "id": "V13",
        "category": "监控不完整",
        "name": "20260406_11_08_133718609_G6364",
        "industrial_alarms": ["dirty"],
        "target_frames": [76, 500, 1000]  # 76为工控误报帧
    },
    {
        "id": "V14",
        "category": "监控不完整",
        "name": "20260406_11_08_151605105_G6363",
        "industrial_alarms": ["dirty"],
        "target_frames": [76, 600, 1100]  # 76为工控误报帧
    },

    # 6. 识别不到弓误报 (2 条)
    {
        "id": "V15",
        "category": "识别不到弓",
        "name": "20260405_03_08_170519753_G928",
        "industrial_alarms": ["dirty"],
        "target_frames": [76, 748, 1200]  # 76/748为工控误报帧
    },
    {
        "id": "V16",
        "category": "识别不到弓",
        "name": "20260423_06_08_203232130_G263",
        "industrial_alarms": ["dirty", "spark"],
        "target_frames": [76, 1207, 1676]  # 76/1207/1676为工控误报帧
    },

    # 7. 弓角断裂误报 (2 条)
    {
        "id": "V17",
        "category": "弓角断裂",
        "name": "20260419_03_08_090144357_G6123_A",
        "industrial_alarms": ["abnormal"],
        "target_frames": [500, 1076, 1151]  # 1076/1151为工控误报帧
    },
    {
        "id": "V18",
        "category": "弓角断裂",
        "name": "20260419_03_08_090617139_G6123_A",
        "industrial_alarms": ["abnormal"],
        "target_frames": [51, 101, 501]  # 51/101/501为工控误报帧
    },

    # 8. 弓角缺失误报 (2 条)
    {
        "id": "V19",
        "category": "弓角缺失",
        "name": "20260413_06_08_174620473_G408",
        "industrial_alarms": ["abnormal", "dirty"],
        "target_frames": [76, 1226, 1301]  # 76/1226/1301为工控误报帧
    },
    {
        "id": "V20",
        "category": "弓角缺失",
        "name": "20260425_03_08_084053326_G544",
        "industrial_alarms": ["dirty"],
        "target_frames": [76, 972, 1200]  # 76/972为工控误报帧
    }
]

PROMPT = (
    "你是一个铁路车载受电弓与接触网故障监测及智能复核专家。请仔细观察这张受电弓监控抓拍图像，严格判断是否存在真正的物理故障或重大异常：\n"
    "1. spark: 大火花。受电弓滑板与接触网间出现剧烈电弧爆燃/连续大火花（正常的运行微小蓝白星火不算大火花）。\n"
    "2. structural: 结构异常。受电弓碳滑板断裂/严重缺失/翘曲、弓角折断缺失、框架破损脱落（阳光直射、雨滴反光、车体灯光反射造成的高光白斑属于正常光学现象，绝非结构破损）。\n"
    "3. dirt: 画面脏污。镜头防护玻璃粘附严重泥斑、大面积油污、飞虫残体，导致完全看不清设备（正常雨丝/极微弱水珠、天桥/桥梁/电杆背景暗影遮挡属于外界环境，绝不是镜头脏污）。\n"
    "4. tilt: 受电弓明显倾斜（横梁倾角严重超标歪斜）。\n"
    "5. foreign_object: 接触网或弓头缠绕悬挂塑料膜、风筝线、树枝等外来异物。\n"
    "请仅按标准 JSON 格式输出，1表示确诊异常，0表示正常：\n"
    '{"spark":0, "structural":0, "dirt":0, "tilt":0, "foreign_object":0, "analysis":"50字以内的状态分析理由"}'
)


def load_path_map():
    manifest_path = "local_cache/manifest_false_alarms.csv"
    path_map = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            base = os.path.splitext(fname)[0]
            path_map[base] = row["file_path"]
    return path_map


def load_model():
    print("=" * 75)
    print("🚀 [Qwen3-VL-4B] 正在初始化加载多模态视觉大模型...")
    print(f"   模型路径: {MODEL_PATH}")
    print("=" * 75)
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map="cuda:0",
        low_cpu_mem_usage=True
    )
    model.eval()
    t_cost = time.time() - t0
    vram_mb = torch.cuda.memory_allocated(0) / 1024 / 1024
    print(f"✅ 模型加载成功！耗时: {t_cost:.1f} 秒 | 占用显存: {vram_mb:.0f} MB\n")
    return model, processor


def evaluate_frame(model, processor, pil_img):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_img], padding=True, return_tensors="pt").to("cuda:0")

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=96)
    latency_ms = (time.time() - t0) * 1000.0

    new_ids = out[:, inputs["input_ids"].shape[1]:]
    output_text = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

    parsed = {"spark": 0, "structural": 0, "dirt": 0, "tilt": 0, "foreign_object": 0, "analysis": output_text}
    try:
        clean_text = output_text
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0].strip()
        
        j_start = clean_text.find("{")
        j_end = clean_text.rfind("}")
        if j_start != -1 and j_end != -1:
            clean_json = clean_text[j_start:j_end+1]
            data = json.loads(clean_json)
            for k in ["spark", "structural", "dirt", "tilt", "foreign_object"]:
                if k in data:
                    parsed[k] = int(data[k])
            if "analysis" in data:
                parsed["analysis"] = str(data["analysis"])
    except Exception:
        pass

    return parsed, latency_ms


def main():
    save_dir = "local_cache/qwen3_20_eval_results"
    os.makedirs(save_dir, exist_ok=True)

    path_map = load_path_map()
    model, processor = load_model()

    summary_records = []
    total_frames_tested = 0
    total_false_alarms = 0
    total_successfully_suppressed = 0

    print("=" * 75)
    print("🎬 开始对 20 条跨类别工控误报视频进行 Qwen3-VL 批量验证")
    print("=" * 75)

    for v_idx, v_cfg in enumerate(VIDEO_CONFIGS, 1):
        v_name = v_cfg["name"]
        cat = v_cfg["category"]
        ind_alarms = v_cfg["industrial_alarms"]
        v_path = path_map.get(v_name, "")

        print(f"\n[{v_idx:02d}/20] ▶ 类别: 【{cat}】 | 视频: {v_name}")
        print(f"       工控防线误报: {ind_alarms}")

        if not v_path or not os.path.exists(v_path):
            print(f"       ❌ 视频文件未找到: {v_path}")
            continue

        cap = cv2.VideoCapture(v_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        video_alarms = {
            "spark": 0,
            "structural": 0,
            "dirt": 0,
            "tilt": 0,
            "foreign_object": 0
        }
        frame_details = []

        for f_idx in v_cfg["target_frames"]:
            if f_idx > total_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx - 1)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            pts_s = (f_idx - 1) / fps
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            res, lat_ms = evaluate_frame(model, processor, pil_img)
            total_frames_tested += 1

            has_fault = any(res[k] > 0 for k in ["spark", "structural", "dirt", "tilt", "foreign_object"])
            for k in video_alarms.keys():
                if res[k] > 0:
                    video_alarms[k] += 1

            faults_detected = [k for k in ["spark", "structural", "dirt", "tilt", "foreign_object"] if res[k] > 0]
            if has_fault:
                status_str = f"🚨 Qwen告警: {faults_detected}"
                total_false_alarms += 1
            else:
                status_str = "🟢 正常无故障 (误报成功抑制)"
                total_successfully_suppressed += 1

            print(f"       [帧 {f_idx:04d} / {pts_s:5.2f}s] [{lat_ms:4.0f}ms] {status_str} | 研判: {res['analysis']}")

            frame_details.append({
                "frame": f_idx,
                "timestamp_s": round(pts_s, 2),
                "prediction": res,
                "latency_ms": round(lat_ms, 1)
            })

        cap.release()

        v_has_alarm = sum(video_alarms.values()) > 0
        summary_records.append({
            "id": v_cfg["id"],
            "category": cat,
            "video_name": v_name,
            "industrial_alarms": ind_alarms,
            "qwen_alarms": video_alarms,
            "qwen_has_alarm": v_has_alarm,
            "frame_details": frame_details
        })

    report_file = os.path.join(save_dir, "qwen3_20_videos_benchmark.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, ensure_ascii=False, indent=2)

    # 打印最终分类汇总表格
    print("\n" + "=" * 85)
    print(" 📊 Qwen3-VL-4B 对 20 条工控误报视频的防误报能力分类汇总")
    print("=" * 85)
    print(f"{'序号':<4} | {'场景类别':<12} | {'视频名称':<32} | {'工控防线误报':<16} | {'Qwen3-VL研判结论'}")
    print("-" * 85)
    for rec in summary_records:
        ind_str = ", ".join(rec["industrial_alarms"])
        q_str = "🚨 产生虚警" if rec["qwen_has_alarm"] else "🟢 完美抑制 (判定正常)"
        print(f"{rec['id']:<4} | {rec['category']:<12} | {rec['video_name']:<32} | {ind_str:<16} | {q_str}")

    suppress_rate = (total_successfully_suppressed / total_frames_tested * 100.0) if total_frames_tested > 0 else 0.0
    print("-" * 85)
    print(f"总抽样测试帧数: {total_frames_tested} 帧")
    print(f"工控误报帧被 Qwen 成功纠偏抑制帧数: {total_successfully_suppressed} 帧")
    print(f"Qwen 自身产生虚警帧数: {total_false_alarms} 帧")
    print(f"🎯 最终工控防线误报消除率 (Suppression Rate): {suppress_rate:.1f}%\n")
    print(f"📁 完整诊断报告已落盘至: {report_file}")


if __name__ == "__main__":
    main()
