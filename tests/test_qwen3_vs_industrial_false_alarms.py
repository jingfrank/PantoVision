#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-4B 对比评测脚本 (针对工控防线误报视频)
用途：抽取 5 段工控防线存在误报的典型视频，对比 Qwen3-VL-4B 大模型的语义研判能力与误报表现。
"""

import os
import sys
import time
import json
import cv2
import torch
import numpy as np
from PIL import Image

# 禁用输出缓冲
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

MODEL_PATH = "models/Qwen3-VL-4B-Instruct"

# 5 条代表性评测视频配置
TEST_VIDEOS = [
    {
        "id": "Video_1",
        "name": "20260419_03_08_075944571_G6123_A",
        "path": "/media/ibd/USB_DISK/典型故障视频/误报视频/受电弓高亮/CR400-AF-S-2365/20260419_03_08_075944571_G6123_A.MP4",
        "customer_label": "受电弓高亮 (正常运行强顺光)",
        "industrial_result": "【误报】51.0s过天桥误报画面脏污，6s/11s/40s反光误报结构异常",
        # 挑选具有代表性的帧：包括工控防线误报帧（151=6s结构, 276=11s结构, 1276=51s脏污）+ 正常帧
        "target_frames": [151, 276, 500, 1001, 1251, 1276, 1300, 1450]
    },
    {
        "id": "Video_2",
        "name": "20260404_03_08_212100221_G253",
        "path": "/media/ibd/USB_DISK/典型故障视频/误报视频/火花/CR400BF-AS-3269/20260404_03_08_212100221_G253.mp4",
        "customer_label": "火花 (夜间微小滑板火花，非大火花故障)",
        "industrial_result": "【正确抑制】工控YOLO准确过滤轻微放电，大火花0告警",
        "target_frames": [50, 100, 175, 250, 325, 400, 475, 550]
    },
    {
        "id": "Video_3",
        "name": "20260401_03_08_143752388_G246",
        "path": "/media/ibd/USB_DISK/典型故障视频/误报视频/受电弓结构异常-弓角缺失/CR400BF-AS-3252/20260401_03_08_143752388_G246.mp4",
        "customer_label": "客户误报: 弓角缺失 (实际弓角完好)",
        "industrial_result": "【误报】工控FastAD误报结构异常与画面脏污",
        "target_frames": [25, 75, 150, 225, 300, 450, 600, 750]
    },
    {
        "id": "Video_4",
        "name": "20260420_03_08_162209584_G6309_A",
        "path": "/media/ibd/USB_DISK/典型故障视频/误报视频/受电弓高亮/CR400-FS-S-2370/20260420_03_08_162209584_G6309_A.MP4",
        "customer_label": "受电弓高亮 (金属表面高光白斑)",
        "industrial_result": "【误报】工控FastAD误报结构异常与画面脏污",
        "target_frames": [50, 150, 250, 350, 500, 750, 1000, 1250]
    },
    {
        "id": "Video_5",
        "name": "20260423_03_08_184852161_G6308_A",
        "path": "/media/ibd/USB_DISK/典型故障视频/误报视频/受电弓结构异常-弓角断裂/CR400-FS-S-2370/20260423_03_08_184852161_G6308_A.MP4",
        "customer_label": "客户误报: 弓角断裂 (实际无断裂损伤)",
        "industrial_result": "【误报】工控FastAD时序EMA误报画面脏污",
        "target_frames": [50, 150, 250, 400, 600, 800, 1100, 1350]
    }
]

PROMPT = (
    "你是一个铁路车载受电弓与接触网故障诊断专家。请仔细观察这张受电弓监控抓拍图像，严格判断是否存在真正的物理故障或重大异常：\n"
    "1. spark: 大火花。受电弓滑板与接触网间出现剧烈电弧爆燃/连续大火花（正常的运行微小蓝白星火不算大火花）。\n"
    "2. structural: 结构异常。受电弓碳滑板断裂/严重缺失、弓角折断、框架严重变形或部件脱落（阳光反光产生的高亮白斑属于正常光学现象，不是破损）。\n"
    "3. dirt: 画面脏污。镜头防护玻璃粘附严重泥斑、大面积油污、飞虫残体，导致无法辨认设备（列车穿过上方天桥/跨线桥产生的背景暗影遮挡属于外界环境，不是镜头脏污）。\n"
    "4. tilt: 受电弓明显倾斜（横梁倾角严重超标）。\n"
    "5. foreign_object: 接触网或弓头缠绕悬挂塑料膜、风筝线、树枝等外来异物。\n"
    "请仅按标准 JSON 格式输出，1表示确诊异常，0表示正常：\n"
    '{"spark":0, "structural":0, "dirt":0, "tilt":0, "foreign_object":0, "analysis":"50字以内的状态分析理由"}'
)


def load_model():
    print("=" * 70)
    print("🚀 [Qwen3-VL-4B] 正在初始化加载多模态视觉大模型...")
    print(f"   模型路径: {MODEL_PATH}")
    print("=" * 70)
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

    # 解析 JSON
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
    save_dir = "local_cache/qwen3_eval_results"
    os.makedirs(save_dir, exist_ok=True)

    model, processor = load_model()

    summary_records = []

    for v_idx, v_cfg in enumerate(TEST_VIDEOS, 1):
        v_path = v_cfg["path"]
        v_name = v_cfg["name"]
        print("-" * 75)
        print(f"🎬 [{v_idx}/{len(TEST_VIDEOS)}] 正在测试视频: {v_name}")
        print(f"   ├─ 客户原标签: {v_cfg['customer_label']}")
        print(f"   └─ 工控防线表现: {v_cfg['industrial_result']}")

        if not os.path.exists(v_path):
            print(f"   ❌ 视频文件不存在: {v_path}")
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

            has_fault = any(res[k] > 0 for k in ["spark", "structural", "dirt", "tilt", "foreign_object"])
            for k in video_alarms.keys():
                if res[k] > 0:
                    video_alarms[k] += 1

            faults_detected = [k for k in ["spark", "structural", "dirt", "tilt", "foreign_object"] if res[k] > 0]
            status_str = f"🚨 告警: {faults_detected}" if has_fault else "✅ 正常无异常"

            print(f"   [帧 {f_idx:04d} / {pts_s:5.2f}s] [{lat_ms:4.0f}ms] {status_str} | 研判: {res['analysis']}")

            frame_details.append({
                "frame": f_idx,
                "timestamp_s": round(pts_s, 2),
                "prediction": res,
                "latency_ms": round(lat_ms, 1)
            })

        cap.release()

        # 综合判定该视频在 Qwen3-VL-4B 下的表现
        has_any_alarm = sum(video_alarms.values()) > 0
        summary_records.append({
            "video_id": v_cfg["id"],
            "video_name": v_name,
            "customer_label": v_cfg["customer_label"],
            "industrial_result": v_cfg["industrial_result"],
            "qwen3_alarms": video_alarms,
            "qwen3_has_alarm": has_any_alarm,
            "frame_details": frame_details
        })

    # 保存评测结果为 JSON
    report_file = os.path.join(save_dir, "qwen3_vs_industrial_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 75)
    print(" 📊 Qwen3-VL-4B 与工控防线对比测试总结")
    print("=" * 75)
    for rec in summary_records:
        alarm_types = [k for k, v in rec["qwen3_alarms"].items() if v > 0]
        q_result = f"🚨 产生告警: {alarm_types}" if rec["qwen3_has_alarm"] else "🟢 判定正常 (无误报)"
        print(f"▶ {rec['video_id']} ({rec['video_name']}):")
        print(f"   ├─ 客户原标签: {rec['customer_label']}")
        print(f"   ├─ 工控防线表现: {rec['industrial_result']}")
        print(f"   └─ Qwen3-VL表现: {q_result}")

    print(f"\n📁 详细评测记录已保存至: {report_file}")


if __name__ == "__main__":
    main()
