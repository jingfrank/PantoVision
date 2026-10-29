#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/extract_candidate_train_frames.py
从 653 个现场“误报视频”中均衡分层抽取待审核的 FastAD 训练候选帧。

功能说明：
1. 遍历 /home/ibd/jyb/误报视频 目录下的 10 大场景目录；
2. 跨不同车型（CR400AF/BF/FS等）均匀采样视频；
3. 严格避开已在测试集 /home/ibd/jyb/Dataset/ 中使用的帧序号（防止数据泄露）；
4. 抽取约 250 帧高质量候选帧，保存至独立目录 /home/ibd/jyb/candidate_train_frames/；
5. 按场景分类建子目录，方便人工在文件管理器中直观缩略图查看，剔除火花与脏污帧。
"""

import os
import sys
import json
import csv
import random
import cv2
from typing import Dict, List, Any

# 随机种子，确保可复现
random.seed(42)

SOURCE_BASE = "/home/ibd/jyb/误报视频"
OUTPUT_BASE = "/home/ibd/jyb/candidate_train_frames"
TEST_MANIFEST = "/home/ibd/jyb/Dataset/manifest.json"

# 每个类别的抽样目标帧数 (总计约 250 帧)
CATEGORY_TARGETS = {
    "受电弓高亮": 35,
    "太阳光": 30,
    "受电弓结构异常-碳滑板变形": 35,
    "受电弓结构异常-弓角断裂": 30,
    "受电弓结构异常-弓角缺失": 35,
    "受电弓结构异常-监控不完整": 30,
    "受电弓结构异常-识别不到弓": 10,
    "雨天": 25,
    "灯光": 8,
    "火花": 12,
}

# 候选采样的帧序号（避开测试集常用的 100, 150, 200, 400, 450, 500, 600, 700, 750, 800）
SAFE_FRAME_INDICES = [250, 350, 550, 650, 850, 950, 1100, 1250]


def collect_test_set_signatures(manifest_path: str) -> set:
    """收集测试集中已使用的视频名与帧序号组合，确保零交叉"""
    signatures = set()
    if not os.path.exists(manifest_path):
        return signatures

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        fn = item.get("filename", "")
        # 例如 glint_0040_CR400-FS-S-2370_f00200.jpg
        signatures.add(fn)
        if "source_file" in item and "frame_idx" in item:
            signatures.add(f"{os.path.basename(item['source_file'])}__f{item['frame_idx']}")

    return signatures


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    test_signatures = collect_test_set_signatures(TEST_MANIFEST)
    print(f"已加载测试集签名防碰撞库: {len(test_signatures)} 条规则")

    extracted_records = []
    item_id = 1

    for cat_name, target_count in CATEGORY_TARGETS.items():
        cat_dir = os.path.join(SOURCE_BASE, cat_name)
        if not os.path.exists(cat_dir):
            print(f"[警告] 场景目录不存在: {cat_dir}")
            continue

        out_cat_dir = os.path.join(OUTPUT_BASE, cat_name)
        os.makedirs(out_cat_dir, exist_ok=True)

        # 收集该场景下的所有视频
        video_paths = []
        for root, dirs, files in os.walk(cat_dir):
            for f in sorted(files):
                if f.lower().endswith((".mp4", ".avi", ".mkv")):
                    video_paths.append(os.path.join(root, f))

        if not video_paths:
            continue

        # 随机打乱视频顺序
        random.shuffle(video_paths)

        cat_extracted = 0
        v_idx = 0
        while cat_extracted < target_count and v_idx < len(video_paths):
            vp = video_paths[v_idx]
            v_name = os.path.basename(vp)
            parent_train_model = os.path.basename(os.path.dirname(vp))

            cap = cv2.VideoCapture(vp)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

            if total_frames < 100:
                cap.release()
                v_idx += 1
                continue

            # 每个视频尝试取 1~2 帧
            frames_to_try = [f for f in SAFE_FRAME_INDICES if f < total_frames - 25]
            random.shuffle(frames_to_try)

            v_success = 0
            for f_idx in frames_to_try:
                sig_key = f"{v_name}__f{f_idx}"
                if sig_key in test_signatures:
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    continue

                # 简单校验：非纯黑/纯白帧
                mean_val = frame.mean()
                if mean_val < 15 or mean_val > 240:
                    continue

                # 保存帧
                clean_train_name = parent_train_model.replace(" ", "_")
                fname = f"cand_{item_id:04d}_{clean_train_name}_f{f_idx:05d}.jpg"
                save_path = os.path.join(out_cat_dir, fname)
                cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                rel_save_path = os.path.relpath(save_path, OUTPUT_BASE)
                extracted_records.append({
                    "id": item_id,
                    "relative_path": rel_save_path,
                    "category": cat_name,
                    "train_model": parent_train_model,
                    "source_video": vp,
                    "video_filename": v_name,
                    "frame_idx": f_idx,
                    "timestamp_s": round(f_idx / fps, 2),
                    "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                })

                item_id += 1
                cat_extracted += 1
                v_success += 1

                if cat_extracted >= target_count or v_success >= 2:
                    break

            cap.release()
            v_idx += 1

        print(f"场景 【{cat_name:25s}】: 成功抽取 {cat_extracted:3d} 帧 (目标: {target_count}) -> 保存至 {out_cat_dir}")

    # 保存元数据清单
    manifest_csv = os.path.join(OUTPUT_BASE, "candidate_manifest.csv")
    csv_headers = ["id", "relative_path", "category", "train_model", "video_filename", "frame_idx", "timestamp_s", "resolution", "source_video"]
    with open(manifest_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(extracted_records)

    # 生成审核指引说明
    readme_path = os.path.join(OUTPUT_BASE, "README_审核说明.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"""# FastAD 训练候选帧审核说明

**生成时间**: {os.popen('date').read().strip()}
**总抽取候选帧数**: **{len(extracted_records)} 帧**
**存放目录**: `{OUTPUT_BASE}`

## 审核指引（人工清洗原则）：
1. **删除火花帧**：如果在图片中受电弓与接触网接触点有刺眼的大电弧弧光（大火花），请直接删除；
2. **删除镜头泥污帧**：如果在镜头上有大块固定的水滴泥斑（画面脏污），请直接删除；
3. **保留正常多态背景**：包含逆光、直射太阳光晕、吊弦穿过受电弓构架、雨天水滴反光、隧道阴影等**正常受电弓工作状态**，均请保留；
4. 审核完成后，剩余保留的高质量正常帧即可合并入训练集。
""")

    print("\n" + "=" * 70)
    print(f" 候选帧抽取完成！总计抽取: {len(extracted_records)} 帧")
    print(f" 保存路径: {OUTPUT_BASE}")
    print(f" 审核说明: {readme_path}")
    print(f" 元数据表: {manifest_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
