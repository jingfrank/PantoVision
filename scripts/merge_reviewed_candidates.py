#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/merge_reviewed_candidates.py
将人工审核通过的候选帧合并至 /home/ibd/jyb/gongwang_fastad/dataset (按 8:2 划分入 train/ 与 val/)。

用法：
在人工审核并删除了 /home/ibd/jyb/candidate_train_frames 中的脏污/火花帧后，运行：
    python scripts/merge_reviewed_candidates.py
"""

import os
import shutil
import random

CANDIDATE_DIR = "/home/ibd/jyb/candidate_train_frames"
DATASET_DIR = "/home/ibd/jyb/gongwang_fastad/dataset"

random.seed(42)


def main():
    if not os.path.exists(CANDIDATE_DIR):
        print(f"[错误] 候选目录不存在: {CANDIDATE_DIR}")
        return

    # 收集当前候选目录中剩余的所有有效图片
    valid_files = []
    for root, dirs, files in os.walk(CANDIDATE_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                valid_files.append(os.path.join(root, f))

    if not valid_files:
        print("[警告] 候选目录中未找到待合并图片！")
        return

    print(f"检测到审核通过的候选帧: 共 {len(valid_files)} 帧")

    # 随机打乱按 8:2 切分
    random.shuffle(valid_files)
    split_idx = int(len(valid_files) * 0.8)
    train_add = valid_files[:split_idx]
    val_add = valid_files[split_idx:]

    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir = os.path.join(DATASET_DIR, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 拷贝到 train
    for src in train_add:
        dst = os.path.join(train_dir, os.path.basename(src))
        shutil.copy2(src, dst)

    # 拷贝到 val
    for src in val_add:
        dst = os.path.join(val_dir, os.path.basename(src))
        shutil.copy2(src, dst)

    total_train = len(os.listdir(train_dir))
    total_val = len(os.listdir(val_dir))

    print("=" * 60)
    print(" 审核帧合并完成！")
    print("=" * 60)
    print(f" 新增注入 Train (80%): {len(train_add)} 帧")
    print(f" 新增注入 Val   (20%): {len(val_add)} 帧")
    print("-" * 60)
    print(f" 合并后 Train 总量: {total_train} 帧 (目标 ~400 帧)")
    print(f" 合并后 Val   总量: {total_val} 帧")
    print("=" * 60)


if __name__ == "__main__":
    main()
