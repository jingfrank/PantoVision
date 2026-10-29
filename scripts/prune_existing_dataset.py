#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/prune_existing_dataset.py
精简现有 /home/ibd/jyb/gongwang_fastad/dataset 数据集，为即将导入的新现场视频腾出空间。

策略说明：
1. 原 Train 集共有 406 帧，全量仅来自 3~4 个老视频：
   - spark_2 / spark_1 占了 239 帧 (占比 59%，极度密集冗余)；
   - tilt 占 80 帧；
   - foreign 占 67 帧；
   - frame_* 占 20 帧。
2. 保持等距时序抽样，将 406 帧瘦身至 200 帧左右：
   - spark: 239 帧 -> 均匀抽取 80 帧 (覆盖原有时段但去除相邻重复)；
   - tilt: 80 帧 -> 均匀抽取 50 帧；
   - foreign: 67 帧 -> 均匀抽取 50 帧；
   - frame_*: 保留全部 20 帧；
   合计：80 + 50 + 50 + 20 = 200 帧！
3. 与用户审核后的约 200 帧新现场视频合并后，整体 Train 规模精准控制在 400 帧左右！
4. 原数据集已完整备份在 /home/ibd/jyb/gongwang_fastad/dataset_backup_orig。
"""

import os
import shutil

DATASET_DIR = "/home/ibd/jyb/gongwang_fastad/dataset"
BACKUP_DIR = "/home/ibd/jyb/gongwang_fastad/dataset_backup_orig"

TARGET_COUNTS = {
    "spark": 80,
    "tilt": 50,
    "foreign": 50,
}


def prune_split(split_name: str, target_map: dict):
    split_dir = os.path.join(DATASET_DIR, split_name)
    all_files = sorted(os.listdir(split_dir))

    groups = {"spark": [], "tilt": [], "foreign": [], "frame_*": [], "other": []}
    for f in all_files:
        if f.startswith("frame_"):
            groups["frame_*"].append(f)
        elif "_spark_" in f:
            groups["spark"].append(f)
        elif "_tilt_" in f:
            groups["tilt"].append(f)
        elif "_foreign_" in f:
            groups["foreign"].append(f)
        else:
            groups["other"].append(f)

    keep_files = set()
    # 保留 frame_* 和 other
    for f in groups["frame_*"] + groups["other"]:
        keep_files.add(f)

    # 对各视频进行等步长均匀抽样
    for k, target_n in target_map.items():
        src_files = groups[k]
        if len(src_files) <= target_n:
            for f in src_files:
                keep_files.add(f)
        else:
            step = len(src_files) / float(target_n)
            sampled = [src_files[int(i * step)] for i in range(target_n)]
            for f in sampled:
                keep_files.add(f)

    # 执行删除非保留文件
    removed_count = 0
    for f in all_files:
        if f not in keep_files:
            os.remove(os.path.join(split_dir, f))
            removed_count += 1

    remaining = len(os.listdir(split_dir))
    print(f"[{split_name}] 原始: {len(all_files)} 帧 -> 删减冗余: {removed_count} 帧 -> 当前剩余: {remaining} 帧")


def main():
    if not os.path.exists(BACKUP_DIR):
        print(f"[错误] 尚未检测到完整备份目录 {BACKUP_DIR}，终止以策安全！")
        return

    print("=" * 60)
    print(" 开始精简 /home/ibd/jyb/gongwang_fastad/dataset 数据集...")
    print("=" * 60)

    # 精简 Train (406 -> 200)
    prune_split("train", TARGET_COUNTS)

    # 相应等比精简 Val (112 -> 60)
    val_targets = {"spark": 25, "tilt": 15, "foreign": 15}
    prune_split("val", val_targets)

    print("=" * 60)
    print(" 数据集精简完成！预留空间已就绪。")
    print(f" 原始完整数据位于: {BACKUP_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
