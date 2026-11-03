#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid Search Script for FastAD3 Quantile Parameters (fm_quantiles & n_quantiles in [2..5]).
Evaluates hyperparameter combinations on stratified frames across benchmark videos.
Generates comprehensive heatmaps, comparison matrices, and summary reports.
"""

import os
import sys
import time
import json
import cv2
import torch
import numpy as np
import pandas as pd
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastad_detector import PantographFastADModel, PantographFaultClassifier


def get_ground_truth(video_name: str, timestamp_s: float) -> str:
    """
    Returns exact segment-level ground truth label for typical videos.
    """
    vname = os.path.splitext(video_name)[0]
    if vname == "spark_1":
        return "正常" if timestamp_s < 35.0 else "大火花"
    elif vname in ["spark_2", "tilt_1", "2886f3ca91a7e16d6f19010f1c7e893c", "50a6ccc7880951156014c12b8776aa38", "ea21a67195dda70dee671d7b3d54b929"]:
        return "正常"
    elif "dirty" in vname or "脏污" in vname:
        return "画面脏污"
    elif "foreign" in vname or "异物" in vname:
        return "悬挂异物"
    elif "tilt" in vname or "倾斜" in vname:
        return "受电弓倾斜"
    return "未知"


def run_grid_search(video_paths: list, output_dir: str = "results_grid_search"):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    quantile_range = [2, 3, 4, 5]
    print(f"Starting FastAD Quantile Grid Search on {len(video_paths)} videos...")

    for n_q in quantile_range:
        for fm_q in quantile_range:
            combo_name = f"n_q{n_q}_fm_q{fm_q}"
            print(f"\n---> Evaluating combination: {combo_name}")

            model = PantographFastADModel(
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                layer_to_extract=2,
                img_size=256,
                n_quantiles=n_q,
                fm_quantiles=fm_q,
                regularization=1e-4,
            )

            # Record stats
            for vpath in video_paths:
                if not os.path.exists(vpath):
                    continue
                vname = os.path.basename(vpath)
                cap = cv2.VideoCapture(vpath)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # sample 1 fps
                    if frame_idx % int(fps) == 0:
                        ts = frame_idx / fps
                        gt = get_ground_truth(vname, ts)
                        t0 = time.perf_counter()
                        res = model.predict(frame)
                        lat_ms = (time.perf_counter() - t0) * 1000.0

                        results.append({
                            "combo": combo_name,
                            "n_quantiles": n_q,
                            "fm_quantiles": fm_q,
                            "video": vname,
                            "timestamp_s": ts,
                            "ground_truth": gt,
                            "anomaly_score": res["anomaly_score"],
                            "latency_ms": lat_ms
                        })
                    frame_idx += 1
                cap.release()

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "grid_search_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Completed] Grid search results saved to {csv_path}")


if __name__ == "__main__":
    test_vids = [
        "弓网典型视频/spark_1.mp4",
        "弓网典型视频/foreign_1.mp4",
        "弓网典型视频/tilt_1.mp4",
    ]
    run_grid_search(test_vids)
