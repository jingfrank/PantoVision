#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training & Distribution Fitting Script for Pantograph FastAD Model.
Extracts normal baseline video frames and fits Gaussian & CDF distributions.
"""

import os
import sys
import argparse
import cv2
import torch
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastad_detector import PantographFastADModel


def extract_normal_frames(video_configs: list) -> list:
    """
    Extracts clean/normal baseline frames from video segments.
    """
    normal_frames = []

    for cfg in video_configs:
        vpath = cfg["path"]
        if not os.path.exists(vpath):
            print(f"[Warning] Video path {vpath} does not exist, skipping.")
            continue

        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(cfg.get("start_s", 0) * fps)
        end_frame = int(cfg.get("end_s", total_frames / fps) * fps)
        interval = int(cfg.get("interval_s", 1.0) * fps)
        if interval < 1:
            interval = 1

        print(f"[Dataset] Extracting frames from {os.path.basename(vpath)}: {start_frame}->{end_frame} every {interval} frames")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current = start_frame
        extracted = 0

        while current < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if (current - start_frame) % interval == 0:
                normal_frames.append(frame)
                extracted += 1
            current += 1

        cap.release()
        print(f"[Dataset] Loaded {extracted} clean frames from {os.path.basename(vpath)}")

    return normal_frames


def main():
    parser = argparse.ArgumentParser(description="Train / Fit FastAD Model for Pantograph Anomaly Detection")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference and training device")
    parser.add_argument("--output", type=str, default="models/fastad_pantograph.pt", help="Path to save fitted model state")
    parser.add_argument("--layer", type=int, default=2, help="ResNet feature extraction layer index (2 or 3)")
    parser.add_argument("--img_size", type=int, default=256, help="Input resolution (256 or 384)")
    parser.add_argument("--n_quantiles", type=int, default=3, help="Number of quantiles for CDF estimation")
    parser.add_argument("--fm_quantiles", type=int, default=1, help="Feature map quantile reduction")
    parser.add_argument("--reg", type=float, default=1e-4, help="Covariance regularization epsilon")
    args = parser.parse_args()

    # Video configuration pointing to normal segments
    # Adjust paths if videos are mounted in typical paths
    sample_videos = [
        {"path": "弓网典型视频/spark_2.mp4", "start_s": 0.0, "end_s": 120.0, "interval_s": 0.5},
        {"path": "弓网典型视频/spark_1.mp4", "start_s": 0.0, "end_s": 35.0, "interval_s": 0.5},
        {"path": "弓网典型视频/tilt_1.mp4", "start_s": 0.0, "end_s": 120.0, "interval_s": 0.5},
        {"path": "弓网典型视频/foreign_1.mp4", "start_s": 0.0, "end_s": 15.0, "interval_s": 0.5},
    ]

    print(f"=== FastAD Training & Fitting Pipeline ===")
    print(f"Target Device: {args.device}, Layer: {args.layer}, Img Size: {args.img_size}")
    print(f"Quantiles: n={args.n_quantiles}, fm={args.fm_quantiles}, Reg: {args.reg}")

    frames = extract_normal_frames(sample_videos)
    if not frames:
        print("[Error] No training frames were extracted. Check video paths.")
        sys.exit(1)

    print(f"[Training] Total clean normal frames collected: {len(frames)}")

    model = PantographFastADModel(
        device=args.device,
        layer_to_extract=args.layer,
        img_size=args.img_size,
        n_quantiles=args.n_quantiles,
        fm_quantiles=args.fm_quantiles,
        regularization=args.reg,
    )

    print("[Training] Fitting FastAD model distributions on normal frames...")
    model.fit(frames)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model.save(args.output)
    print(f"[Done] FastAD model successfully saved to: {args.output}")


if __name__ == "__main__":
    main()
