#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fault Classifier and Spatial-Temporal Decoupling Engine for Pantograph FastAD.
Classifies visual anomalies into:
1. dirt (画面脏污): Static persistent stains in the camera coordinate frame.
2. spark (大火花): Arc discharge in the contact strip region with high brightness.
3. foreign_object (悬挂异物): Open-set anomalies hanging on catenary wires.
4. structural_anomaly (受电弓结构异常): Concrete deformation/damage in pantograph frame.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional


class PantographFaultClassifier:
    """
    Decouples FastAD visual anomalies into specific pantograph fault categories.
    """

    def __init__(
        self,
        # Spark detection parameters (Contact strip interaction zone)
        slide_roi_ratio: Tuple[float, float, float, float] = (0.25, 0.25, 0.75, 0.48),
        spark_brightness_threshold: int = 245,
        spark_min_pixel_count: int = 30000,
        spark_anomaly_score_threshold: float = 2.0,
        # Dirt detection parameters
        dirt_grid_size: int = 64,
        dirt_persistence_frames: int = 4,
        dirt_score_threshold: float = 2.0,
        dirt_pending_cells_threshold: int = 15,
        # Foreign object parameters (Focused on pantograph contact & catenary interaction zone)
        catenary_roi_ratio: Tuple[float, float, float, float] = (
            0.25,
            0.25,
            0.85,
            0.50,
        ),
        foreign_min_area: int = 100,
        foreign_score_threshold: float = 2.2,
        max_blobs_filter: int = 12,  # Rejects station ceiling/dense trusses
        # Pantograph structural anomaly parameters (15%~70% Y covering pantograph head and frame)
        pantograph_roi_ratio: Tuple[float, float, float, float] = (
            0.15,
            0.15,
            0.85,
            0.70,
        ),
        structural_min_area: int = 250,
    ):
        self.slide_roi_ratio = slide_roi_ratio
        self.spark_brightness_threshold = spark_brightness_threshold
        self.spark_min_pixel_count = spark_min_pixel_count
        self.spark_anomaly_score_threshold = spark_anomaly_score_threshold

        self.dirt_grid_size = dirt_grid_size
        self.dirt_persistence_frames = dirt_persistence_frames
        self.dirt_score_threshold = dirt_score_threshold
        self.dirt_pending_cells_threshold = dirt_pending_cells_threshold

        self.catenary_roi_ratio = catenary_roi_ratio
        self.foreign_min_area = foreign_min_area
        self.foreign_score_threshold = foreign_score_threshold
        self.max_blobs_filter = max_blobs_filter

        self.pantograph_roi_ratio = pantograph_roi_ratio
        self.structural_min_area = structural_min_area

        # Temporal grid history for dirt persistence
        self.dirt_grid_history: Optional[np.ndarray] = None
        self.history_length = 0

    def reset_state(self):
        """Resets temporal tracking state."""
        self.dirt_grid_history = None
        self.history_length = 0

    def analyze_frame(
        self,
        frame: np.ndarray,
        fastad_result: dict,
        timestamp: float = 0.0,
    ) -> dict:
        """
        Analyzes a single video frame with FastAD anomaly map.
        """
        h, w = frame.shape[:2]
        anomaly_map = fastad_result["anomaly_map"]
        img_anomaly_score = fastad_result["anomaly_score"]

        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        )

        img_thresh = fastad_result.get(
            "image_threshold", fastad_result.get("calibrated_image_threshold", 4.0)
        )
        pix_thresh = fastad_result.get(
            "pixel_threshold", fastad_result.get("calibrated_pixel_threshold", 2.8)
        )

        # -------------------------------------------------------------
        # 1. Spark (大火花) Detection: Contact strip arc discharge (Highest physical priority)
        # -------------------------------------------------------------
        rx1, ry1, rx2, ry2 = (
            int(self.slide_roi_ratio[0] * w),
            int(self.slide_roi_ratio[1] * h),
            int(self.slide_roi_ratio[2] * w),
            int(self.slide_roi_ratio[3] * h),
        )

        slide_gray = gray[ry1:ry2, rx1:rx2]
        slide_anom = anomaly_map[ry1:ry2, rx1:rx2]

        sat_mask = slide_gray >= self.spark_brightness_threshold
        sat_pixel_count = int(np.sum(sat_mask))
        slide_mean_gray = float(np.mean(slide_gray))
        sat_anom_mean = (
            float(np.mean(slide_anom[sat_mask])) if sat_pixel_count > 0 else 0.0
        )
        sat_anom_max = (
            float(np.max(slide_anom[sat_mask])) if sat_pixel_count > 0 else 0.0
        )

        spark_detected = False
        spark_details = {}

        # 真实大火花物理特征综合判据：
        # 1. 强电弧大面积白炽放电：饱和像素数 >= 60,000 且 灰度均值 >= 165.0 且 饱和区异常均值 >= 1.8
        # 2. 局部剧烈放电电弧：饱和像素数 >= 30,000 且 饱和区异常均值 >= 2.5
        # 3. 超高能强弧漫射：饱和区异常均值 >= 3.5 且 全图异常分超阈值
        if img_anomaly_score >= (img_thresh * 0.95):
            if (
                (sat_pixel_count >= 60000 and slide_mean_gray >= 165.0 and sat_anom_mean >= 1.8)
                or (sat_pixel_count >= self.spark_min_pixel_count and sat_anom_mean >= 2.5)
                or (sat_pixel_count >= 10000 and sat_anom_mean >= 3.5)
            ):
                spark_detected = True

        if spark_detected:
            spark_details = {
                "pixel_count": sat_pixel_count,
                "mean_gray": slide_mean_gray,
                "sat_anom_mean": sat_anom_mean,
                "max_score": sat_anom_max,
                "roi": [rx1, ry1, rx2, ry2],
            }

        # -------------------------------------------------------------
        # 2. Camera Dirt (画面脏污) Detection: Multi-frame static persistence
        # 排除底部 OSD 水印(y > 0.82) 与 大火花放电滑板区域，避免火花误累积为脏污
        # -------------------------------------------------------------
        h_dirt_limit = int(h * 0.82)
        dirt_anom_clean = anomaly_map[:h_dirt_limit, :].copy()
        if spark_detected or sat_pixel_count >= 40000:
            dirt_anom_clean[ry1:min(h_dirt_limit, ry2), rx1:rx2] = 0.0

        gw, gh = self.dirt_grid_size, self.dirt_grid_size
        grid_anom = cv2.resize(dirt_anom_clean, (gw, gh), interpolation=cv2.INTER_AREA)
        grid_active = (grid_anom >= pix_thresh).astype(np.float32)

        if self.dirt_grid_history is None:
            self.dirt_grid_history = grid_active
            self.history_length = 1
        else:
            self.dirt_grid_history = 0.85 * self.dirt_grid_history + 0.15 * grid_active
            self.history_length += 1

        dirt_detected = False
        dirt_pending = False
        dirt_details = {}
        if self.history_length >= self.dirt_persistence_frames:
            persistent_mask = self.dirt_grid_history >= 0.60
            dirt_count = int(np.sum(persistent_mask))
            if dirt_count >= 6 and img_anomaly_score > (img_thresh * 0.90) and not spark_detected:
                dirt_detected = True
                dirt_details = {
                    "persistent_grid_cells": dirt_count,
                    "max_history_val": float(np.max(self.dirt_grid_history)),
                    "state": "confirmed",
                }
        else:
            # 脏污候选观察态 (Pending State):
            # 在时序累积未满 dirt_persistence_frames (前 1~3 帧) 时，如果单帧网格激活数显著异常 (>= dirt_pending_cells_threshold)
            # 且整体异常分超标，标记为脏污观察态，强制冻结/抑制向下穿透到结构异常与悬挂异物
            active_grid_count = int(np.sum(grid_active))
            if (
                active_grid_count >= self.dirt_pending_cells_threshold
                and img_anomaly_score > (img_thresh * 0.90)
                and not spark_detected
            ):
                dirt_pending = True
                dirt_details = {
                    "active_grid_cells": active_grid_count,
                    "history_length": self.history_length,
                    "state": "pending",
                }

        # -------------------------------------------------------------
        # 3. Foreign Object (悬挂异物) Detection: Catenary area isolated structural anomaly
        # -------------------------------------------------------------
        cx1, cy1, cx2, cy2 = (
            int(self.catenary_roi_ratio[0] * w),
            int(self.catenary_roi_ratio[1] * h),
            int(self.catenary_roi_ratio[2] * w),
            int(self.catenary_roi_ratio[3] * h),
        )

        catenary_anom = anomaly_map[cy1:cy2, cx1:cx2].copy()
        foreign_bin = (catenary_anom >= pix_thresh * 1.10).astype(np.uint8) * 255
        f_contours, _ = cv2.findContours(
            foreign_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        foreign_detected = False
        foreign_boxes = []

        # Dense truss rejection: if too many blobs appear simultaneously, it's a station ceiling
        # 注意：大火花放电强辉光或处于脏污/脏污观察态时，抑制异物检测误报
        if (
            not dirt_detected
            and not dirt_pending
            and not spark_detected
            and len(f_contours) <= self.max_blobs_filter
        ):
            for cnt in f_contours:
                area = cv2.contourArea(cnt)
                if area >= self.foreign_min_area:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    foreign_boxes.append(
                        [cx1 + bx, cy1 + by, cx1 + bx + bw, cy1 + by + bh]
                    )

            if len(foreign_boxes) > 0 and img_anomaly_score > (img_thresh * 0.90):
                foreign_detected = True

        # -------------------------------------------------------------
        # 4. Structural Anomaly (受电弓结构异常) Detection: Concrete frame deformation
        # -------------------------------------------------------------
        px1, py1, px2, py2 = (
            int(self.pantograph_roi_ratio[0] * w),
            int(self.pantograph_roi_ratio[1] * h),
            int(self.pantograph_roi_ratio[2] * w),
            int(self.pantograph_roi_ratio[3] * h),
        )
        panto_anom = anomaly_map[py1:py2, px1:px2].copy()
        if spark_detected:
            panto_gray = gray[py1:py2, px1:px2]
            panto_anom[panto_gray >= self.spark_brightness_threshold] = 0.0

        struct_bin = (panto_anom >= pix_thresh * 1.25).astype(np.uint8) * 255
        s_contours, _ = cv2.findContours(
            struct_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        structural_detected = False
        structural_boxes = []
        # 注意：大火花放电强辉光、处于脏污或脏污观察态时，严禁向下穿透误报结构异常
        if not dirt_detected and not dirt_pending and not spark_detected:
            for cnt in s_contours:
                area = cv2.contourArea(cnt)
                if area >= self.structural_min_area:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    structural_boxes.append(
                        [px1 + bx, py1 + by, px1 + bx + bw, py1 + by + bh]
                    )

            if len(structural_boxes) > 0 and img_anomaly_score > (img_thresh * 1.15):
                structural_detected = True

        # -------------------------------------------------------------
        # 5. Summary and Decoupled Fault Decision
        # -------------------------------------------------------------
        faults_found = []
        # 1. Contact strip arc discharge (spark)
        # 核心修正：大火花产生瞬间强电弧，具有最高物理诊断优先级！
        if spark_detected:
            faults_found.append("大火花")
        # 2. Persistent camera dirt (camera lens stain, only if NOT active spark)
        if dirt_detected and not spark_detected:
            faults_found.append("画面脏污")
        elif dirt_pending and not spark_detected:
            faults_found.append("画面脏污(观察中)")
        # 3. Structural breakdown (severe shape deformation in frame/base, excluding spark flash and dirt/dirt_pending)
        if (
            structural_detected
            and not spark_detected
            and not dirt_detected
            and not dirt_pending
        ):
            faults_found.append("受电弓结构异常")
        # 4. Foreign object hanging on catenary (excluding spark flash and dirt/dirt_pending)
        if (
            foreign_detected
            and not spark_detected
            and not dirt_detected
            and not dirt_pending
        ):
            faults_found.append("悬挂异物")

        # Confident Fallback: If no specific physical defect is corroborated, fallback to 正常!
        if faults_found:
            has_fault = True
            primary_fault = f"异常-{faults_found[0]}"
        else:
            has_fault = False
            primary_fault = "正常"

        return {
            "has_fault": has_fault,
            "primary_fault": primary_fault,
            "faults": faults_found,
            "image_anomaly_score": img_anomaly_score,
            "image_anomaly_threshold": img_thresh,
            "timestamp": timestamp,
            "spark": {"detected": spark_detected, "details": spark_details},
            "dirt": {
                "detected": dirt_detected,
                "pending": dirt_pending,
                "details": dirt_details,
            },
            "foreign_object": {"detected": foreign_detected, "bboxes": foreign_boxes},
            "structural": {"detected": structural_detected, "bboxes": structural_boxes},
        }

    def annotate_frame(
        self,
        frame: np.ndarray,
        fault_result: dict,
        overlay_heatmap: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Draws diagnostic annotations and status overlay onto the frame.
        """
        out = overlay_heatmap.copy() if overlay_heatmap is not None else frame.copy()
        h, w = out.shape[:2]

        primary = fault_result["primary_fault"]
        score = fault_result["image_anomaly_score"]
        thresh = fault_result.get(
            "image_anomaly_threshold", fault_result.get("image_threshold", 4.0)
        )
        t_sec = fault_result.get("timestamp", 0.0)

        # Header banner (widened to 530px to comfortably fit threshold and score)
        color = (0, 0, 255) if fault_result["has_fault"] else (0, 200, 0)
        cv2.rectangle(out, (10, 10), (530, 85), (20, 20, 20), -1)
        cv2.rectangle(out, (10, 10), (530, 85), color, 2)

        # High-performance native Chinese text rendering
        chinese_font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
        if not os.path.exists(chinese_font_path):
            chinese_font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

        from PIL import Image, ImageDraw, ImageFont

        try:
            pil_font_title = ImageFont.truetype(chinese_font_path, 22)
            pil_font_sub = ImageFont.truetype(chinese_font_path, 16)
        except Exception:
            pil_font_title = ImageFont.load_default()
            pil_font_sub = ImageFont.load_default()

        # Convert to RGB PIL Image
        img_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Title and stats with crisp Chinese fonts & explicit threshold display
        title_color = (255, 60, 60) if fault_result["has_fault"] else (60, 230, 60)
        draw.text(
            (20, 20),
            f"受电弓故障监测: {primary}",
            fill=title_color,
            font=pil_font_title,
        )
        draw.text(
            (20, 52),
            f"异常评分: {score:.2f} (阈值: {thresh:.2f}) | 时间: {t_sec:.1f}s",
            fill=(230, 230, 230),
            font=pil_font_sub,
        )

        # Draw foreign object / structural anomaly bboxes (only if not dirt and not dirt_pending)
        dirt_info = fault_result.get("dirt", {})
        if not dirt_info.get("detected", False) and not dirt_info.get("pending", False):
            for bbox in fault_result.get("foreign_object", {}).get("bboxes", []):
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=2)
                draw.text(
                    (x1, max(5, y1 - 20)),
                    "悬挂异物",
                    fill=(255, 255, 0),
                    font=pil_font_sub,
                )

            # Draw structural anomaly bboxes
            for bbox in fault_result.get("structural", {}).get("bboxes", []):
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], outline=(255, 50, 50), width=2)
                draw.text(
                    (x1, max(5, y1 - 20)),
                    "结构异常",
                    fill=(255, 50, 50),
                    font=pil_font_sub,
                )

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
