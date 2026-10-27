#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAD Pantograph Structural Anomaly Detector Plugin.
Implements BaseDetector with dynamic pantograph bounding-box spatial masking.
"""

from typing import Any, Optional, Dict, List
import numpy as np
import cv2
import logging
from core.data_types import DetectionResult, DetectionContext
from core.detectors.fastad_base import FastADBaseDetector
from core.detectors.registry import DetectorRegistry

logger = logging.getLogger(__name__)


@DetectorRegistry.register("FastadStructDetector")
class FastadStructDetector(FastADBaseDetector):
    name: str = "struct_fastad"
    fault_type: str = "structure_abnormal"
    input_size: tuple = (384, 384)

    def initialize(self, config: Dict[str, Any]) -> None:
        model_path = config.get("model_path", "models/fastad_384.pt")
        self.min_defect_area = config.get("min_defect_area", 250)
        self.pantograph_roi_ratio = config.get("pantograph_roi_ratio", config.get("fallback_roi_ratio", [0.15, 0.15, 0.85, 0.70]))
        self.img_threshold = config.get("img_threshold", None)
        self.pixel_threshold = config.get("pixel_threshold", None)

        # 使用共享模型注册表加载（同路径只加载一次 GPU 权重）
        self.model = self._load_shared_model(model_path, config)

    def postprocess(self, raw_output: Any, context: Optional[DetectionContext] = None) -> DetectionResult:
        # 视场偏转防御拦截：画面锐利但无合法受电弓，彻底静默结构异常检测
        if context is not None and getattr(context, "is_camera_offset", False):
            return DetectionResult(
                fault_type=self.fault_type,
                is_abnormal=False,
                score=0.0,
                level="normal",
                bbox=None,
                extra={"status": "camera_offset_suppressed", "defect_count": 0, "bboxes": []}
            )

        fastad_res = raw_output
        anomaly_map = fastad_res["anomaly_map"]
        img_score = float(fastad_res["anomaly_score"])
        pix_thresh = float(self.pixel_threshold if self.pixel_threshold is not None else fastad_res.get("pixel_threshold", fastad_res.get("calibrated_pixel_threshold", 2.8)))
        img_thresh = float(self.img_threshold if self.img_threshold is not None else fastad_res.get("image_threshold", fastad_res.get("calibrated_image_threshold", 4.0)))

        h, w = anomaly_map.shape[:2]

        # 1. 严格使用 YOLO 检出的受电弓目标框，不使用原版静态几何 ROI，不进行任何外扩
        panto_bbox = context.panto_bbox if (context and context.panto_bbox) else None
        if not panto_bbox:
            # 视野中未定位到受电弓目标（如 dirty_2 严重脏污），安全返回正常，杜绝误报
            return DetectionResult(
                fault_type=self.fault_type,
                is_abnormal=False,
                score=0.0,
                level="normal",
                bbox=None,
                extra={"defect_count": 0, "bboxes": []}
            )

        px1, py1, px2, py2 = panto_bbox
        px1 = max(0, min(w, int(px1)))
        py1 = max(0, min(h, int(py1)))
        px2 = max(0, min(w, int(px2)))
        py2 = max(0, min(h, int(py2)))

        if px2 <= px1 or py2 <= py1:
            return DetectionResult(
                fault_type=self.fault_type,
                is_abnormal=False,
                score=0.0,
                level="normal",
                bbox=None,
                extra={"defect_count": 0, "bboxes": []}
            )

        panto_anom = anomaly_map[py1:py2, px1:px2].copy()

        # 2. 扣除火花强放电区 (避免强白炽反光穿透)
        if context is not None and context.spark_bboxes:
            for sb in context.spark_bboxes:
                sx1, sy1, sx2, sy2 = sb
                rx1 = max(0, sx1 - px1)
                ry1 = max(0, sy1 - py1)
                rx2 = min(px2 - px1, sx2 - px1)
                ry2 = min(py2 - py1, sy2 - py1)
                if rx2 > rx1 and ry2 > ry1:
                    panto_anom[ry1:ry2, rx1:rx2] = 0.0

        struct_bin = (panto_anom >= pix_thresh * 1.25).astype(np.uint8) * 255
        contours, _ = cv2.findContours(struct_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        structural_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_defect_area:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                structural_boxes.append([px1 + cx, py1 + cy, px1 + cx + cw, py1 + cy + ch])

        is_abnormal = len(structural_boxes) > 0 and (img_score > img_thresh * 1.15)
        level = "critical" if is_abnormal else "normal"
        norm_score = min(1.0, round(img_score / (img_thresh * 1.5), 3)) if is_abnormal else 0.0

        return DetectionResult(
            fault_type=self.fault_type,
            is_abnormal=is_abnormal,
            score=norm_score,
            level=level,
            bbox=structural_boxes[0] if structural_boxes else None,
            extra={
                "defect_count": len(structural_boxes),
                "bboxes": structural_boxes,
                "image_score": round(img_score, 2),
                "image_thresh": round(img_thresh, 2),
            }
        )
