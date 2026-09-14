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
from core.detectors.base import BaseDetector
from core.detectors.registry import DetectorRegistry
from fastad_detector import PantographFastADModel

logger = logging.getLogger(__name__)


@DetectorRegistry.register("FastadStructDetector")
class FastadStructDetector(BaseDetector):
    name: str = "struct_fastad"
    fault_type: str = "structure_abnormal"
    input_size: tuple = (384, 384)

    def initialize(self, config: Dict[str, Any]) -> None:
        model_path = config.get("model_path", "models/fastad_384.pt")
        self.min_defect_area = config.get("min_defect_area", 250)
        self.pantograph_roi_ratio = config.get("pantograph_roi_ratio", config.get("fallback_roi_ratio", [0.15, 0.15, 0.85, 0.70]))

        logger.info(f"[{self.name}] 正在加载 FastAD 结构异常模型: {model_path} ...")
        self.model = PantographFastADModel.load(model_path)
        logger.info(f"[{self.name}] FastAD 结构异常模型加载完成")

    def preprocess(self, image: Any, context: Optional[DetectionContext] = None) -> Any:
        return image

    def infer(self, tensor: Any) -> Any:
        return self.model.predict(tensor)

    def postprocess(self, raw_output: Any, context: Optional[DetectionContext] = None) -> DetectionResult:
        fastad_res = raw_output
        anomaly_map = fastad_res["anomaly_map"]
        img_score = float(fastad_res["anomaly_score"])
        pix_thresh = float(fastad_res.get("pixel_threshold", fastad_res.get("calibrated_pixel_threshold", 2.8)))
        img_thresh = float(fastad_res.get("image_threshold", fastad_res.get("calibrated_image_threshold", 4.0)))

        h, w = anomaly_map.shape[:2]

        # 1. 采用原版 PantographFaultClassifier 空间几何 ROI (0.15, 0.15, 0.85, 0.70)
        px1 = int(self.pantograph_roi_ratio[0] * w)
        py1 = int(self.pantograph_roi_ratio[1] * h)
        px2 = int(self.pantograph_roi_ratio[2] * w)
        py2 = int(self.pantograph_roi_ratio[3] * h)

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
