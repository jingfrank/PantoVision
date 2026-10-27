#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAD Catenary Open-Set Foreign Object Detector Plugin.
Implements BaseDetector for hanging debris on overhead catenary wires.
"""

from typing import Any, Optional, Dict, List
import numpy as np
import cv2
import logging
from core.data_types import DetectionResult, DetectionContext
from core.detectors.fastad_base import FastADBaseDetector
from core.detectors.registry import DetectorRegistry

logger = logging.getLogger(__name__)


@DetectorRegistry.register("FastadForeignDetector")
class FastadForeignDetector(FastADBaseDetector):
    name: str = "foreign_fastad"
    fault_type: str = "foreign_object"
    input_size: tuple = (384, 384)

    def initialize(self, config: Dict[str, Any]) -> None:
        model_path = config.get("model_path", "models/fastad_384.pt")
        self.catenary_roi_ratio = config.get("catenary_roi_ratio", [0.25, 0.25, 0.85, 0.50])
        self.min_foreign_area = config.get("min_foreign_area", 100)
        self.max_blobs_filter = config.get("max_blobs_filter", 12)
        self.foreign_score_threshold = config.get("foreign_score_threshold", 2.2)
        self.pixel_threshold = config.get("pixel_threshold", None)
        self.img_threshold = config.get("img_threshold", None)

        # 使用共享模型注册表加载（同路径只加载一次 GPU 权重）
        self.model = self._load_shared_model(model_path, config)

    def postprocess(self, raw_output: Any, context: Optional[DetectionContext] = None) -> DetectionResult:
        # 视场偏转防御拦截：画面锐利但无合法受电弓，彻底静默悬挂异物检测
        if context is not None and getattr(context, "is_camera_offset", False):
            return DetectionResult(
                fault_type=self.fault_type,
                is_abnormal=False,
                score=0.0,
                level="normal",
                bbox=None,
                extra={"status": "camera_offset_suppressed", "foreign_count": 0, "bboxes": []}
            )

        fastad_res = raw_output
        anomaly_map = fastad_res["anomaly_map"]
        img_score = float(fastad_res["anomaly_score"])
        pix_thresh = float(self.pixel_threshold if self.pixel_threshold is not None else fastad_res.get("pixel_threshold", fastad_res.get("calibrated_pixel_threshold", 2.8)))
        img_thresh = float(self.img_threshold if self.img_threshold is not None else fastad_res.get("image_threshold", fastad_res.get("calibrated_image_threshold", 4.0)))

        h, w = anomaly_map.shape[:2]
        cx1 = int(self.catenary_roi_ratio[0] * w)
        cy1 = int(self.catenary_roi_ratio[1] * h)
        cx2 = int(self.catenary_roi_ratio[2] * w)
        cy2 = int(self.catenary_roi_ratio[3] * h)

        catenary_anom = anomaly_map[cy1:cy2, cx1:cx2].copy()
        foreign_bin = (catenary_anom >= pix_thresh * 1.10).astype(np.uint8) * 255
        f_contours, _ = cv2.findContours(foreign_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        foreign_boxes = []
        # 密集桁架过滤：若同时出现大量斑块，判定为站台钢架结构，予以剔除
        if len(f_contours) <= self.max_blobs_filter:
            for cnt in f_contours:
                area = cv2.contourArea(cnt)
                if area >= self.min_foreign_area:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    foreign_boxes.append([cx1 + bx, cy1 + by, cx1 + bx + bw, cy1 + by + bh])

        is_abnormal = len(foreign_boxes) > 0 and (img_score > img_thresh * 0.90)
        level = "warning" if is_abnormal else "normal"
        norm_score = min(1.0, round(img_score / (img_thresh * 1.5), 3)) if is_abnormal else 0.0

        return DetectionResult(
            fault_type=self.fault_type,
            is_abnormal=is_abnormal,
            score=norm_score,
            level=level,
            bbox=foreign_boxes[0] if foreign_boxes else None,
            extra={
                "foreign_count": len(foreign_boxes),
                "bboxes": foreign_boxes,
                "image_score": round(img_score, 2),
                "image_thresh": round(img_thresh, 2),
            }
        )
