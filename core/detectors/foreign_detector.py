#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAD Catenary Open-Set Foreign Object Detector Plugin.
Implements BaseDetector for hanging debris on overhead catenary wires
with Dynamic YOLO Pantograph Guidance, Temporal BBox Hold, and Truncation Guard.
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
        self.min_foreign_area = config.get("min_foreign_area", 250)
        self.max_blobs_filter = config.get("max_blobs_filter", 12)
        self.pixel_threshold = config.get("pixel_threshold", None)
        self.img_threshold = config.get("img_threshold", None)
        self.panto_bbox_hold_sec = config.get("panto_bbox_hold_sec", 1.0)
        self.top_truncation_margin = config.get("top_truncation_margin", 25)

        self.last_valid_panto_bbox: Optional[List[int]] = None
        self.last_panto_ts: float = -1.0

        # 使用共享模型注册表加载（同路径只加载一次 GPU 权重）
        self.model = self._load_shared_model(model_path, config)

    def postprocess(self, raw_output: Any, context: Optional[DetectionContext] = None) -> DetectionResult:
        # 1. 视场偏转防御拦截：画面锐利但无合法受电弓，彻底静默悬挂异物检测
        if context is not None and getattr(context, "is_camera_offset", False):
            self.last_valid_panto_bbox = None
            self.last_panto_ts = -1.0
            return DetectionResult(
                fault_type=self.fault_type,
                is_abnormal=False,
                score=0.0,
                level="normal",
                bbox=None,
                extra={"status": "camera_offset_suppressed", "foreign_count": 0, "bboxes": []}
            )

        # 受电弓显示不全拦截 (监控不完整/残缺半弓/边缘截断)：严禁执行悬挂异物检测
        if context is not None and getattr(context, "is_panto_incomplete", False):
            self.last_valid_panto_bbox = None
            self.last_panto_ts = -1.0
            return DetectionResult(
                fault_type=self.fault_type,
                is_abnormal=False,
                score=0.0,
                level="normal",
                bbox=None,
                extra={"status": "panto_incomplete_suppressed", "foreign_count": 0, "bboxes": []}
            )

        # 2. 受电弓动态定位与时序平滑保持 (BBox Hold)
        current_ts = context.packet.timestamp if (context and context.packet) else 0.0
        panto_bbox = context.panto_bbox if (context and context.panto_bbox) else None

        if panto_bbox:
            self.last_valid_panto_bbox = panto_bbox
            self.last_panto_ts = current_ts
        elif (
            self.last_valid_panto_bbox is not None
            and (current_ts - self.last_panto_ts) <= self.panto_bbox_hold_sec
        ):
            # 偶发性单帧漏检，平滑复用最近有效受电弓位置
            panto_bbox = self.last_valid_panto_bbox

        if not panto_bbox:
            # 持续无受电弓定位（降弓/镜头偏离/严重遮挡）：坚决静默，避免对着车顶虚空报警
            return DetectionResult(
                fault_type=self.fault_type,
                is_abnormal=False,
                score=0.0,
                level="normal",
                bbox=None,
                extra={"status": "no_pantograph_suppressed", "foreign_count": 0, "bboxes": []}
            )

        fastad_res = raw_output
        anomaly_map = fastad_res["anomaly_map"]
        img_score = float(fastad_res["anomaly_score"])
        pix_thresh = float(self.pixel_threshold if self.pixel_threshold is not None else fastad_res.get("pixel_threshold", fastad_res.get("calibrated_pixel_threshold", 2.8)))
        img_thresh = float(self.img_threshold if self.img_threshold is not None else fastad_res.get("image_threshold", fastad_res.get("calibrated_image_threshold", 4.0)))

        h, w = anomaly_map.shape[:2]
        px1, py1, px2, py2 = panto_bbox
        px1 = max(0, min(w, int(px1)))
        py1 = max(0, min(h, int(py1)))
        px2 = max(0, min(w, int(px2)))
        py2 = max(0, min(h, int(py2)))

        pw = px2 - px1
        ph = py2 - py1

        # 3. 顶边截断盲区防护：受电弓露出一半、顶端贴边出画 (py1 < top_truncation_margin)
        #    真实接触网已飞出画面上沿盲区，强行向上扩展只会将顶部截断边缘当成异物误报
        if py1 < self.top_truncation_margin:
            return DetectionResult(
                fault_type=self.fault_type,
                is_abnormal=False,
                score=0.0,
                level="normal",
                bbox=None,
                extra={"status": "top_truncation_suppressed", "foreign_count": 0, "bboxes": []}
            )

        # 4. 动态计算接触网工作区 (Dynamic Catenary Working ROI)
        #    水平方向：在受电弓宽度基础上适度外扩 (5%)
        #    垂直方向：从受电弓顶端向上延伸 1.2 倍弓高作为高空接触网区域，底部延伸至碳滑板上工作面
        margin_x = min(60, max(20, int(pw * 0.05)))
        cx1 = max(0, px1 - margin_x)
        cx2 = min(w, px2 + margin_x)
        catenary_span_h = max(100, int(ph * 1.2))
        cy1 = max(0, py1 - catenary_span_h)
        cy2 = min(h, py1 + int(ph * 0.40))

        if cx2 <= cx1 or cy2 <= cy1:
            return DetectionResult(
                fault_type=self.fault_type,
                is_abnormal=False,
                score=0.0,
                level="normal",
                bbox=None,
                extra={"status": "invalid_dynamic_roi", "foreign_count": 0, "bboxes": []}
            )

        catenary_anom = anomaly_map[cy1:cy2, cx1:cx2].copy()

        # 5. 扣除火花强放电区 (避免强白炽反光穿透至接触网)
        if context is not None and context.spark_bboxes:
            for sb in context.spark_bboxes:
                sx1, sy1, sx2, sy2 = sb
                rx1 = max(0, sx1 - cx1)
                ry1 = max(0, sy1 - cy1)
                rx2 = min(cx2 - cx1, sx2 - cx1)
                ry2 = min(cy2 - cy1, sy2 - cy1)
                if rx2 > rx1 and ry2 > ry1:
                    catenary_anom[ry1:ry2, rx1:rx2] = 0.0

        foreign_bin = (catenary_anom >= pix_thresh * 1.25).astype(np.uint8) * 255
        f_contours, _ = cv2.findContours(foreign_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        foreign_boxes = []
        # 密集桁架过滤：若同时出现大量斑块，判定为站台钢架结构，予以剔除
        if len(f_contours) <= self.max_blobs_filter:
            for cnt in f_contours:
                area = cv2.contourArea(cnt)
                if area >= self.min_foreign_area:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    foreign_boxes.append([cx1 + bx, cy1 + by, cx1 + bx + bw, cy1 + by + bh])

        is_abnormal = len(foreign_boxes) > 0 and (img_score > img_thresh * 1.15)
        level = "critical" if is_abnormal else "normal"
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
                "dynamic_roi": [cx1, cy1, cx2, cy2],
                "image_score": round(img_score, 2),
                "image_thresh": round(img_thresh, 2),
            }
        )

    def reset(self) -> None:
        self.last_valid_panto_bbox = None
        self.last_panto_ts = -1.0
