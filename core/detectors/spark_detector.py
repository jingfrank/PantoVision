#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO-based Big Spark Detector Plugin.
Implements BaseDetector for high-speed (30 FPS) pantograph arc discharge detection.
"""

from typing import Any, Optional, Dict, List
import numpy as np
import logging
from ultralytics import YOLO

from core.data_types import DetectionResult, DetectionContext
from core.detectors.base import BaseDetector
from core.detectors.registry import DetectorRegistry

logger = logging.getLogger(__name__)


@DetectorRegistry.register("YoloSparkDetector")
class YoloSparkDetector(BaseDetector):
    name: str = "spark_yolo"
    fault_type: str = "big_spark"
    input_size: tuple = (640, 640)

    def initialize(self, config: Dict[str, Any]) -> None:
        model_path = config.get("model_path", "models/best.pt")
        self.conf_thresh = config.get("conf_thresh", 0.30)
        self.spark_area_ratio = config.get("spark_area_ratio", 0.05)
        self.spark_min_area = config.get("spark_min_area", 400)
        self.spark_duration_frames = config.get("spark_duration_frames", 12)  # ~500-700ms at 25fps

        logger.info(f"[{self.name}] 正在加载 YOLO 权重: {model_path} ...")
        self.model = YOLO(model_path)
        logger.info(f"[{self.name}] YOLO 加载成功, 类别: {self.model.names}")

        self.current_duration_count = 0
        self.alarmed = False

    def preprocess(self, image: Any, context: Optional[DetectionContext] = None) -> Any:
        return image

    def infer(self, tensor: Any) -> Any:
        # Ultralytics 直接支持 numpy 图像输入
        return self.model(tensor, conf=self.conf_thresh, verbose=False)

    def postprocess(self, raw_output: Any, context: Optional[DetectionContext] = None) -> DetectionResult:
        res = raw_output[0] if isinstance(raw_output, list) else raw_output
        panto_boxes = []
        spark_boxes = []
        max_spark_conf = 0.0

        for box in res.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = [int(v) for v in box.xyxy[0].tolist()]

            if cls_id == 1:  # pantograph
                panto_boxes.append((xyxy, conf))
            elif cls_id == 2:  # spark
                spark_boxes.append((xyxy, conf))
                if conf > max_spark_conf:
                    max_spark_conf = conf

        # 1. 提取并共享受电弓目标框至上下文黑板 (Blackboard Export)
        best_panto_box = None
        if panto_boxes:
            panto_boxes.sort(key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]), reverse=True)
            best_panto_box = panto_boxes[0][0]
            if context is not None:
                context.panto_bbox = best_panto_box
                context.panto_confidence = panto_boxes[0][1]

        # 降弓工况门控：副弓脱网无受流电流，清空火花时序计数器，静默大火花告警；
        # 但保留 YOLO 提取的受电弓目标框供后续 YoloFloatDetector 复用，避免重复推理
        is_panto_raised = context.packet.is_panto_raised if (context and context.packet) else True
        if not is_panto_raised:
            self.current_duration_count = 0
            self.alarmed = False
            return DetectionResult(
                fault_type=self.fault_type,
                is_abnormal=False,
                score=0.0,
                level="normal",
                bbox=best_panto_box,
                extra={
                    "panto_bbox": best_panto_box,
                    "is_panto_raised": False,
                    "status": "panto_lowered_muted",
                    "spark_count": 0,
                }
            )

        # 2. 火花空间尺度与持续时间判决
        has_spark_target = False
        target_spark_bbox = None

        if spark_boxes:
            panto_area = (
                (best_panto_box[2] - best_panto_box[0]) * (best_panto_box[3] - best_panto_box[1])
                if best_panto_box else 1.0
            )

            for s_box, s_conf in spark_boxes:
                s_area = (s_box[2] - s_box[0]) * (s_box[3] - s_box[1])
                # 空间面积尺度比门限
                if s_area >= self.spark_min_area or (s_area / panto_area) >= self.spark_area_ratio:
                    has_spark_target = True
                    target_spark_bbox = s_box
                    break

        if has_spark_target:
            self.current_duration_count += 1
        else:
            self.current_duration_count = max(0, self.current_duration_count - 1)

        single_img_mode = bool(context and context.shared_data and context.shared_data.get("single_image_mode", False))
        if single_img_mode:
            is_abnormal = has_spark_target
            level = "critical" if is_abnormal else "normal"
        else:
            is_abnormal = self.current_duration_count >= self.spark_duration_frames
            level = "critical" if is_abnormal else ("warning" if self.current_duration_count >= 5 else "normal")

        if context is not None and spark_boxes:
            context.spark_bboxes = [b[0] for b in spark_boxes]

        return DetectionResult(
            fault_type=self.fault_type,
            is_abnormal=is_abnormal,
            score=round(max_spark_conf, 3),
            level=level,
            bbox=target_spark_bbox or (spark_boxes[0][0] if spark_boxes else None),
            extra={
                "duration_frames": self.current_duration_count,
                "duration_thresh": self.spark_duration_frames,
                "panto_bbox": best_panto_box,
                "spark_count": len(spark_boxes),
            }
        )

    def reset(self) -> None:
        self.current_duration_count = 0
        self.alarmed = False
