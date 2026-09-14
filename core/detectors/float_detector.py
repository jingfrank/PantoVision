#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pantograph Abnormal Float Detector Plugin.
Implements BaseDetector for temporal trajectory displacement & TCMS pantograph status monitoring.
"""

from typing import Any, Optional, Dict
from collections import deque
import numpy as np
import logging
from core.data_types import DetectionResult, DetectionContext
from core.detectors.base import BaseDetector
from core.detectors.registry import DetectorRegistry

logger = logging.getLogger(__name__)


@DetectorRegistry.register("YoloFloatDetector")
class YoloFloatDetector(BaseDetector):
    name: str = "float_yolo"
    fault_type: str = "abnormal_float"
    input_size: tuple = (640, 640)

    def initialize(self, config: Dict[str, Any]) -> None:
        self.window_size = config.get("window_size", 30)
        self.float_speed_thresh = config.get("float_speed_thresh", 40.0)  # px/s 突变速度
        self.panto_down_height_limit = config.get("panto_down_height_limit", 70.0)  # 降弓状态高度限值
        self.consecutive_alarm_frames = config.get("consecutive_alarm_frames", 5)

        self.trajectory = deque(maxlen=self.window_size)
        self.abnormal_frame_count = 0

    def preprocess(self, image: Any, context: Optional[DetectionContext] = None) -> Any:
        # 优先直接读取黑板上下文中 YOLO 提取的受电弓框，零额外计算开销
        panto_box = context.panto_bbox if context is not None else None
        return panto_box

    def infer(self, tensor: Any) -> Any:
        # 直接透传受电弓框
        return tensor

    def postprocess(self, raw_output: Any, context: Optional[DetectionContext] = None) -> DetectionResult:
        panto_box = raw_output
        is_panto_raised = context.packet.is_panto_raised if (context and context.packet) else True
        timestamp = context.packet.timestamp if (context and context.packet) else 0.0

        is_float_candidate = False
        metrics = {}

        if panto_box is not None:
            px1, py1, px2, py2 = panto_box
            yc = (py1 + py2) / 2.0
            height = float(py2 - py1)

            # 工况 1：TCMS 指示处于降弓状态，但受电弓依然处于抬升位置
            if not is_panto_raised:
                if height > self.panto_down_height_limit:
                    is_float_candidate = True
                    metrics["reason"] = "down_state_raised"

            # 工况 2：正常运行受流中，垂向中心发生剧烈高频超限跳变
            else:
                if len(self.trajectory) >= 2:
                    prev_t, prev_yc = self.trajectory[-1]
                    dt = max(0.01, timestamp - prev_t)
                    speed = abs(yc - prev_yc) / dt
                    metrics["vertical_speed"] = round(speed, 1)
                    if speed > self.float_speed_thresh:
                        is_float_candidate = True
                        metrics["reason"] = "excessive_vertical_bounce"

            self.trajectory.append((timestamp, yc))

        if is_float_candidate:
            self.abnormal_frame_count += 1
        else:
            self.abnormal_frame_count = max(0, self.abnormal_frame_count - 1)

        is_abnormal = self.abnormal_frame_count >= self.consecutive_alarm_frames
        level = "critical" if is_abnormal else ("warning" if self.abnormal_frame_count >= 2 else "normal")

        return DetectionResult(
            fault_type=self.fault_type,
            is_abnormal=is_abnormal,
            score=1.0 if is_abnormal else 0.0,
            level=level,
            bbox=panto_box,
            extra={
                "abnormal_frame_count": self.abnormal_frame_count,
                "is_panto_raised": is_panto_raised,
                **metrics,
            }
        )

    def reset(self) -> None:
        self.trajectory.clear()
        self.abnormal_frame_count = 0
