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

        # 1. 正常升弓状态（is_panto_raised = True）：
        #    受电弓处于接触网工作高度，属于正常工况，重置飘弓计数器，不触发异常飘弓告警
        if is_panto_raised:
            self.abnormal_frame_count = 0
            self.trajectory.clear()
            return DetectionResult(
                fault_type=self.fault_type,
                is_abnormal=False,
                score=0.0,
                level="normal",
                bbox=panto_box,
                extra={
                    "abnormal_frame_count": 0,
                    "is_panto_raised": True,
                    "reason": "normal_raised",
                },
            )

        # 2. 降弓状态（is_panto_raised = False）：
        #    若视觉检测到受电弓高度超过降弓高度限值（panto_down_height_limit），
        #    说明受电弓在降弓时发生风动漂浮或机械弹起，连续超标帧数后触发“异常飘弓故障”
        is_float_candidate = False
        metrics = {}

        if panto_box is not None:
            px1, py1, px2, py2 = panto_box
            yc = (py1 + py2) / 2.0
            height = float(py2 - py1)
            metrics["height"] = round(height, 1)

            if height > self.panto_down_height_limit:
                is_float_candidate = True
                metrics["reason"] = "down_state_raised"
            else:
                self.abnormal_frame_count = 0

            self.trajectory.append((timestamp, yc))
        else:
            self.abnormal_frame_count = 0

        if is_float_candidate:
            self.abnormal_frame_count += 1
        else:
            self.abnormal_frame_count = 0

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
                "is_panto_raised": False,
                "panto_down_height_limit": self.panto_down_height_limit,
                **metrics,
            },
        )

    def reset(self) -> None:
        self.trajectory.clear()
        self.abnormal_frame_count = 0
