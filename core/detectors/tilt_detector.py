#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRNet 4-Keypoint Pantograph Tilt Detector Plugin.
Implements BaseDetector using local support baseline geometric decoupling.
"""

from typing import Any, Optional, Dict
import logging
from core.data_types import DetectionResult, DetectionContext
from core.detectors.base import BaseDetector
from core.detectors.registry import DetectorRegistry
from keypoint_tilt.detector import PantographTiltDetector

logger = logging.getLogger(__name__)


@DetectorRegistry.register("HrnetTiltDetector")
class HrnetTiltDetector(BaseDetector):
    name: str = "tilt_hrnet"
    fault_type: str = "tilt"
    input_size: tuple = (288, 384)

    def initialize(self, config: Dict[str, Any]) -> None:
        weights_path = config.get("weights_path", "models/best_hrnet_w32_panto.pth")
        engine_path = config.get("engine_path", None)
        prefer_trt = config.get("prefer_tensorrt", True)
        width = config.get("width", 32)
        target_size = tuple(config.get("target_size", (288, 384)))
        heatmap_size = tuple(config.get("heatmap_size", (72, 96)))
        warning_thresh = config.get("structural_warning_thresh", 1.2)
        alarm_thresh = config.get("structural_alarm_thresh", 1.8)
        install_bias = config.get("install_bias_deg", 0.0)

        logger.info(
            f"[{self.name}] 正在初始化 HRNet 倾斜检测器 (weights={weights_path}, "
            f"engine={engine_path}, prefer_trt={prefer_trt}) ..."
        )
        self.detector = PantographTiltDetector(
            weights_path=weights_path,
            engine_path=engine_path,
            prefer_tensorrt=prefer_trt,
            width=width,
            num_joints=4,
            target_size=target_size,
            heatmap_size=heatmap_size,
            structural_warning_thresh=warning_thresh,
            structural_alarm_thresh=alarm_thresh,
            install_bias_deg=install_bias,
        )
        self.is_trt = getattr(self.detector, "is_trt", False)
        backend_str = "TensorRT FP16" if self.is_trt else "PyTorch"
        logger.info(
            f"[{self.name}] HRNet 倾斜检测器初始化成功 [后端: {backend_str}] "
            f"(门限: 预警={warning_thresh}°, 告警={alarm_thresh}°)"
        )

    def preprocess(self, image: Any, context: Optional[DetectionContext] = None) -> Any:
        return image

    def infer(self, tensor: Any) -> Any:
        # 委托 PantographTiltDetector 前向推理与几何解算
        return self.detector.predict(tensor)

    def postprocess(self, raw_output: Any, context: Optional[DetectionContext] = None) -> DetectionResult:
        res, vis_img = raw_output
        confs = res.get("confidences", [])
        min_conf = min(confs) if confs else 1.0
        if min_conf < 0.40:
            return DetectionResult(
                fault_type=self.fault_type,
                is_abnormal=False,
                score=0.0,
                level="normal",
                bbox=None,
                keypoints=res.get("keypoints", []),
                extra={"status": "LOW_CONFIDENCE", "min_conf": round(min_conf, 3)}
            )

        status = res.get("status", "NORMAL")
        rel_tilt = float(res.get("relative_tilt_deg", 0.0))
        is_tilt_fault = status in ("ALARM_TILT", "WARNING_TILT")
        is_abnormal = is_tilt_fault

        if status == "ALARM_TILT":
            level = "critical"
        elif status == "WARNING_TILT":
            level = "warning"
        else:
            level = "normal"

        kpts = res.get("keypoints", [])
        norm_score = min(1.0, round(abs(rel_tilt) / 5.0, 3))

        return DetectionResult(
            fault_type=self.fault_type,
            is_abnormal=is_abnormal,
            score=norm_score,
            level=level,
            bbox=None,
            keypoints=kpts,
            extra={
                "relative_tilt_deg": rel_tilt,
                "tilt_direction": res.get("tilt_direction", "PARALLEL"),
                "status": status,
                "support_angle_deg": res.get("support_angle_deg", 0.0),
                "horn_angle_deg": res.get("horn_angle_deg", 0.0),
                "sustained_alarm": res.get("sustained_alarm", False),
            }
        )

    def reset(self) -> None:
        if hasattr(self, "detector") and hasattr(self.detector, "analyzer"):
            self.detector.analyzer.history.clear()
            self.detector.analyzer.alarmed = False
