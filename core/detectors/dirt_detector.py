#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAD Temporal Grid Lens Dirt Detector Plugin.
Implements BaseDetector for camera-coordinate static persistent stain detection.
"""

from typing import Any, Optional, Dict
import numpy as np
import cv2
import logging
from core.data_types import DetectionResult, DetectionContext
from core.detectors.base import BaseDetector
from core.detectors.registry import DetectorRegistry
from fastad_detector import PantographFastADModel

logger = logging.getLogger(__name__)


@DetectorRegistry.register("FastadDirtDetector")
class FastadDirtDetector(BaseDetector):
    name: str = "dirt_fastad"
    fault_type: str = "dirty"
    input_size: tuple = (384, 384)

    def initialize(self, config: Dict[str, Any]) -> None:
        model_path = config.get("model_path", "models/fastad_384.pt")
        self.grid_size = config.get("grid_size", 64)
        self.ema_decay = config.get("ema_decay", 0.85)
        self.persistence_frames = config.get("persistence_frames", 4)
        self.min_active_cells = config.get("min_active_cells", 6)
        self.dirt_score_thresh = config.get("dirt_score_thresh", 2.0)

        logger.info(f"[{self.name}] 正在加载 FastAD 脏污模型权重: {model_path} ...")
        self.model = PantographFastADModel.load(model_path)
        logger.info(f"[{self.name}] FastAD 脏污模型加载成功")

        self.dirt_grid_history: Optional[np.ndarray] = None
        self.history_length = 0

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
        # 排除底部 OSD 水印区 (y > 0.82)
        h_dirt_limit = int(h * 0.82)
        dirt_clean = anomaly_map[:h_dirt_limit, :].copy()

        # 若上下文存在大火花放电，扣除放电强辉光区，避免火花累加为脏污
        if context is not None and context.spark_bboxes:
            for sb in context.spark_bboxes:
                sx1, sy1, sx2, sy2 = sb
                dirt_clean[max(0, sy1):min(h_dirt_limit, sy2), max(0, sx1):min(w, sx2)] = 0.0

        gw, gh = self.grid_size, self.grid_size
        grid_anom = cv2.resize(dirt_clean, (gw, gh), interpolation=cv2.INTER_AREA)
        grid_active = (grid_anom >= pix_thresh).astype(np.float32)

        # 时序 EMA 衰减累积 (网格坐标系绝对静止滤波)
        if self.dirt_grid_history is None:
            self.dirt_grid_history = grid_active
            self.history_length = 1
        else:
            self.dirt_grid_history = self.ema_decay * self.dirt_grid_history + (1.0 - self.ema_decay) * grid_active
            self.history_length += 1

        is_dirt_confirmed = False
        is_dirt_pending = False
        persistent_count = 0

        single_img_mode = bool(context and context.shared_data and context.shared_data.get("single_image_mode", False))
        if single_img_mode:
            # 单张静态图片缺乏时序多帧校验条件，不误触发脏污观察态与确诊态 (对齐 test_structural_anomaly bypass_dirt_pending)
            is_dirt_confirmed = False
            is_dirt_pending = False
        elif self.history_length >= self.persistence_frames:
            persistent_mask = self.dirt_grid_history >= 0.60
            persistent_count = int(np.sum(persistent_mask))
            if persistent_count >= self.min_active_cells and img_score > (img_thresh * 0.90):
                is_dirt_confirmed = True
        else:
            active_count = int(np.sum(grid_active))
            if active_count >= 15 and img_score > (img_thresh * 0.90):
                is_dirt_pending = True

        level = "warning" if is_dirt_confirmed else ("warning" if is_dirt_pending else "normal")
        norm_score = min(1.0, round(persistent_count / 20.0, 3)) if is_dirt_confirmed else 0.0

        return DetectionResult(
            fault_type=self.fault_type,
            is_abnormal=is_dirt_confirmed,
            score=norm_score,
            level=level,
            bbox=None,
            extra={
                "persistent_cells": persistent_count,
                "history_length": self.history_length,
                "is_pending": is_dirt_pending,
                "image_score": round(img_score, 2),
                "image_thresh": round(img_thresh, 2),
            }
        )

    def reset(self) -> None:
        self.dirt_grid_history = None
        self.history_length = 0
