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
from core.detectors.fastad_base import FastADBaseDetector
from core.detectors.registry import DetectorRegistry

logger = logging.getLogger(__name__)


@DetectorRegistry.register("FastadDirtDetector")
class FastadDirtDetector(FastADBaseDetector):
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

        # 使用共享模型注册表加载（同路径只加载一次 GPU 权重）
        self.model = self._load_shared_model(model_path)

        self.dirt_grid_history: Optional[np.ndarray] = None
        self.history_length = 0

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

        # 自适应开阔视场校验 (Adaptive Sky/Aperture Check)
        sky_cells = 0
        panto_bbox = context.panto_bbox if (context and context.panto_bbox) else None
        if panto_bbox:
            # 正常检出受电弓时：以弓头顶端以上的纯天空视场作为基准
            top_grid = max(1, int(panto_bbox[1] / h_dirt_limit * gh))
            sky_cells = int(np.sum(grid_active[:top_grid, :]))

            # 若受电弓存在但上方天空极其洁净 (sky_cells < 4)，说明异常严格限制在构件内部，绝非镜头光学污染
            if sky_cells < 4:
                self.dirt_grid_history = None
                self.history_length = 0
                return DetectionResult(
                    fault_type=self.fault_type,
                    is_abnormal=False,
                    score=0.0,
                    level="normal",
                    bbox=None,
                    extra={"sky_cells": sky_cells, "is_pending": False, "history_length": 0}
                )
        else:
            # 严重脏污场景 (如 dirty_2.mp4 泥浆糊死镜头致 YOLO 漏检受电弓)：
            # 回退采用车顶固定物理机位先验：顶部 25% 网格行 (0~16行) 恒定为高空开阔视域
            sky_cells = int(np.sum(grid_active[:int(gh * 0.25), :]))

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
                "sky_cells": sky_cells,
                "image_score": round(img_score, 2),
                "image_thresh": round(img_thresh, 2),
            }
        )

    def reset(self) -> None:
        self.dirt_grid_history = None
        self.history_length = 0
