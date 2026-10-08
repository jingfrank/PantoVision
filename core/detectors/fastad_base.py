#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAD Shared Base Detector (FastAD 系列检测器共用基类).

实现两层共享优化，在保持插件完全解耦的前提下消除冗余计算与显存浪费：
1. 模型实例共享：类级注册表确保同一 model_path 只加载一次 GPU 权重；
2. 推理结果缓存：通过 DetectionContext.shared_data 黑板模式，同帧内多个
   FastAD 检测器共享骨干推理结果（anomaly_map + anomaly_score）。
"""

import os
import logging
import threading
from typing import Any, Optional, Dict
from abc import abstractmethod

from core.data_types import DetectionResult, DetectionContext
from core.detectors.base import BaseDetector
from fastad_detector import PantographFastADModel

logger = logging.getLogger(__name__)


class FastADBaseDetector(BaseDetector):
    """
    FastAD 系列检测器的共用基类。

    子类只需实现：
    - initialize(config): 读取自身特有参数 (如 grid_size, roi_ratio 等)，
                          并调用 self._load_shared_model(model_path) 获取模型引用
    - postprocess(raw_output, context): 基于共享 anomaly_map 的独立后处理逻辑

    骨干推理与缓存管理由本基类统一处理。
    """

    # ── 类级模型实例注册表 (同路径只加载一次) ──────────────────
    _shared_models: Dict[str, PantographFastADModel] = {}
    _model_lock = threading.Lock()
    # ──────────────────────────────────────────────────────────

    def _load_shared_model(
        self, model_path: str, config: Optional[Dict[str, Any]] = None
    ) -> PantographFastADModel:
        """
        从类级注册表加载或复用 FastAD 模型实例。
        同一 model_path 只会执行一次 PantographFastADModel.load()，
        后续检测器直接复用已加载的 GPU 权重。
        """
        abs_path = os.path.abspath(model_path)
        with self._model_lock:
            if abs_path not in self._shared_models:
                cfg = config or {}
                prefer_trt = cfg.get("prefer_tensorrt", True)
                engine_path = cfg.get("engine_path", None)
                use_gpu_preproc = cfg.get("use_gpu_preprocess", True)
                logger.info(
                    f"[{self.name}] 首次加载 FastAD 共享模型: {model_path} (prefer_trt={prefer_trt}, gpu_preproc={use_gpu_preproc})"
                )
                self._shared_models[abs_path] = PantographFastADModel.load(
                    model_path,
                    prefer_tensorrt=prefer_trt,
                    engine_path=engine_path,
                    use_gpu_preprocess=use_gpu_preproc,
                )
                logger.info(f"[{self.name}] FastAD 共享模型加载成功")
            else:
                logger.info(
                    f"[{self.name}] 复用已加载的 FastAD 共享模型: {model_path}"
                )
        self._model_path_abs = abs_path
        return self._shared_models[abs_path]

    def preprocess(
        self, image: Any, context: Optional[DetectionContext] = None
    ) -> Any:
        """
        暂存 context 引用，供 infer() 执行黑板缓存查找。
        单流内主循环为单线程顺序执行，此模式线程安全。
        """
        self._current_context = context
        return image

    def infer(self, tensor: Any) -> Any:
        """
        智能推理：优先从黑板缓存读取同帧已有的 FastAD 结果，
        若缓存未命中则执行推理并写入缓存供后续检测器复用。
        """
        ctx = getattr(self, "_current_context", None)
        cache_key = f"fastad_result:{self._model_path_abs}"

        # 黑板缓存命中：直接复用，跳过 GPU 推理 (耗时 < 0.01ms)
        if ctx is not None and cache_key in ctx.shared_data:
            return ctx.shared_data[cache_key]

        # 缓存未命中：执行完整 FastAD 推理 (~155ms on Jetson)
        result = self.model.predict(tensor)

        # 写入黑板缓存，供同帧后续 FastAD 检测器复用
        if ctx is not None:
            ctx.shared_data[cache_key] = result

        return result

    @abstractmethod
    def postprocess(
        self, raw_output: Any, context: Optional[DetectionContext] = None
    ) -> DetectionResult:
        """各 FastAD 检测器独有的后处理逻辑 (ROI 裁剪、阈值判定等)"""
        ...
