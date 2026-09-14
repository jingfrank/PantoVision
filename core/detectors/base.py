#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BaseDetector Interface Specification.
Every detector plugin (YOLO, FastAD, HRNet) must inherit from BaseDetector
and implement preprocess, infer, and postprocess methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
import logging
import time
from core.data_types import DetectionResult, DetectionContext

logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """
    统一检测器抽象基类 (Universal Detector Abstract Base).
    采用模板方法模式 (Template Method Pattern) 与适配器模式 (Adapter Pattern)。
    """
    name: str = "base_detector"           # 唯一插件标识, 如 "spark_yolo", "tilt_hrnet"
    fault_type: str = "unknown"           # 负责的故障类别
    input_size: tuple = (640, 640)        # 模型期望输入尺寸 (H, W)
    enabled: bool = True                  # 启停开关

    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        if name:
            self.name = name
        self.config = config or {}
        self.initialize(self.config)

    def initialize(self, config: Dict[str, Any]) -> None:
        """子类可选覆盖：加载模型权重与配置"""
        pass

    @abstractmethod
    def preprocess(self, image: Any, context: Optional[DetectionContext] = None) -> Any:
        """
        数据前处理：将原始 BGR 图像转化为模型输入张量或裁剪 ROI
        """
        pass

    @abstractmethod
    def infer(self, tensor: Any) -> Any:
        """
        前向推理：调用底层核心网络 (YOLO / HRNet / FastAD) 执行推理
        """
        pass

    @abstractmethod
    def postprocess(self, raw_output: Any, context: Optional[DetectionContext] = None) -> DetectionResult:
        """
        后处理：将网络原始输出解码为统一的 DetectionResult 标准结果
        """
        pass

    def run(self, image: Any, context: Optional[DetectionContext] = None) -> DetectionResult:
        """
        统一执行模板方法，内置【失败隔离沙箱】(Failure Isolation Sandbox)。
        任何单模型异常自动被捕获并标记为 level='unknown'，绝不向外渗透崩溃整个系统。
        """
        if not self.enabled:
            return DetectionResult(
                fault_type=self.fault_type,
                is_abnormal=False,
                score=0.0,
                level="normal",
                extra={"status": "disabled"}
            )

        try:
            t0 = time.perf_counter()
            x = self.preprocess(image, context=context)
            y = self.infer(x)
            result = self.postprocess(y, context=context)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            result.extra["latency_ms"] = round(latency_ms, 2)
            return result
        except Exception as exc:
            logger.error(f"[FailureIsolation][{self.name}] 推理管道异常被沙箱安全拦截: {exc}", exc_info=True)
            return DetectionResult(
                fault_type=self.fault_type,
                is_abnormal=False,
                score=0.0,
                level="unknown",
                extra={"error": str(exc), "status": "failed"}
            )

    def reset(self) -> None:
        """重置内部时序寄存器与缓存"""
        pass
