#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detector Plugin Registry (注册表模式).
Enables dynamic discovery, declarative loading from YAML, and on/off switching.
"""

import logging
from typing import Dict, Type, Any, Optional
from core.detectors.base import BaseDetector

logger = logging.getLogger(__name__)


class DetectorRegistry:
    """全局检测器插件注册表"""
    _registry: Dict[str, Type[BaseDetector]] = {}

    @classmethod
    def register(cls, name: Optional[str] = None):
        """装饰器：注册检测器插件类"""
        def decorator(subclass: Type[BaseDetector]):
            key = name or subclass.__name__
            if key in cls._registry:
                logger.warning(f"[Registry] 插件 '{key}' 重复注册，将被覆盖: {subclass}")
            cls._registry[key] = subclass
            return subclass
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseDetector]:
        """获取已注册的检测器类"""
        if name not in cls._registry:
            raise KeyError(f"[Registry] 未注册的检测器类型: '{name}', 可用插件列表: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def build(cls, cfg: Dict[str, Any]) -> BaseDetector:
        """根据配置字典构建检测器实例"""
        plugin_type = cfg.get("type", "")
        name = cfg.get("name", plugin_type)
        detector_cls = cls.get(plugin_type)
        detector = detector_cls(name=name, config=cfg.get("params", {}))
        detector.enabled = cfg.get("enabled", True)
        detector.fault_type = cfg.get("fault_type", detector.fault_type)
        return detector

    @classmethod
    def list_all(cls) -> Dict[str, Type[BaseDetector]]:
        """列出所有已注册的检测器插件"""
        return dict(cls._registry)
