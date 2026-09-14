#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detectors package.
Exposes BaseDetector, DetectorRegistry, and all registered detector plugins.
"""

from core.detectors.base import BaseDetector
from core.detectors.registry import DetectorRegistry
from core.detectors.spark_detector import YoloSparkDetector
from core.detectors.tilt_detector import HrnetTiltDetector
from core.detectors.dirt_detector import FastadDirtDetector
from core.detectors.float_detector import YoloFloatDetector
from core.detectors.struct_detector import FastadStructDetector
from core.detectors.foreign_detector import FastadForeignDetector

__all__ = [
    "BaseDetector",
    "DetectorRegistry",
    "YoloSparkDetector",
    "HrnetTiltDetector",
    "FastadDirtDetector",
    "YoloFloatDetector",
    "FastadStructDetector",
    "FastadForeignDetector",
]
