#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Data Contracts for Pantograph AI Monitoring System.
Defines standard data classes passed between Ingestion, Scheduler, Detectors, Fusion, and Output.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Tuple
import numpy as np


@dataclass
class DetectionResult:
    """
    统一检测器输出规范 (Universal Detector Output Contract).
    不论底层是 YOLO、FastAD/PatchCore 还是 HRNet，所有检测器均返回该结构。
    """
    fault_type: str                 # "big_spark" / "tilt" / "dirty" / "foreign_object" / "structure_abnormal" / "abnormal_float"
    is_abnormal: bool               # 是否检出异常候选
    score: float                    # 统一置信度/异常分数 (0.0 ~ 1.0)
    level: str = "normal"           # "normal" / "warning" / "critical" / "unknown" —— 分级告警
    bbox: Optional[List[int]] = None     # [x1, y1, x2, y2] 目标框 (有目标框则填, 没有的留空如倾斜)
    keypoints: Optional[List[List[float]]] = None  # HRNet 专用 4 关键点 [[x, y, conf], ...]
    extra: Dict[str, Any] = field(default_factory=dict)  # 各模型私有信息 (角度值、网格数、原始诊断详情)


@dataclass
class FramePacket:
    """
    统一帧数据包 (Standardized Video Frame Packet).
    由数据接入层产生，传递给调度层与检测器。
    """
    frame: np.ndarray             # 原始 BGR 图像 (H, W, 3)
    timestamp: float              # 物理时间戳 (秒)
    frame_idx: int                # 单调递增帧索引
    car_number: int = 6           # 车厢号 (如 3 或 6)
    camera_id: int = 0            # 位置号 (如 0, 1, 2)
    is_panto_raised: bool = True  # TCMS 升降弓状态
    speed: float = 0.0            # 车速 (km/h)
    train_id: str = "CR400BF"     # 车次标识
    kilometer_post: Optional[int] = None  # 公里标 (米, 0xFFFFFFFF 为未知)


@dataclass
class DetectionContext:
    """
    共享检测上下文 (Blackboard Pattern).
    用于同一帧内各检测器之间共享中间计算结果 (例如 YOLO 检出的受电弓目标框供后续插件复用).
    """
    packet: FramePacket
    panto_bbox: Optional[List[int]] = None       # [px1, py1, px2, py2]
    panto_confidence: float = 0.0
    is_camera_offset: bool = False               # 视场偏转标记: 画面清晰锐利但无合法受电弓
    is_panto_incomplete: bool = False            # 受电弓显示不全标记: 紧贴边缘截断或残缺半弓
    top_lap_var: float = 0.0                     # 顶部高空视场拉普拉斯方差
    spark_bboxes: List[List[int]] = field(default_factory=list)
    shared_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionResult:
    """
    融合决策综合输出 (Unified Fusion & Diagnostic Decision).
    由结果融合层产出，驱动输出层与告警推送。
    """
    has_fault: bool                              # 全局是否存在任何异常
    primary_fault: str                           # 核心主告警名称 (如 "正常", "异常-大火花")
    active_alarms: List[Dict[str, Any]]          # 生效的告警清单 [{'fault_type': '大火花', 'level': 'critical', ...}]
    detector_results: Dict[str, DetectionResult] # 各插件的原始结果字典
    annotated_frame: Optional[np.ndarray] = None # 叠加画板与OSD后的输出图像
    timestamp: float = 0.0                       # 事件时间戳
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        if not self.active_alarms:
            return 0.0
        return max([float(a.get("score", 0.0)) for a in self.active_alarms], default=0.0)

    @property
    def overall_level(self) -> str:
        if not self.active_alarms:
            return "normal"
        levels = [a.get("level", "normal") for a in self.active_alarms]
        if "critical" in levels:
            return "critical"
        if "warning" in levels:
            return "warning"
        return "normal"

