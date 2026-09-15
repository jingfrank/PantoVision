#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusion & Decision Layer (结果融合与决策层).
Cross-module correlation, mutual exclusion, false-positive suppression, and priority arbitration.
"""

from typing import Dict, List, Any, Optional
import logging
from core.data_types import DetectionResult, FusionResult, FramePacket

logger = logging.getLogger(__name__)

FAULT_NAME_MAP = {
    "big_spark": "大火花",
    "structure_abnormal": "受电弓结构异常",
    "dirty": "画面脏污",
    "tilt": "受电弓倾斜",
    "foreign_object": "悬挂异物",
    "abnormal_float": "异常飘弓",
}

# 故障优先级仲裁顺序 (从高到低)
PRIORITY_ORDER = [
    "big_spark",
    "structure_abnormal",
    "tilt",
    "foreign_object",
    "abnormal_float",
    "dirty",
]


class FusionEngine:
    """
    智能融合决策引擎.
    执行跨模块抑制规则与全局状态裁决。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def decide(self, results: Dict[str, DetectionResult], packet: Optional[FramePacket] = None) -> FusionResult:
        """
        跨模块互斥、抑制与综合裁决
        """
        timestamp = packet.timestamp if packet else 0.0

        # 提取各检测器状态
        spark_res = results.get("spark") or results.get("big_spark")
        dirt_res = results.get("dirt") or results.get("dirty")
        tilt_res = results.get("tilt")
        float_res = results.get("float") or results.get("abnormal_float")
        struct_res = results.get("structural") or results.get("structure_abnormal")
        foreign_res = results.get("foreign_object")

        spark_active = bool(spark_res and (spark_res.is_abnormal or spark_res.extra.get("spark_count", 0) > 0))
        dirt_active = bool(dirt_res and dirt_res.is_abnormal)

        # -------------------------------------------------------------
        # 规则 1：大火花强电弧抑制 (Spark Dominance)
        # 火花发生瞬间白炽漫反射强烈，强制抑制结构异常、悬挂异物与画面脏污误报
        # -------------------------------------------------------------
        suppress_struct = False
        suppress_foreign = False
        suppress_dirty = False

        if spark_active:
            suppress_struct = True
            suppress_foreign = True
            suppress_dirty = True

        # -------------------------------------------------------------
        # 规则 2：画面脏污穿透抑制 (Dirt Isolation)
        # 处于镜头脏污或脏污观察态时，抑制悬挂异物与结构异常向下穿透误报
        # -------------------------------------------------------------
        if dirt_active or (dirt_res and dirt_res.extra.get("is_pending", False)):
            suppress_foreign = True
            suppress_struct = True

        # 收集有效告警清单
        active_alarms: List[Dict[str, Any]] = []

        for det_name, res in results.items():
            f_type = res.fault_type

            # 抑制检查
            if f_type == "structure_abnormal" and suppress_struct:
                continue
            if f_type == "foreign_object" and suppress_foreign:
                continue
            if f_type == "dirty" and suppress_dirty:
                continue
            if res.level == "unknown":
                # 失败隔离状态：不触发误报警，仅记录
                continue

            if res.is_abnormal:
                chinese_name = FAULT_NAME_MAP.get(f_type, f_type)
                active_alarms.append({
                    "detector_name": det_name,
                    "fault_type": f_type,
                    "chinese_name": chinese_name,
                    "level": res.level,
                    "score": res.score,
                    "bbox": res.bbox,
                    "keypoints": res.keypoints,
                    "extra": res.extra,
                })

        # -------------------------------------------------------------
        # 规则 3：主告警优先级仲裁 (Priority Arbitration)
        # -------------------------------------------------------------
        has_fault = len(active_alarms) > 0
        primary_fault = "正常"

        if has_fault:
            # 按照 PRIORITY_ORDER 选择最高优先级的故障作为 primary_fault
            chosen = None
            for p_type in PRIORITY_ORDER:
                for alarm in active_alarms:
                    if alarm["fault_type"] == p_type:
                        chosen = alarm
                        break
                if chosen:
                    break

            if chosen is None:
                chosen = active_alarms[0]

            primary_fault = f"异常-{chosen['chinese_name']}"

        # 按 PRIORITY_ORDER 对告警列表排序，确保 active_alarms[0] 始终是最高优先级故障
        # 供 render_hud 分发器通过 active_alarms[0] 可靠获取主告警类型
        def _alarm_priority(alarm):
            ft = alarm.get("fault_type", "")
            return PRIORITY_ORDER.index(ft) if ft in PRIORITY_ORDER else len(PRIORITY_ORDER)

        active_alarms.sort(key=_alarm_priority)

        return FusionResult(
            has_fault=has_fault,
            primary_fault=primary_fault,
            active_alarms=active_alarms,
            detector_results=results,
            annotated_frame=None,
            timestamp=timestamp,
            metadata={
                "suppress_struct": suppress_struct,
                "suppress_foreign": suppress_foreign,
                "suppress_dirty": suppress_dirty,
                "alarm_count": len(active_alarms),
            }
        )
