#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusion & Decision Layer (结果融合与决策层).
Cross-module correlation, mutual exclusion, false-positive suppression, and priority arbitration.
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from core.data_types import DetectionResult, FramePacket, FusionResult

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
    执行跨模块抑制规则与全局状态裁决，内置跨帧 Sample-and-Hold 机制。

    Sample-and-Hold 机制
    --------------------
    分频检测器（如 FastAD dirty/structural/foreign_object、HRNet tilt）不在每帧执行，
    调度器仅在检测器触发时将其结果注入 results dict。在空窗帧（检测器未运行），
    FusionEngine 从 cached_results 中取出上一次判决，补入本帧的合并结果集，
    再统一经过抑制规则裁决，从而消除 OSD 与告警的周期性闪烁。

    Hold 结果的特殊处理
    ------------------
    - bbox / keypoints 字段置 None：位置信息在高速场景下迅速失效，不叠加到 OSD。
    - extra["held"] = True：便于 OSD 层区分标注样式（如虚线框 vs 实线框）。
    - extra["hold_since"] / extra["hold_age_sec"]：可观测性字段，供日志与调试使用。

    超出 hold_timeout_sec 的缓存不会补入，等效于检测器静默返回正常态。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # Sample-and-Hold 保持超时（秒），略大于分频检测器的采样周期以留容差
        self.hold_timeout_sec: float = float(self.config.get("hold_timeout_sec", 1.5))
        # 跨帧状态缓存: detector_name -> (DetectionResult, timestamp_when_cached)
        self.cached_results: Dict[str, Tuple[DetectionResult, float]] = {}

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        重置所有跨帧状态缓存.
        在切换测试视频或离线批处理开始时调用，确保上一段视频的 Hold 状态不污染新段。
        注意：上层流水线应同时调用 DetectionScheduler.reset_all()，两者互不依赖。
        """
        self.cached_results.clear()
        logger.info("[FusionEngine] 跨帧状态缓存已重置")

    def decide(
        self,
        results: Dict[str, DetectionResult],
        packet: Optional[FramePacket] = None,
    ) -> FusionResult:
        """
        跨模块互斥、抑制与综合裁决.

        Parameters
        ----------
        results:
            DetectionScheduler 本帧实际运行并返回的检测结果字典，
            key 为 detector_name（与 detectors.yaml 中 name 字段对应）。
            未在本帧执行的分频检测器不会出现在此 dict 中。
        packet:
            当前帧数据包，用于读取时间戳。
        """
        timestamp = packet.timestamp if packet else 0.0

        # ------------------------------------------------------------------
        # Phase 0: Sample-and-Hold — 补入分频检测器在空窗帧的持续结果
        # ------------------------------------------------------------------
        merged, held_detectors = self._apply_sample_and_hold(results, timestamp)

        # ------------------------------------------------------------------
        # 提取各检测器状态（从合并后的结果集中读取）
        # ------------------------------------------------------------------
        spark_res = merged.get("spark") or merged.get("big_spark")
        dirt_res = merged.get("dirt") or merged.get("dirty")
        tilt_res = merged.get("tilt")
        float_res = merged.get("float") or merged.get("abnormal_float")
        struct_res = merged.get("structural") or merged.get("structure_abnormal")
        foreign_res = merged.get("foreign_object")

        spark_active = bool(
            spark_res
            and (spark_res.is_abnormal or spark_res.extra.get("spark_count", 0) > 0)
        )
        dirt_active = bool(dirt_res and dirt_res.is_abnormal)

        # ------------------------------------------------------------------
        # 规则 1：大火花强电弧抑制 (Spark Dominance)
        # 火花发生瞬间白炽漫反射强烈，强制抑制结构异常、悬挂异物与画面脏污误报。
        # Hold 结果同样参与此规则，行为与新鲜结果一致。
        # ------------------------------------------------------------------
        suppress_struct = False
        suppress_foreign = False
        suppress_dirty = False

        if spark_active:
            suppress_struct = True
            suppress_foreign = True
            suppress_dirty = True

        # ------------------------------------------------------------------
        # 规则 2：画面脏污穿透抑制 (Dirt Isolation)
        # 处于镜头脏污或脏污观察态时，抑制悬挂异物与结构异常向下穿透误报。
        # ------------------------------------------------------------------
        if dirt_active or (dirt_res and dirt_res.extra.get("is_pending", False)):
            suppress_foreign = True
            suppress_struct = True

        # 收集有效告警清单
        active_alarms: List[Dict[str, Any]] = []

        for det_name, res in merged.items():
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
                active_alarms.append(
                    {
                        "detector_name": det_name,
                        "fault_type": f_type,
                        "chinese_name": chinese_name,
                        "level": res.level,
                        "score": res.score,
                        "bbox": res.bbox,          # Hold 结果此字段已置 None
                        "keypoints": res.keypoints, # Hold 结果此字段已置 None
                        "extra": res.extra,         # 含 held/hold_since/hold_age_sec
                    }
                )

        # ------------------------------------------------------------------
        # 规则 3：主告警优先级仲裁 (Priority Arbitration)
        # ------------------------------------------------------------------
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

        # 确保 active_alarms 列表始终按 PRIORITY_ORDER 降序排列 (最高优先级在前)
        def _alarm_priority(alarm):
            ft = alarm.get("fault_type", "")
            return PRIORITY_ORDER.index(ft) if ft in PRIORITY_ORDER else len(PRIORITY_ORDER)

        active_alarms.sort(key=_alarm_priority)

        return FusionResult(
            has_fault=has_fault,
            primary_fault=primary_fault,
            active_alarms=active_alarms,
            detector_results=merged,
            annotated_frame=None,
            timestamp=timestamp,
            metadata={
                "suppress_struct": suppress_struct,
                "suppress_foreign": suppress_foreign,
                "suppress_dirty": suppress_dirty,
                "alarm_count": len(active_alarms),
                # 可观测性：本帧哪些检测器使用了 Hold 结果
                "held_detectors": sorted(held_detectors),
                "fresh_detectors": sorted(set(results.keys())),
            },
        )

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _apply_sample_and_hold(
        self,
        fresh_results: Dict[str, DetectionResult],
        timestamp: float,
    ) -> Tuple[Dict[str, DetectionResult], Set[str]]:
        """
        执行 Sample-and-Hold 逻辑，返回合并后的结果集与本帧被 Hold 的检测器名称集合。

        步骤：
        1. 刷新缓存：将本帧新鲜结果写入 cached_results（覆盖旧值）。
        2. 补入 Hold：对缓存中有效期内、但本帧未运行的检测器，创建 Hold 副本补入合并集。
           - bbox / keypoints 置 None（位置信息失效）。
           - extra 注入 held=True、hold_since、hold_age_sec 三个可观测字段。
        3. 超时缓存不补入（等效于该检测器本轮无输出，裁决层视其为"无信息"）。
        """
        # Step 1: 以 detector_name 为 key，刷新本帧新鲜结果到缓存
        for name, res in fresh_results.items():
            self.cached_results[name] = (res, timestamp)

        # Step 2: 构建合并结果集（从新鲜结果出发，补入 Hold）
        merged: Dict[str, DetectionResult] = dict(fresh_results)
        held_detectors: Set[str] = set()

        for name, (cached_res, cached_ts) in self.cached_results.items():
            if name in merged:
                # 本帧已有新鲜结果，跳过
                continue

            age = timestamp - cached_ts
            if age > self.hold_timeout_sec:
                # 缓存已超时，静默丢弃，不补入
                logger.debug(
                    "[FusionEngine] Hold 超时，检测器 [%s] 缓存丢弃 (age=%.2fs > timeout=%.1fs)",
                    name,
                    age,
                    self.hold_timeout_sec,
                )
                continue

            # 创建 Hold 副本，避免修改缓存中的原始对象
            held = copy.copy(cached_res)
            held.extra = dict(cached_res.extra)   # shallow copy extra dict
            held.extra["held"] = True
            held.extra["hold_since"] = cached_ts
            held.extra["hold_age_sec"] = round(age, 3)
            held.bbox = None       # 高速场景下位置已失效，不叠加到 OSD
            held.keypoints = None  # 同上

            merged[name] = held
            held_detectors.add(name)

            logger.debug(
                "[FusionEngine] Hold 生效: 检测器 [%s] (is_abnormal=%s, age=%.3fs)",
                name,
                cached_res.is_abnormal,
                age,
            )

        return merged, held_detectors
