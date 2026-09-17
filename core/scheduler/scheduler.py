#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detection Scheduler Layer (检测调度层).
Reads detectors.yaml, manages detector lifecycles, execution intervals, and failure isolation.
"""

import time
import yaml
import logging
from typing import Dict, List, Any, Optional

from core.data_types import FramePacket, DetectionContext, DetectionResult
from core.detectors.registry import DetectorRegistry
from core.detectors.base import BaseDetector

logger = logging.getLogger(__name__)


class DetectionScheduler:
    """
    配置驱动的检测器调度引擎 (Configuration-Driven Scheduler Engine).
    实现多频调度、按需启停与沙箱失败隔离。
    """

    def __init__(self, config_yaml_path: str = "config/detectors.yaml"):
        self.config_path = config_yaml_path
        self.detectors: List[BaseDetector] = []
        self.detector_map: Dict[str, BaseDetector] = {}
        self.last_run_time: Dict[str, float] = {}
        self.last_run_frame: Dict[str, int] = {}
        self.load_configuration(config_yaml_path)

    def load_configuration(self, yaml_path: str) -> None:
        """加载或重载 YAML 配置文件"""
        logger.info(f"[Scheduler] 正在从 {yaml_path} 加载检测器调度配置 ...")
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.global_cfg = cfg.get("global", {})
        detector_cfgs = cfg.get("detectors", [])

        self.detectors.clear()
        self.detector_map.clear()
        self.last_run_time.clear()
        self.last_run_frame.clear()

        for d_cfg in detector_cfgs:
            name = d_cfg.get("name", "")
            plugin_type = d_cfg.get("type", "")
            is_enabled = d_cfg.get("enabled", True)

            try:
                detector = DetectorRegistry.build(d_cfg)
                detector.schedule_cfg = d_cfg.get("schedule", {"mode": "every_frame"})
                self.detectors.append(detector)
                self.detector_map[name] = detector
                self.last_run_time[name] = -999.0
                self.last_run_frame[name] = -999
                logger.info(f"[Scheduler] 成功挂载插件: [{name}] (类型: {plugin_type}, 启用: {is_enabled})")
            except Exception as e:
                logger.error(f"[Scheduler] 挂载检测器插件 [{name}]({plugin_type}) 失败: {e}", exc_info=True)

        self._assign_staggered_phases()
        logger.info(f"[Scheduler] 检测调度层初始化完毕，共挂载 {len(self.detectors)} 个插件")

    def _assign_staggered_phases(self) -> None:
        """
        为同周期 interval 模式检测器自动分配均匀相位偏移。
        按 sample_interval_sec 分组，组内均匀分配触发时间槽。
        若 detector 已经在 schedule 中显式指定了 phase_offset_sec，则予以保留。
        """
        from collections import defaultdict

        interval_groups: Dict[float, List[BaseDetector]] = defaultdict(list)
        for det in self.detectors:
            sched = getattr(det, "schedule_cfg", {})
            if sched.get("mode") == "interval":
                sec = float(sched.get("sample_interval_sec", 1.0))
                interval_groups[sec].append(det)

        for sec_interval, group in interval_groups.items():
            if len(group) <= 1:
                continue

            slot_width = sec_interval / len(group)
            for idx, det in enumerate(group):
                sched = det.schedule_cfg
                if "phase_offset_sec" not in sched:
                    sched["phase_offset_sec"] = round(idx * slot_width, 4)

                offset = sched["phase_offset_sec"]
                logger.info(
                    f"[Scheduler-Stagger] 插件 [{det.name}] "
                    f"错峰相位: +{offset:.3f}s / {sec_interval:.1f}s 周期"
                )

    def set_enabled(self, name: str, enabled: bool) -> bool:
        """现场动态启停控制接口"""
        if name in self.detector_map:
            self.detector_map[name].enabled = enabled
            logger.info(f"[Scheduler] 插件 [{name}] 状态切换为: {'ENABLED' if enabled else 'DISABLED'}")
            return True
        logger.warning(f"[Scheduler] 未找到插件: [{name}]")
        return False

    def _should_run(self, detector: BaseDetector, packet: FramePacket) -> bool:
        """检查分频调度准则 (every_frame / interval)"""
        name = detector.name
        sched = getattr(detector, "schedule_cfg", {"mode": "every_frame"})
        mode = sched.get("mode", "every_frame")

        if mode == "every_frame":
            interval = sched.get("interval_frames", 1)
            last_f = self.last_run_frame.get(name, -999)
            return (packet.frame_idx - last_f) >= interval

        elif mode == "interval":
            sec_interval = float(sched.get("sample_interval_sec", 1.0))
            phase_offset = float(sched.get("phase_offset_sec", 0.0))
            last_t = self.last_run_time.get(name, -999.0)

            if last_t < 0:
                # 首次运行：根据相位偏移决定首次触发时间
                phase_in_cycle = packet.timestamp % sec_interval
                half_frame_tol = sec_interval * 0.02
                return phase_in_cycle >= (phase_offset - half_frame_tol)

            half_frame_tol = sec_interval * 0.02
            return (packet.timestamp - last_t) >= (sec_interval - half_frame_tol)

        return True

    def dispatch(
        self,
        packet: FramePacket,
        context: Optional[DetectionContext] = None,
        force_run_all: bool = False
    ) -> Dict[str, DetectionResult]:
        """
        核心调度分发：执行沙箱保护、黑板上下文共享与失败隔离。
        """
        if context is None:
            context = DetectionContext(packet=packet)
        results: Dict[str, DetectionResult] = {}

        for detector in self.detectors:
            name = detector.name

            # 1. 开关控制检查
            if not detector.enabled:
                continue

            # 2. 分频调度检查 (若 force_run_all=True 如单图测试，则强制运行所有启用插件)
            if not force_run_all and not self._should_run(detector, packet):
                continue

            # 3. 沙箱执行与失败隔离
            try:
                res = detector.run(packet.frame, context=context)

                # interval 模式：将 last_run_time 对齐到理论触发时间，而非实际时间戳。
                # 实际时间戳可能比理论值早最多 half_frame_tol，直接记录会导致
                # 下一周期的 "上次触发时间" 也提前，从而产生累积漂移。
                # 对齐策略：theoretical = last_t + N * interval（N 使理论值 ≤ 实际时间戳）
                sched = getattr(detector, "schedule_cfg", {})
                if sched.get("mode") == "interval":
                    sec_interval = float(sched.get("sample_interval_sec", 1.0))
                    phase_offset = float(sched.get("phase_offset_sec", 0.0))
                    prev_t = self.last_run_time.get(name, -999.0)

                    if prev_t < 0:
                        cycle_base = packet.timestamp - (packet.timestamp % sec_interval)
                        self.last_run_time[name] = cycle_base + phase_offset
                    else:
                        elapsed = packet.timestamp - prev_t
                        n_intervals = max(1, int(elapsed / sec_interval))
                        aligned_t = prev_t + n_intervals * sec_interval
                        self.last_run_time[name] = aligned_t
                else:
                    self.last_run_time[name] = packet.timestamp

                self.last_run_frame[name] = packet.frame_idx
                results[name] = res

            except Exception as e:
                # 即使 detector.run 内部抛出不可预知的异常，在此层完全截断
                logger.error(f"[Scheduler-FailureIsolation] 检测器 [{name}] 发生致命未捕获异常: {e}", exc_info=True)
                results[name] = DetectionResult(
                    fault_type=getattr(detector, "fault_type", name),
                    is_abnormal=False,
                    score=0.0,
                    level="unknown",
                    extra={"error": str(e), "status": "failed_scheduler_intercepted"}
                )

        return results

    def reset_all(self) -> None:
        """重置所有检测器状态"""
        for d in self.detectors:
            try:
                d.reset()
            except Exception:
                pass
        self.last_run_time.clear()
        self.last_run_frame.clear()
