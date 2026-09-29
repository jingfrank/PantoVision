#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Monitor (实时性能监控器).

采集每帧端到端耗时、各检测器推理延迟、推流延迟（stream lag）与帧队列深度，
每隔 report_interval_sec 秒通过 logging 输出一次统计汇报。

使用方式:
    pm = PerformanceMonitor(source_fps=25.0, report_interval_sec=10.0)
    # 每帧主循环末尾调用:
    pm.tick(packet, raw_results, wall_ms=38.2, queue_depth=2)
"""

import time
import logging
import threading
from collections import defaultdict
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.data_types import FramePacket, DetectionResult

logger = logging.getLogger(__name__)


class _DetectorStats:
    """单个检测器的统计累积器"""
    __slots__ = ("runs", "total_ms", "max_ms", "_samples")

    def __init__(self):
        self.runs: int = 0
        self.total_ms: float = 0.0
        self.max_ms: float = 0.0
        self._samples: List[float] = []

    def record(self, ms: float) -> None:
        self.runs += 1
        self.total_ms += ms
        if ms > self.max_ms:
            self.max_ms = ms
        self._samples.append(ms)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.runs if self.runs else 0.0

    def reset(self) -> None:
        self.runs = 0
        self.total_ms = 0.0
        self.max_ms = 0.0
        self._samples.clear()


class PerformanceMonitor:
    """
    实时性能监控器（轻量、线程安全）。

    指标说明:
        wall_ms       - 主循环对每帧计时的端到端耗时（scheduler + fusion + HUD）
        algo_fps      - 统计周期内算法实际帧吞吐率
        stream_lag_s  - time.time() - packet.timestamp
                        正值 = 算法处理慢于推流（积压），负值 = 算法追上甚至超前
        queue_depth   - 帧队列瞬时积压深度
        detector_*_ms - 各检测器推理耗时（来自 result.extra["latency_ms"]）

    Args:
        source_fps:          视频源标称帧率，用于计算实时比率
        report_interval_sec: 统计汇报周期（秒），每隔此时间输出一次 logger.info
        enabled:             总开关，False 时 tick() 立即返回，零开销
        logger_name:         使用的 logger 名称（支持多路流独立 logger）
    """

    def __init__(
        self,
        source_fps: float = 25.0,
        report_interval_sec: float = 10.0,
        enabled: bool = True,
        logger_name: str = "perf_monitor",
    ):
        self.source_fps = source_fps
        self.report_interval_sec = report_interval_sec
        self.enabled = enabled
        self._logger = logging.getLogger(logger_name)
        self._lock = threading.Lock()

        # 窗口统计
        self._window_start: float = time.monotonic()
        self._frame_count: int = 0

        # 端到端耗时样本
        self._wall_ms_samples: List[float] = []

        # 流延迟样本
        self._lag_samples: List[float] = []

        # 队列深度样本
        self._queue_depth_samples: List[int] = []

        # 取流与解码耗时样本
        self._ingest_ms_samples: List[float] = []

        # 各检测器统计
        self._detector_stats: Dict[str, _DetectorStats] = defaultdict(_DetectorStats)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def tick(
        self,
        packet: "FramePacket",
        results: Dict[str, "DetectionResult"],
        wall_ms: float,
        queue_depth: int = 0,
        ingest_ms: Optional[float] = None,
    ) -> None:
        """
        每帧处理结束后调用一次。

        Args:
            packet:      当前帧包，用于读取 timestamp 和 frame_idx
            results:     本帧各检测器输出（含 extra["latency_ms"]）
            wall_ms:     本帧端到端耗时（毫秒，由调用方用 perf_counter 计算）
            queue_depth: 当前帧队列积压深度（RTSPStreamReader.frame_queue.qsize()）
            ingest_ms:   当前帧从取流队列读取与解码耗时（毫秒）
        """
        if not self.enabled:
            return

        now_wall = time.time()
        now_mono = time.monotonic()

        with self._lock:
            self._frame_count += 1

            # 端到端耗时
            self._wall_ms_samples.append(wall_ms)

            # 解码与取流耗时
            if ingest_ms is not None:
                self._ingest_ms_samples.append(ingest_ms)

            # 流延迟：wall clock - 帧入队时的系统时间戳
            lag = now_wall - packet.timestamp
            self._lag_samples.append(lag)

            # 队列深度
            self._queue_depth_samples.append(queue_depth)

            # 各检测器推理耗时
            for det_name, res in results.items():
                ms = res.extra.get("latency_ms") if res.extra else None
                if ms is not None:
                    self._detector_stats[det_name].record(float(ms))

            # 判断是否到汇报周期
            elapsed = now_mono - self._window_start
            if elapsed >= self.report_interval_sec:
                self._report(elapsed)
                self._reset_window(now_mono)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _percentile(self, samples: List[float], pct: float) -> float:
        """计算百分位数（简单线性插值）"""
        if not samples:
            return 0.0
        sorted_s = sorted(samples)
        k = (len(sorted_s) - 1) * pct / 100.0
        lo, hi = int(k), min(int(k) + 1, len(sorted_s) - 1)
        return sorted_s[lo] + (sorted_s[hi] - sorted_s[lo]) * (k - lo)

    def _report(self, elapsed_sec: float) -> None:
        """生成并输出统计汇报（已持有锁）"""
        n = self._frame_count
        algo_fps = n / elapsed_sec if elapsed_sec > 0 else 0.0
        fps_ratio = (algo_fps / self.source_fps * 100.0) if self.source_fps > 0 else 0.0

        # 端到端耗时统计
        wall_avg = sum(self._wall_ms_samples) / n if n else 0.0
        wall_max = max(self._wall_ms_samples) if self._wall_ms_samples else 0.0
        wall_p95 = self._percentile(self._wall_ms_samples, 95)

        # 流延迟统计
        lag_avg = sum(self._lag_samples) / n if n else 0.0
        lag_max = max(self._lag_samples) if self._lag_samples else 0.0

        # 队列深度统计
        q_avg = sum(self._queue_depth_samples) / n if n else 0.0
        q_max = max(self._queue_depth_samples) if self._queue_depth_samples else 0

        # 解码取流耗时统计
        ingest_str = ""
        if self._ingest_ms_samples:
            ingest_avg = sum(self._ingest_ms_samples) / len(self._ingest_ms_samples)
            ingest_max = max(self._ingest_ms_samples)
            ingest_str = f"解码取流耗时: avg={ingest_avg:.1f}ms  max={ingest_max:.1f}ms"

        sep = "═" * 54
        lines = [
            sep,
            f"统计周期: {elapsed_sec:.1f}s  处理帧数: {n}帧",
            f"算法 FPS:  {algo_fps:.1f} fps  "
            f"(源标称: {self.source_fps:.1f} fps)  比率: {fps_ratio:.1f}%",
            f"端到端耗时:  avg={wall_avg:.1f}ms  max={wall_max:.1f}ms  p95={wall_p95:.1f}ms",
        ]
        if ingest_str:
            lines.append(ingest_str)
        lines.extend([
            f"流延迟(Lag): avg={lag_avg:+.3f}s  max={lag_max:+.3f}s  "
            f"{'⚠ 积压' if lag_avg > 1.0 else '✓ 正常'}",
            f"队列深度:   avg={q_avg:.1f}  max={q_max}",
        ])

        # 各检测器明细
        if self._detector_stats:
            lines.append("检测器推理耗时 (本周期运行帧):")
            for det_name, stats in sorted(self._detector_stats.items()):
                lines.append(
                    f"  {det_name:<22s}  runs={stats.runs:4d}"
                    f"  avg={stats.avg_ms:6.1f}ms"
                    f"  max={stats.max_ms:6.1f}ms"
                )

        lines.append(sep)

        for line in lines:
            self._logger.info("[PerfMonitor] %s", line)

    def _reset_window(self, now_mono: float) -> None:
        """重置统计窗口（已持有锁）"""
        self._window_start = now_mono
        self._frame_count = 0
        self._wall_ms_samples.clear()
        self._ingest_ms_samples.clear()
        self._lag_samples.clear()
        self._queue_depth_samples.clear()
        for stats in self._detector_stats.values():
            stats.reset()
