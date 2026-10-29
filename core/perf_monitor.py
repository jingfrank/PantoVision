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
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

try:
    import psutil
    _PSUTIL_PROCESS = psutil.Process()
except Exception:
    _PSUTIL_PROCESS = None

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

if TYPE_CHECKING:
    from core.data_types import FramePacket, DetectionResult

logger = logging.getLogger(__name__)


def _get_gpu_mem_mb() -> Tuple[float, float, float]:
    """返回 GPU 显存占用 (allocated_mb, reserved_mb, peak_allocated_mb)"""
    if _HAS_TORCH and torch.cuda.is_available():
        try:
            alloc = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
            res = torch.cuda.memory_reserved() / (1024.0 * 1024.0)
            peak = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            return round(alloc, 2), round(res, 2), round(peak, 2)
        except Exception:
            pass
    return 0.0, 0.0, 0.0


def _get_ram_rss_mb() -> float:
    """返回当前进程常驻物理内存 RSS (MB)"""
    if _PSUTIL_PROCESS is not None:
        try:
            return round(_PSUTIL_PROCESS.memory_info().rss / (1024.0 * 1024.0), 2)
        except Exception:
            pass
    return 0.0


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
        auto_export_report: bool = True,
        report_dir: Optional[str] = None,
    ):
        self.source_fps = source_fps
        self.report_interval_sec = report_interval_sec
        self.enabled = enabled
        self.logger_name = logger_name
        self._logger = logging.getLogger(logger_name)
        self.auto_export_report = auto_export_report
        self.report_dir = report_dir
        self._lock = threading.Lock()

        # 窗口统计
        self._window_start: float = time.monotonic()
        self._frame_count: int = 0

        # 端到端耗时样本 (主循环调度+融合+渲染+存帧)
        self._wall_ms_samples: List[float] = []

        # 全链路单帧处理总耗时样本 (取流解码 + 主循环处理)
        self._e2e_ms_samples: List[float] = []

        # 流延迟样本
        self._lag_samples: List[float] = []

        # 队列深度样本
        self._queue_depth_samples: List[int] = []

        # 取流与解码耗时样本
        self._ingest_ms_samples: List[float] = []

        # 各检测器窗口统计
        self._detector_stats: Dict[str, _DetectorStats] = defaultdict(_DetectorStats)

        # ── 全生命周期累积统计 (用于评测报告导出) ────────────────────────
        self._lifetime_start_mono: float = self._window_start
        self._lifetime_frames: int = 0
        self._lifetime_wall_ms: List[float] = []
        self._lifetime_e2e_ms: List[float] = []
        self._lifetime_ingest_ms: List[float] = []
        self._lifetime_lag_s: List[float] = []
        self._lifetime_peak_vram_mb: float = 0.0
        self._lifetime_peak_ram_mb: float = 0.0
        self._detector_lifetime_stats: Dict[str, _DetectorStats] = defaultdict(_DetectorStats)

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

        # 计算全链路单帧处理总耗时 (取流解码 + 主循环处理)
        e2e_ms = (ingest_ms or 0.0) + wall_ms

        with self._lock:
            self._frame_count += 1
            self._lifetime_frames += 1

            # 窗口样本
            self._wall_ms_samples.append(wall_ms)
            self._e2e_ms_samples.append(e2e_ms)

            # 解码与取流耗时
            if ingest_ms is not None:
                self._ingest_ms_samples.append(ingest_ms)

            # 流延迟：wall clock - 帧入队时的系统时间戳
            lag = now_wall - packet.timestamp
            self._lag_samples.append(lag)

            # 队列深度
            self._queue_depth_samples.append(queue_depth)

            # 全生命周期累积样本 (最多保留 20000 帧样本防止长期运行内存膨胀)
            if len(self._lifetime_wall_ms) < 20000:
                self._lifetime_wall_ms.append(wall_ms)
                self._lifetime_e2e_ms.append(e2e_ms)
                if ingest_ms is not None:
                    self._lifetime_ingest_ms.append(ingest_ms)
                self._lifetime_lag_s.append(lag)

            # 硬件资源监控 (每 5 帧刷新一次以兼顾性能与时效)
            if self._frame_count % 5 == 1:
                alloc_mb, _, peak_mb = _get_gpu_mem_mb()
                ram_mb = _get_ram_rss_mb()
                cur_vram_max = max(alloc_mb, peak_mb)
                if cur_vram_max > self._lifetime_peak_vram_mb:
                    self._lifetime_peak_vram_mb = cur_vram_max
                if ram_mb > self._lifetime_peak_ram_mb:
                    self._lifetime_peak_ram_mb = ram_mb

            # 各检测器推理耗时 (窗口 + 生命周期)
            for det_name, res in results.items():
                ms = res.extra.get("latency_ms") if res.extra else None
                if ms is not None:
                    ms_val = float(ms)
                    self._detector_stats[det_name].record(ms_val)
                    self._detector_lifetime_stats[det_name].record(ms_val)

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

        # 全链路总时延统计 (Ingest + Wall)
        e2e_avg = sum(self._e2e_ms_samples) / n if n else 0.0
        e2e_max = max(self._e2e_ms_samples) if self._e2e_ms_samples else 0.0
        e2e_p95 = self._percentile(self._e2e_ms_samples, 95)

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
            ingest_str = f"取流解码耗时: avg={ingest_avg:.1f}ms  max={ingest_max:.1f}ms"

        # 硬件资源占用统计
        vram_alloc, vram_res, vram_peak = _get_gpu_mem_mb()
        ram_rss = _get_ram_rss_mb()
        if max(vram_alloc, vram_peak) > self._lifetime_peak_vram_mb:
            self._lifetime_peak_vram_mb = max(vram_alloc, vram_peak)
        if ram_rss > self._lifetime_peak_ram_mb:
            self._lifetime_peak_ram_mb = ram_rss

        sep = "═" * 58
        lines = [
            sep,
            f"统计周期: {elapsed_sec:.1f}s  处理帧数: {n}帧",
            f"系统吞吐 FPS: {algo_fps:.1f} fps  (源标称: {self.source_fps:.1f} fps)  实时比率: {fps_ratio:.1f}%",
            f"全链路单帧耗时: avg={e2e_avg:.1f}ms  p95={e2e_p95:.1f}ms  max={e2e_max:.1f}ms",
            f"主循环处理耗时: avg={wall_avg:.1f}ms  p95={wall_p95:.1f}ms  max={wall_max:.1f}ms",
        ]
        if ingest_str:
            lines.append(ingest_str)
        lines.extend([
            f"硬件资源占用: GPU显存已分配={vram_alloc:.1f}MB (峰值={self._lifetime_peak_vram_mb:.1f}MB) | 内存RSS={ram_rss:.1f}MB (峰值={self._lifetime_peak_ram_mb:.1f}MB)",
            f"流延迟(Lag):   avg={lag_avg:+.3f}s  max={lag_max:+.3f}s  {'⚠ 积压' if lag_avg > 1.0 else '✓ 正常'}",
            f"队列深度:      avg={q_avg:.1f}  max={q_max}",
        ])

        # 各检测器明细
        if self._detector_stats:
            lines.append("各检测器单体推理耗时 (本周期运行帧):")
            for det_name, stats in sorted(self._detector_stats.items()):
                lines.append(
                    f"  {det_name:<22s}  runs={stats.runs:4d}"
                    f"  avg={stats.avg_ms:6.1f}ms"
                    f"  max={stats.max_ms:6.1f}ms"
                )

        lines.append(sep)

        for line in lines:
            self._logger.info("[PerfMonitor] %s", line)

        # 实时自动持久化当前阶段的评测大盘
        if self.auto_export_report and self._lifetime_frames > 0:
            try:
                target_dir = self.report_dir or os.path.join(os.getcwd(), "local_cache", "reports")
                safe_name = self.logger_name.replace(".", "_")
                live_file = os.path.join(target_dir, f"live_perf_report_{safe_name}.md")
                self.export_report(live_file, title=f"动车组受电弓网监测系统 - 实时推流性能大盘 ({safe_name})")
            except Exception as e:
                self._logger.debug("自动导出实时性能大盘异常: %s", e)

    def _reset_window(self, now_mono: float) -> None:
        """重置统计窗口（已持有锁）"""
        self._window_start = now_mono
        self._frame_count = 0
        self._wall_ms_samples.clear()
        self._e2e_ms_samples.clear()
        self._ingest_ms_samples.clear()
        self._lag_samples.clear()
        self._queue_depth_samples.clear()
        for stats in self._detector_stats.values():
            stats.reset()

    # ------------------------------------------------------------------
    # 评测报告与全生命周期汇总导出接口
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """
        获取系统自启动或测试以来的全生命周期综合评测指标字典。
        包含单帧延迟分布 (Mean/P50/P90/P95/P99/Max)、吞吐能力 FPS、
        GPU 显存占用 (MB)、Host 内存占用 (MB) 及各检测器开销。
        """
        with self._lock:
            total_frames = self._lifetime_frames
            total_elapsed = max(time.monotonic() - self._lifetime_start_mono, 1e-4)
            avg_fps = total_frames / total_elapsed if total_elapsed > 0 else 0.0

            def _summarize(samples: List[float]) -> Dict[str, float]:
                if not samples:
                    return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "min": 0.0}
                return {
                    "mean": round(sum(samples) / len(samples), 2),
                    "p50": round(self._percentile(samples, 50), 2),
                    "p90": round(self._percentile(samples, 90), 2),
                    "p95": round(self._percentile(samples, 95), 2),
                    "p99": round(self._percentile(samples, 99), 2),
                    "max": round(max(samples), 2),
                    "min": round(min(samples), 2),
                }

            e2e_stats = _summarize(self._lifetime_e2e_ms)
            wall_stats = _summarize(self._lifetime_wall_ms)
            ingest_stats = _summarize(self._lifetime_ingest_ms)

            vram_alloc, vram_res, vram_peak = _get_gpu_mem_mb()
            ram_rss = _get_ram_rss_mb()
            peak_vram = max(self._lifetime_peak_vram_mb, vram_alloc, vram_peak)
            peak_ram = max(self._lifetime_peak_ram_mb, ram_rss)

            detectors_summary = {}
            for det_name, stats in sorted(self._detector_lifetime_stats.items()):
                detectors_summary[det_name] = {
                    "runs": stats.runs,
                    "avg_ms": round(stats.avg_ms, 2),
                    "max_ms": round(stats.max_ms, 2),
                }

            return {
                "total_frames": total_frames,
                "total_elapsed_sec": round(total_elapsed, 2),
                "throughput_fps": round(avg_fps, 2),
                "source_nominal_fps": round(self.source_fps, 2),
                "realtime_ratio_pct": round((avg_fps / self.source_fps * 100.0) if self.source_fps > 0 else 0.0, 1),
                "latency_e2e_ms": e2e_stats,
                "latency_main_wall_ms": wall_stats,
                "latency_ingest_ms": ingest_stats,
                "gpu_vram_allocated_mb": vram_alloc,
                "gpu_vram_reserved_mb": vram_res,
                "gpu_vram_peak_mb": round(peak_vram, 2),
                "host_ram_rss_mb": ram_rss,
                "host_ram_peak_mb": round(peak_ram, 2),
                "detectors": detectors_summary,
            }

    def get_report_markdown(self, title: str = "受电弓网视频故障监测系统 - 全链路端到端性能评测报告") -> str:
        """生成符合工业规范与评测报告标准的 Markdown 格式量化表"""
        summary = self.get_summary()
        e2e = summary["latency_e2e_ms"]
        wall = summary["latency_main_wall_ms"]
        ingest = summary["latency_ingest_ms"]

        gpu_name = torch.cuda.get_device_name(0) if (_HAS_TORCH and torch.cuda.is_available()) else "CPU / 无CUDA"

        lines = [
            f"# {title}",
            "",
            f"- **测试设备**: `{gpu_name}`",
            f"- **测试总帧数**: `{summary['total_frames']} 帧` | **测试总时长**: `{summary['total_elapsed_sec']} 秒`",
            f"- **视频源标称帧率**: `{summary['source_nominal_fps']} FPS`",
            "",
            "---",
            "",
            "## 一、 系统全链路单帧处理延迟分布 (Latency)",
            "",
            "| 处理阶段 / 链路层级 | 平均耗时 (Mean) | P50 中位数 | P90 耗时 | P95 耗时 | P99 尾部耗时 | 最大峰值 (Max) | 33.3ms预算占用率 |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            f"| **全链路端到端总延迟 (E2E Total)** | **{e2e['mean']:.2f} ms** | **{e2e['p50']:.2f} ms** | {e2e['p90']:.2f} ms | **{e2e['p95']:.2f} ms** | {e2e['p99']:.2f} ms | {e2e['max']:.2f} ms | **{(e2e['mean']/33.33*100.0):.1f}%** |",
            f"| └─ 取流与视频解码 (Ingest/Decode) | {ingest['mean']:.2f} ms | {ingest['p50']:.2f} ms | {ingest['p90']:.2f} ms | {ingest['p95']:.2f} ms | {ingest['p99']:.2f} ms | {ingest['max']:.2f} ms | {(ingest['mean']/33.33*100.0):.1f}% |",
            f"| └─ 主循环处理 (调度+融合+渲染+存帧) | {wall['mean']:.2f} ms | {wall['p50']:.2f} ms | {wall['p90']:.2f} ms | {wall['p95']:.2f} ms | {wall['p99']:.2f} ms | {wall['max']:.2f} ms | {(wall['mean']/33.33*100.0):.1f}% |",
            "",
            "---",
            "",
            "## 二、 吞吐能力与实时性指标 (Throughput)",
            "",
            "| 性能指标项 | 实测数值 | 达标判定标准 | 判定结论 |",
            "| :--- | :--- | :--- | :--- |",
            f"| **系统吞吐能力 (FPS / QPS)** | **{summary['throughput_fps']:.2f} FPS** | ≥ 25.0 FPS (满足车载实时流处理) | {'✅ 达标' if summary['throughput_fps'] >= 25.0 else '⚠️ 不足'} |",
            f"| **实时吞吐比率 (Real-Time Ratio)** | **{summary['realtime_ratio_pct']:.1f}%** | ≥ 95.0% (无视频流积压/发散) | {'✅ 极佳' if summary['realtime_ratio_pct'] >= 95.0 else '⚠️ 积压'} |",
            "",
            "---",
            "",
            "## 三、 硬件资源开销 (GPU 显存 / 系统内存 / CPU)",
            "",
            "| 资源类别 | 稳态运行值 | 峰值开销 (Peak) | 说明 |",
            "| :--- | :--- | :--- | :--- |",
            f"| **GPU 显存 (Allocated / Peak)** | **{summary['gpu_vram_allocated_mb']:.1f} MB** | **{summary['gpu_vram_peak_mb']:.1f} MB** | 包含模型权重与 TensorRT Execution Context |",
            f"| **GPU 显存预留 (Reserved)** | {summary['gpu_vram_reserved_mb']:.1f} MB | - | PyTorch CUDA 预分配缓存池 |",
            f"| **系统物理内存 (Host RAM RSS)** | **{summary['host_ram_rss_mb']:.1f} MB** | **{summary['host_ram_peak_mb']:.1f} MB** | 包含 60s 环形视频缓冲队列与解耦线程 |",
            "",
            "---",
            "",
            "## 四、 各检测器单体推理延迟拆解",
            "",
            "| 检测器名称 | 运行次数 (Runs) | 平均耗时 (Mean) | 最大耗时 (Max) | 调度策略 |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]

        for det_name, d_info in summary["detectors"].items():
            lines.append(
                f"| `{det_name}` | {d_info['runs']} 次 | **{d_info['avg_ms']:.2f} ms** | {d_info['max_ms']:.2f} ms | 分频/错峰调度 |"
            )

        lines.extend(["", "---", ""])
        return "\n".join(lines)

    def export_report(self, filepath: str, title: str = "受电弓网视频故障监测系统 - 全链路端到端性能评测报告") -> str:
        """导出 Markdown 报告及配套的 JSON 指标归档文件"""
        abs_path = os.path.abspath(filepath)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        md_content = self.get_report_markdown(title=title)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        json_path = os.path.splitext(abs_path)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.get_summary(), f, indent=2, ensure_ascii=False)

        self._logger.info("[PerfMonitor] 评测报告已成功导出: %s 与 %s", abs_path, json_path)
        return abs_path

    def log_summary(self) -> None:
        """在日志中打印全生命周期的完整评测汇总"""
        summary = self.get_summary()
        e2e = summary["latency_e2e_ms"]
        sep = "★" * 58
        self._logger.info(sep)
        self._logger.info("受电弓网视频故障监测系统 - 全生命周期综合性能汇报:")
        self._logger.info("  总处理帧数: %d 帧 | 平均吞吐: %.2f FPS (实时比率: %.1f%%)",
                          summary["total_frames"], summary["throughput_fps"], summary["realtime_ratio_pct"])
        self._logger.info("  全链路单帧耗时: avg=%.2fms | p50=%.2fms | p95=%.2fms | max=%.2fms",
                          e2e["mean"], e2e["p50"], e2e["p95"], e2e["max"])
        self._logger.info("  GPU 显存占用: 稳态=%.1fMB | 峰值=%.1fMB",
                          summary["gpu_vram_allocated_mb"], summary["gpu_vram_peak_mb"])
        self._logger.info("  Host 内存占用: 稳态=%.1fMB | 峰值=%.1fMB",
                          summary["host_ram_rss_mb"], summary["host_ram_peak_mb"])
        self._logger.info(sep)

        # 自动导出最终全生命周期评测报告
        if self.auto_export_report and summary["total_frames"] > 0:
            try:
                target_dir = self.report_dir or os.path.join(os.getcwd(), "local_cache", "reports")
                safe_name = self.logger_name.replace(".", "_")
                final_file = os.path.join(target_dir, f"final_perf_report_{safe_name}.md")
                self.export_report(final_file, title=f"动车组受电弓网视频监测系统 - 全生命周期综合性能评测报告 ({safe_name})")
            except Exception as e:
                self._logger.debug("自动导出全生命周期性能报告异常: %s", e)
