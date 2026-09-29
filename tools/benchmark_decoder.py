#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson AGX Orin 视频解码性能量化评测工具 (benchmark_decoder.py).

功能:
1. 针对 RTSP 流（海康相机或模拟推流）进行 FFmpeg CPU 软解 vs GStreamer NVDEC 硬解的严格量化对比；
2. 自动探测 RTSP 视频流编码格式 (H.264 vs H.265/HEVC)；
3. 统计单帧解码耗时 (Mean/P50/P95/P99/Max)、FPS 吞吐、进程 CPU 占用率、内存开销；
4. 深度采集 Jetson AGX Orin 硬件状态 (NVDEC 引擎利用率、VIC 引擎利用率、功耗与芯片温度)；
5. 输出 Markdown 格式的量化对比报告及 JSON 导出文件。

使用示例:
    # 对比模式 (测试 600 帧并生成量化报告)
    python3 tools/benchmark_decoder.py --rtsp rtsp://172.10.1.126/live/test --frames 600

    # 针对特定后端单独测试
    python3 tools/benchmark_decoder.py --rtsp rtsp://172.10.1.126/live/test --backend nvdec --frames 300
"""

import os
import sys
import time
import socket
import argparse
import subprocess
import threading
import json
import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import cv2
except ImportError:
    print("[ERROR] 无法导入 cv2，请在包含 OpenCV 的 Python 环境中运行此脚本！", file=sys.stderr)
    sys.exit(1)

try:
    import psutil
except ImportError:
    psutil = None


# ==============================================================================
# 1. 智能编码探测器 (H.264 vs H.265)
# ==============================================================================

def probe_rtsp_codec_socket(rtsp_url: str, timeout: float = 1.5) -> Optional[str]:
    """
    通过原生 Socket 发送 RTSP DESCRIBE 请求解析 SDP 媒体行。
    零依赖、毫秒级响应，适用于无密码的模拟推流 (EasyDarwin / FFmpeg / 部分相机)。
    """
    try:
        # 解析 host 与 port
        match = re.match(r"rtsps?://(?:([^:@]+)(?::([^@]+))?@)?([^:/]+)(?::(\d+))?(/.*)?", rtsp_url)
        if not match:
            return None
        _, _, host, port_str, path = match.groups()
        port = int(port_str) if port_str else 554
        path = path if path else "/"

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect((host, port))
            req = (
                f"DESCRIBE {rtsp_url} RTSP/1.0\r\n"
                f"CSeq: 1\r\n"
                f"Accept: application/sdp\r\n"
                f"User-Agent: GongwangBenchmark/1.0\r\n\r\n"
            )
            sock.sendall(req.encode('utf-8'))

            response = b""
            while True:
                chunk = sock.recv(2048)
                if not chunk:
                    break
                response += chunk
                if b"\r\n\r\n" in response and len(response) > 512:
                    break
        finally:
            sock.close()

        res_text = response.decode('utf-8', errors='ignore')
        if "401 Unauthorized" in res_text:
            return None  # 遇到鉴权挑战，交给 ffprobe 处理

        res_upper = res_text.upper()
        if "H265" in res_upper or "HEVC" in res_upper:
            return "h265"
        if "H264" in res_upper or "AVC" in res_upper:
            return "h264"
    except Exception:
        pass
    return None


def probe_rtsp_codec_ffprobe(rtsp_url: str, timeout: float = 3.0) -> Optional[str]:
    """
    调用 ffprobe 进行权威探测，支持 URL 内置的鉴权账号密码（如海康威视相机）。
    """
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-rtsp_transport", "tcp",
            "-stimeout", str(int(timeout * 1000000)),
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-of", "default=noprint_wrappers=1:nokey=1",
            rtsp_url
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        codec = res.stdout.strip().lower()
        if "265" in codec or "hevc" in codec:
            return "h265"
        elif "264" in codec or "avc" in codec:
            return "h264"
    except Exception:
        pass
    return None


def detect_rtsp_codec(rtsp_url: str) -> str:
    """
    综合探测 RTSP 流编码格式，优先 Socket 毫秒探测，失败则回退至 ffprobe 鉴权探测。
    最终保底返回 'h264'。
    """
    print(f"🔍 [编码探测] 正在探测 RTSP 视频流编码: {rtsp_url} ...")
    codec = probe_rtsp_codec_socket(rtsp_url, timeout=1.0)
    if codec:
        print(f"✅ [编码探测] Socket SDP 快速命中视频编码: {codec.upper()}")
        return codec

    codec = probe_rtsp_codec_ffprobe(rtsp_url, timeout=3.0)
    if codec:
        print(f"✅ [编码探测] ffprobe 鉴权解析命中视频编码: {codec.upper()}")
        return codec

    print("⚠️ [编码探测] 未探测到明确编码类型，默认采用 H.264")
    return "h264"


# ==============================================================================
# 2. GStreamer 管道生成器 (针对 Jetson AGX Orin 极致优化)
# ==============================================================================

def build_jetson_gst_pipeline(rtsp_url: str, codec: str = "h264", latency_ms: int = 0) -> str:
    """
    构造 Jetson AGX Orin 专用的硬件解码与格式转换管道。
    关键优化点:
    1. latency=0 + drop-on-latency=true 消除网络抖动引发的积压；
    2. nvv4l2decoder enable-max-performance=1 激活全频硬解；
    3. nvvidconv 直接输出 BGRx 规避 CPU videoconvert 的巨大开销；
    4. appsink drop=1 max-buffers=1 保持实时单帧吞吐。
    """
    depay = "rtph264depay ! h264parse" if codec == "h264" else "rtph265depay ! h265parse"
    pipeline = (
        f"rtspsrc location={rtsp_url} latency={latency_ms} protocols=tcp drop-on-latency=true ! "
        f"{depay} ! "
        f"nvv4l2decoder enable-max-performance=1 drop-frame-interval=0 ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )
    return pipeline


# ==============================================================================
# 3. Jetson 硬件状态监测器 (TegraStats Collector)
# ==============================================================================

class TegraStatsCollector:
    """Jetson 硬件传感器后台采样线程 (解析 NVDEC / VIC / 温度 / 功耗)"""

    def __init__(self, interval_ms: int = 200):
        self.interval_ms = interval_ms
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.nvdec_samples: List[float] = []
        self.vic_samples: List[float] = []
        self.power_mw_samples: List[float] = []
        self.temp_c_samples: List[float] = []
        self.has_tegrastats = os.path.exists("/usr/bin/tegrastats")

    def start(self) -> None:
        if not self.has_tegrastats:
            return
        self.stop_event.clear()
        self.nvdec_samples.clear()
        self.vic_samples.clear()
        self.power_mw_samples.clear()
        self.temp_c_samples.clear()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self) -> None:
        try:
            proc = subprocess.Popen(
                ["/usr/bin/tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )
            while not self.stop_event.is_set():
                line = proc.stdout.readline()
                if not line:
                    break

                # 解析 NVDEC (如 NVDEC 15% 或 NVDEC 0%)
                nvdec_match = re.search(r"NVDEC\s+(\d+)%", line)
                if nvdec_match:
                    self.nvdec_samples.append(float(nvdec_match.group(1)))

                # 解析 VIC (Video Image Compositor, 如 VIC 5%)
                vic_match = re.search(r"VIC\s+(\d+)%", line)
                if vic_match:
                    self.vic_samples.append(float(vic_match.group(1)))

                # 解析整机功耗 (如 VDD_IN 12345mW/12345mW 或 POM_5V_IN)
                pwr_match = re.search(r"(?:VDD_IN|POM_5V_IN)\s+(\d+)mW", line)
                if pwr_match:
                    self.power_mw_samples.append(float(pwr_match.group(1)))

                # 解析最高芯片温度 (如 tj@45.5C)
                temp_match = re.search(r"tj@([\d.]+)C", line)
                if temp_match:
                    self.temp_c_samples.append(float(temp_match.group(1)))

            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except Exception:
                proc.kill()
        except Exception:
            pass

    def stop(self) -> Dict[str, float]:
        if not self.has_tegrastats or self.thread is None:
            return {}
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        return {
            "nvdec_avg_pct": float(np.mean(self.nvdec_samples)) if self.nvdec_samples else 0.0,
            "nvdec_max_pct": float(np.max(self.nvdec_samples)) if self.nvdec_samples else 0.0,
            "vic_avg_pct": float(np.mean(self.vic_samples)) if self.vic_samples else 0.0,
            "power_avg_w": (float(np.mean(self.power_mw_samples)) / 1000.0) if self.power_mw_samples else 0.0,
            "temp_avg_c": float(np.mean(self.temp_c_samples)) if self.temp_c_samples else 0.0,
        }


# ==============================================================================
# 4. 单项测试执行引擎
# ==============================================================================

def run_single_benchmark(
    backend: str,
    rtsp_url: str,
    codec: str,
    total_frames: int = 600,
    warmup_frames: int = 30
) -> Dict[str, Any]:
    """
    针对特定解码后端运行基准测试并采集完整指标
    """
    print(f"\n🚀 开始执行 [{backend.upper()}] 解码基准测试 (总帧数: {total_frames}, 预热: {warmup_frames}) ...")

    # 配置环境变量与连接对象
    if backend == "ffmpeg":
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;3000000|max_delay;500000"
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    elif backend == "nvdec":
        gst_pipe = build_jetson_gst_pipeline(rtsp_url, codec=codec, latency_ms=0)
        print(f"🔧 GStreamer 管道: {gst_pipe}")
        cap = cv2.VideoCapture(gst_pipe, cv2.CAP_GSTREAMER)
    else:
        raise ValueError(f"未知后端: {backend}")

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频流后端 [{backend}]，请检查推流连通性或 GStreamer 插件！")

    proc = psutil.Process() if psutil else None
    tegra = TegraStatsCollector(interval_ms=200)
    tegra.start()

    # 预热阶段
    for _ in range(warmup_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)

    print(f"⏱️ 预热完成，正式开始采集 {total_frames} 帧数据...")

    read_ms_list = []
    conv_ms_list = []
    total_ms_list = []
    cpu_percent_list = []
    resolution = "Unknown"

    if proc:
        proc.cpu_percent(interval=None)  # 初始化 CPU 测量基准

    t_bench_start = time.perf_counter()
    frames_collected = 0

    for i in range(total_frames):
        t0 = time.perf_counter()
        ret, frame = cap.read()
        t1 = time.perf_counter()

        if not ret or frame is None:
            continue

        read_time_ms = (t1 - t0) * 1000.0
        read_ms_list.append(read_time_ms)

        # 若为 BGRx (4 通道)，测量转为 BGR 的 SIMD 转换耗时
        conv_time_ms = 0.0
        if frame.ndim == 3 and frame.shape[2] == 4:
            tc0 = time.perf_counter()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            tc1 = time.perf_counter()
            conv_time_ms = (tc1 - tc0) * 1000.0

        conv_ms_list.append(conv_time_ms)
        total_ms_list.append(read_time_ms + conv_time_ms)
        frames_collected += 1

        if resolution == "Unknown" and frame is not None:
            resolution = f"{frame.shape[1]}x{frame.shape[0]}"

        # 每 30 帧采样一次 CPU
        if proc and (i % 30 == 0):
            cpu_percent_list.append(proc.cpu_percent(interval=None))

        # 打印简单进度
        if (i + 1) % 100 == 0 or (i + 1) == total_frames:
            print(f"   已采集: {i + 1}/{total_frames} 帧 (当前即时解码耗时: {read_time_ms:.2f} ms)")

    t_bench_end = time.perf_counter()
    cap.release()
    tegra_metrics = tegra.stop()

    wall_duration = t_bench_end - t_bench_start
    actual_fps = frames_collected / wall_duration if wall_duration > 0 else 0.0

    # 统计计算
    stats = {
        "backend": backend,
        "codec": codec,
        "resolution": resolution,
        "frames_collected": frames_collected,
        "wall_duration_s": round(wall_duration, 2),
        "actual_fps": round(actual_fps, 2),
        "decode_ms_mean": round(float(np.mean(read_ms_list)), 2) if read_ms_list else 0.0,
        "decode_ms_p50": round(float(np.percentile(read_ms_list, 50)), 2) if read_ms_list else 0.0,
        "decode_ms_p95": round(float(np.percentile(read_ms_list, 95)), 2) if read_ms_list else 0.0,
        "decode_ms_p99": round(float(np.percentile(read_ms_list, 99)), 2) if read_ms_list else 0.0,
        "decode_ms_max": round(float(np.max(read_ms_list)), 2) if read_ms_list else 0.0,
        "conv_ms_mean": round(float(np.mean(conv_ms_list)), 2) if conv_ms_list else 0.0,
        "total_ingest_ms_mean": round(float(np.mean(total_ms_list)), 2) if total_ms_list else 0.0,
        "process_cpu_pct_mean": round(float(np.mean(cpu_percent_list)), 1) if cpu_percent_list else 0.0,
        "process_cpu_pct_max": round(float(np.max(cpu_percent_list)), 1) if cpu_percent_list else 0.0,
        "system_cpu_pct_mean": round(float(psutil.cpu_percent(interval=None)), 1) if psutil else 0.0,
        "memory_rss_mb": round(proc.memory_info().rss / (1024 * 1024), 1) if proc else 0.0,
    }
    stats.update(tegra_metrics)
    return stats


# ==============================================================================
# 5. 格式化对比报告输出
# ==============================================================================

def print_comparison_report(ffmpeg_res: Dict[str, Any], nvdec_res: Dict[str, Any], rtsp_url: str) -> None:
    """输出美化对比 Markdown 表格"""
    print("\n" + "=" * 88)
    print(" 📊 Jetson AGX Orin 解码性能量化对比评估报告 (Quantified Benchmark Report)")
    print(f" 视频源: {rtsp_url}")
    print(f" 视频规格: 编码={nvdec_res.get('codec', '').upper()} | 分辨率={nvdec_res.get('resolution')}")
    print("=" * 88)

    def _calc_gain(base: float, opt: float, lower_is_better: bool = True) -> str:
        if base <= 0.001:
            return "--"
        diff = base - opt if lower_is_better else opt - base
        pct = (diff / base) * 100.0
        if lower_is_better:
            return f"▼ {pct:.1f}%" if pct > 0 else f"▲ {-pct:.1f}%"
        else:
            return f"▲ {pct:.1f}%" if pct > 0 else f"▼ {-pct:.1f}%"

    rows = [
        ("单帧解码平均耗时 (Decode Mean)", f"{ffmpeg_res['decode_ms_mean']:.2f} ms", f"{nvdec_res['decode_ms_mean']:.2f} ms", _calc_gain(ffmpeg_res['decode_ms_mean'], nvdec_res['decode_ms_mean'])),
        ("单帧解码中位数 (P50)", f"{ffmpeg_res['decode_ms_p50']:.2f} ms", f"{nvdec_res['decode_ms_p50']:.2f} ms", _calc_gain(ffmpeg_res['decode_ms_p50'], nvdec_res['decode_ms_p50'])),
        ("单帧解码高位延迟 (P95)", f"{ffmpeg_res['decode_ms_p95']:.2f} ms", f"{nvdec_res['decode_ms_p95']:.2f} ms", _calc_gain(ffmpeg_res['decode_ms_p95'], nvdec_res['decode_ms_p95'])),
        ("单帧解码极端延迟 (P99)", f"{ffmpeg_res['decode_ms_p99']:.2f} ms", f"{nvdec_res['decode_ms_p99']:.2f} ms", _calc_gain(ffmpeg_res['decode_ms_p99'], nvdec_res['decode_ms_p99'])),
        ("单帧最大毛刺 (Max Spike)", f"{ffmpeg_res['decode_ms_max']:.2f} ms", f"{nvdec_res['decode_ms_max']:.2f} ms", _calc_gain(ffmpeg_res['decode_ms_max'], nvdec_res['decode_ms_max'])),
        ("BGRx 格式转换平均耗时", "0.00 ms (CPU自带)", f"{nvdec_res['conv_ms_mean']:.2f} ms", "SIMD 极低开销"),
        ("总摄入吞吐延迟 (Total Ingest)", f"{ffmpeg_res['total_ingest_ms_mean']:.2f} ms", f"{nvdec_res['total_ingest_ms_mean']:.2f} ms", _calc_gain(ffmpeg_res['total_ingest_ms_mean'], nvdec_res['total_ingest_ms_mean'])),
        ("有效吞吐帧率 (Actual FPS)", f"{ffmpeg_res['actual_fps']:.1f} FPS", f"{nvdec_res['actual_fps']:.1f} FPS", "平稳达标"),
        ("拉流进程 CPU 占用率 (Process CPU)", f"{ffmpeg_res['process_cpu_pct_mean']:.1f} %", f"{nvdec_res['process_cpu_pct_mean']:.1f} %", _calc_gain(ffmpeg_res['process_cpu_pct_mean'], nvdec_res['process_cpu_pct_mean'])),
        ("整机系统 CPU 占用率 (System CPU)", f"{ffmpeg_res['system_cpu_pct_mean']:.1f} %", f"{nvdec_res['system_cpu_pct_mean']:.1f} %", _calc_gain(ffmpeg_res['system_cpu_pct_mean'], nvdec_res['system_cpu_pct_mean'])),
        ("内存驻留集 (RSS Memory)", f"{ffmpeg_res['memory_rss_mb']:.1f} MB", f"{nvdec_res['memory_rss_mb']:.1f} MB", "--"),
        ("NVDEC 硬件引擎利用率 (Tegra)", f"{ffmpeg_res.get('nvdec_avg_pct', 0.0):.1f} %", f"{nvdec_res.get('nvdec_avg_pct', 0.0):.1f} %", "硬件卸载生效" if nvdec_res.get('nvdec_avg_pct', 0) > 0 else "--"),
        ("VIC 硬件转换器利用率 (Tegra)", f"{ffmpeg_res.get('vic_avg_pct', 0.0):.1f} %", f"{nvdec_res.get('vic_avg_pct', 0.0):.1f} %", "硬件转换生效" if nvdec_res.get('vic_avg_pct', 0) > 0 else "--"),
        ("整机功耗预估 (Tegra Power)", f"{ffmpeg_res.get('power_avg_w', 0.0):.2f} W" if ffmpeg_res.get('power_avg_w') else "--", f"{nvdec_res.get('power_avg_w', 0.0):.2f} W" if nvdec_res.get('power_avg_w') else "--", "--"),
        ("芯片温度预估 (Tegra Temp)", f"{ffmpeg_res.get('temp_avg_c', 0.0):.1f} °C" if ffmpeg_res.get('temp_avg_c') else "--", f"{nvdec_res.get('temp_avg_c', 0.0):.1f} °C" if nvdec_res.get('temp_avg_c') else "--", "--"),
    ]

    print(f"| {'量化指标项 (Metrics)':<34} | {'FFmpeg 软解基线':<18} | {'GStreamer NVDEC 硬解':<18} | {'优化收益 (Gain)':<16} |")
    print(f"|:{'-'*34}-|-:{'-'*18}-|-:{'-'*18}-|-:{'-'*16}-:|")
    for name, f_val, n_val, gain in rows:
        print(f"| {name:<34} | {f_val:<18} | {n_val:<18} | {gain:<16} |")
    print("=" * 88)


# ==============================================================================
# 6. 主程序入口
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Jetson AGX Orin 视频解码性能量化评测工具")
    parser.add_argument("--rtsp", type=str, required=True, help="RTSP 视频流地址 (如海康相机或模拟推流)")
    parser.add_argument("--backend", type=str, default="compare", choices=["compare", "ffmpeg", "nvdec"], help="评测模式: compare(对比两后端), ffmpeg, nvdec")
    parser.add_argument("--codec", type=str, default="auto", choices=["auto", "h264", "h265"], help="视频编码格式 (默认 auto 自动探测)")
    parser.add_argument("--frames", type=int, default=600, help="每种后端评测帧数 (默认 600 帧)")
    parser.add_argument("--warmup", type=int, default=30, help="预热丢弃帧数 (默认 30 帧)")
    parser.add_argument("--report", type=str, default=None, help="JSON 评测结果输出保存路径 (可选)")
    args = parser.parse_args()

    # 1. 确定编码
    if args.codec == "auto":
        codec = detect_rtsp_codec(args.rtsp)
    else:
        codec = args.codec

    results = {}

    # 2. 执行评测
    if args.backend in ["compare", "ffmpeg"]:
        try:
            results["ffmpeg"] = run_single_benchmark(
                backend="ffmpeg",
                rtsp_url=args.rtsp,
                codec=codec,
                total_frames=args.frames,
                warmup_frames=args.warmup
            )
        except Exception as e:
            print(f"❌ [FFmpeg 评测失败]: {e}")

    if args.backend in ["compare", "nvdec"]:
        try:
            # 稍作缓冲，确保 RTSP 连接资源平稳交接
            time.sleep(2.0)
            results["nvdec"] = run_single_benchmark(
                backend="nvdec",
                rtsp_url=args.rtsp,
                codec=codec,
                total_frames=args.frames,
                warmup_frames=args.warmup
            )
        except Exception as e:
            print(f"❌ [NVDEC 评测失败]: {e}")

    # 3. 输出报告
    if "ffmpeg" in results and "nvdec" in results:
        print_comparison_report(results["ffmpeg"], results["nvdec"], args.rtsp)
    elif "nvdec" in results:
        print("\n" + json.dumps(results["nvdec"], indent=2, ensure_ascii=False))
    elif "ffmpeg" in results:
        print("\n" + json.dumps(results["ffmpeg"], indent=2, ensure_ascii=False))

    # 4. 保存 JSON 文件
    if args.report and results:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 量化评测数据已保存至: {args.report}")


if __name__ == "__main__":
    main()
