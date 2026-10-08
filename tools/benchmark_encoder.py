#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson AGX Orin 视频编码性能量化评测工具 (benchmark_encoder.py).

定位:
聚焦核心量化指标，直接对比 硬件编码 (Jetson NVENC) 与 CPU 软件编码 (x264 / mp4v)。

核心量化评估项 (Core Metrics):
1. 速度与吞吐: 编码总耗时 (s)、编码吞吐率 (FPS)、实时比率 (RTF, 越小越快);
2. CPU 资源消耗: 进程 CPU 占用率 (均值/峰值 %)、整机 CPU 占用率 (%);
3. Jetson 硬件卸载状态: NVENC 专用引擎利用率 (%)、整机功耗 (W)、芯片温度 (°C);
4. 输出视频规格: 文件体积 (MB)、实际压缩码率 (Mbps)。

使用示例:
    # 核心指标直接对比 (测试 600 帧 1080P，约 20 秒视频)
    python3 tools/benchmark_encoder.py --frames 600

    # 指定现有真实视频测试
    python3 tools/benchmark_encoder.py --video ../pantograph_loop_0.5s.mp4 --frames 600

    # 单独测试特定后端并导出 JSON
    python3 tools/benchmark_encoder.py --backend nvenc --frames 600 --report ./local_cache/enc_report.json
"""

import os
import sys
import time
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
# 1. Jetson 硬件传感器后台采样器 (TegraStats Collector)
# ==============================================================================

class TegraStatsCollector:
    """Jetson 硬件传感器后台采样线程 (实时采集 NVENC 引擎负荷、整机功耗与温度)"""

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.nvenc_samples: List[float] = []
        self.power_mw_samples: List[float] = []
        self.temp_c_samples: List[float] = []
        self.has_tegrastats = os.path.exists("/usr/bin/tegrastats")

    def start(self) -> None:
        if not self.has_tegrastats:
            return
        self.stop_event.clear()
        self.nvenc_samples.clear()
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

                # 解析 NVENC 硬件负荷 (如 NVENC 45% 或 NVENC 0%)
                nvenc_match = re.search(r"NVENC\s+(\d+)%", line)
                if nvenc_match:
                    self.nvenc_samples.append(float(nvenc_match.group(1)))

                # 解析整机功耗 (如 VDD_IN 18500mW 或 POM_5V_IN)
                pwr_match = re.search(r"(?:VDD_IN|POM_5V_IN)\s+(\d+)mW", line)
                if pwr_match:
                    self.power_mw_samples.append(float(pwr_match.group(1)))

                # 解析最高芯片温度 (如 tj@52.5C)
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
            "nvenc_avg_pct": float(np.mean(self.nvenc_samples)) if self.nvenc_samples else 0.0,
            "nvenc_max_pct": float(np.max(self.nvenc_samples)) if self.nvenc_samples else 0.0,
            "power_avg_w": (float(np.mean(self.power_mw_samples)) / 1000.0) if self.power_mw_samples else 0.0,
            "temp_avg_c": float(np.mean(self.temp_c_samples)) if self.temp_c_samples else 0.0,
        }


# ==============================================================================
# 2. 帧数据源生成 (预加载至内存，彻底消除磁盘读 I/O 干扰)
# ==============================================================================

def prepare_frame_pool(video_path: Optional[str], width: int, height: int, pool_size: int = 60) -> List[np.ndarray]:
    """
    预先在内存中准备连续动态帧池（默认 60 帧，约 2 秒 1080P），
    编码测试时循环从 RAM 提取，确保量化测量的是 100% 纯编码性能，绝不掺杂磁盘读延迟。
    """
    frames = []

    # 1. 尝试从指定或临近视频提取
    candidates = []
    if video_path and os.path.isfile(video_path):
        candidates.append(video_path)
    candidates.extend([
        "../pantograph_loop_0.5s.mp4",
        "../pantograph_tilt.mp4",
        "./pantograph_loop_0.5s.mp4",
    ])

    chosen_video = next((p for p in candidates if os.path.isfile(p)), None)

    if chosen_video:
        print(f"🎬 [数据源准备] 从现有测试视频加载帧池: {chosen_video}")
        cap = cv2.VideoCapture(chosen_video)
        while len(frames) < pool_size and cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            frames.append(frame)
        cap.release()

    # 2. 若帧数不足，自动生成带有复杂纹理与运动的真实感高负荷 1080P 合成帧
    if len(frames) < pool_size:
        print(f"🎨 [数据源准备] 正在生成 {pool_size} 帧动态纹理图像 (分辨率: {width}x{height}) ...")
        for i in range(pool_size):
            # 创建渐变底色
            img = np.zeros((height, width, 3), dtype=np.uint8)
            t = i / pool_size
            img[:, :, 0] = int(40 + 60 * np.sin(t * 2 * np.pi))
            img[:, :, 1] = int(50 + 50 * np.cos(t * 2 * np.pi))
            img[:, :, 2] = int(70 + 40 * np.sin(t * np.pi))

            # 模拟受电弓横梁与接触线 (高频边缘，考量运动搜索)
            y_wire = int(height * 0.35 + 50 * np.sin(t * 4 * np.pi))
            cv2.line(img, (0, y_wire), (width, y_wire + 10), (220, 220, 220), 4)

            # 模拟移动滑板与火花块
            x_box = int((width - 400) * ((i % pool_size) / pool_size))
            cv2.rectangle(img, (x_box, y_wire - 40), (x_box + 400, y_wire + 60), (0, 165, 255), -1)

            # 叠加动态时间戳文字
            ts_str = f"FRAME-{i:04d} {time.strftime('%Y-%m-%d %H:%M:%S')}.{int(t*1000):03d}"
            cv2.putText(img, ts_str, (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)

            frames.append(img)

    print(f"✅ [数据源就绪] 内存帧池就绪 (共 {len(frames)} 帧，RAM 占用约 {len(frames)*width*height*3/(1024*1024):.1f} MB)")
    return frames


# ==============================================================================
# 3. 编码器工厂 (针对 Jetson AGX Orin 深度优化)
# ==============================================================================

def create_video_writer(
    backend: str,
    out_path: str,
    fps: float,
    width: int,
    height: int,
    bitrate_bps: int = 4000000
) -> Tuple[Optional[cv2.VideoWriter], str]:
    """
    创建指定后端的 VideoWriter 实例
    """
    escaped_path = out_path.replace('"', '\\"')

    if backend == "nvenc":
        # 管道 1: Jetson 专用 NVMM 零软转硬件硬编管道
        pipeline_nvmm = (
            f"appsrc ! "
            f"video/x-raw, format=BGR ! "
            f"queue ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGRx ! "
            f"nvvidconv ! "
            f"video/x-raw(memory:NVMM), format=NV12 ! "
            f"nvv4l2h264enc bitrate={bitrate_bps} ! "
            f"h264parse ! "
            f"mp4mux ! "
            f"filesink location=\"{escaped_path}\" sync=false"
        )
        try:
            writer = cv2.VideoWriter(pipeline_nvmm, cv2.CAP_GSTREAMER, 0, fps, (width, height))
            if writer is not None and writer.isOpened():
                return writer, "Jetson-NVENC(NVMM)"
        except Exception:
            pass

        # 管道 2: Jetson I420 备用硬件硬编管道
        pipeline_i420 = (
            f"appsrc ! "
            f"video/x-raw, format=BGR ! "
            f"videoconvert ! "
            f"video/x-raw, format=I420 ! "
            f"nvv4l2h264enc bitrate={bitrate_bps} ! "
            f"h264parse ! "
            f"mp4mux ! "
            f"filesink location=\"{escaped_path}\" sync=false"
        )
        try:
            writer = cv2.VideoWriter(pipeline_i420, cv2.CAP_GSTREAMER, 0, fps, (width, height))
            if writer is not None and writer.isOpened():
                return writer, "Jetson-NVENC(I420)"
        except Exception:
            pass
        return None, "nvenc_failed"

    elif backend == "x264":
        # CPU 极速软编 (x264 ultrafast + zerolatency)
        bitrate_kbps = int(bitrate_bps / 1000)
        pipeline_x264 = (
            f"appsrc ! "
            f"video/x-raw, format=BGR ! "
            f"videoconvert ! "
            f"x264enc speed-preset=ultrafast tune=zerolatency bitrate={bitrate_kbps} ! "
            f"h264parse ! "
            f"mp4mux ! "
            f"filesink location=\"{escaped_path}\" sync=false"
        )
        try:
            writer = cv2.VideoWriter(pipeline_x264, cv2.CAP_GSTREAMER, 0, fps, (width, height))
            if writer is not None and writer.isOpened():
                return writer, "CPU-x264(ultrafast)"
        except Exception:
            pass
        return None, "x264_failed"

    elif backend == "mp4v":
        # OpenCV 原生通用软编 (mp4v)
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            if writer is not None and writer.isOpened():
                return writer, "CPU-mp4v(native)"
        except Exception:
            pass
        return None, "mp4v_failed"

    else:
        raise ValueError(f"未知编码后端: {backend}")


# ==============================================================================
# 4. 单项编码测试执行引擎
# ==============================================================================

def run_single_encode_benchmark(
    backend: str,
    frames_pool: List[np.ndarray],
    total_frames: int,
    fps: float,
    width: int,
    height: int,
    bitrate: int,
    output_dir: str
) -> Dict[str, Any]:
    """
    运行单项编码测试并采集速度、CPU、硬件、文件体积
    """
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"bench_{backend}_{int(time.time())}.mp4")

    print(f"\n🚀 开始执行 [{backend.upper()}] 编码基准测试 (编码目标: {total_frames} 帧, 预期码率: {bitrate/1e6:.1f} Mbps) ...")

    writer, writer_desc = create_video_writer(backend, out_file, fps, width, height, bitrate)
    if writer is None or not writer.isOpened():
        raise RuntimeError(f"无法初始化编码器 [{backend}]，管道未打开！")

    print(f"🔧 成功挂载编码管道: {writer_desc} -> {out_file}")

    proc = psutil.Process() if psutil else None
    cpu_percent_list = []
    if proc:
        proc.cpu_percent(interval=None)

    tegra = TegraStatsCollector(interval_ms=100)
    tegra.start()

    # 正式编码计时
    t_start = time.perf_counter()
    pool_len = len(frames_pool)

    for i in range(total_frames):
        frame = frames_pool[i % pool_len]
        writer.write(frame)

        # 周期性采样 CPU
        if proc and (i % 25 == 0):
            cpu_percent_list.append(proc.cpu_percent(interval=None))

        if (i + 1) % 150 == 0 or (i + 1) == total_frames:
            cur_elapsed = time.perf_counter() - t_start
            cur_fps = (i + 1) / cur_elapsed if cur_elapsed > 0 else 0.0
            print(f"   已编码: {i + 1}/{total_frames} 帧 (当前实时编码速度: {cur_fps:.1f} FPS)")

    # 停止写入并完全关闭文件
    writer.release()
    t_end = time.perf_counter()
    tegra_metrics = tegra.stop()

    wall_duration = t_end - t_start
    actual_fps = total_frames / wall_duration if wall_duration > 0 else 0.0
    video_duration_s = total_frames / fps
    rtf = wall_duration / video_duration_s if video_duration_s > 0 else 0.0

    # 统计输出文件体积与码率
    file_size_mb = 0.0
    actual_bitrate_mbps = 0.0
    if os.path.isfile(out_file):
        file_size_bytes = os.path.getsize(out_file)
        file_size_mb = file_size_bytes / (1024 * 1024)
        if video_duration_s > 0:
            actual_bitrate_mbps = (file_size_bytes * 8.0 / video_duration_s) / 1e6

    stats = {
        "backend": backend,
        "writer_desc": writer_desc,
        "total_frames": total_frames,
        "resolution": f"{width}x{height}",
        "video_duration_s": round(video_duration_s, 2),
        "wall_duration_s": round(wall_duration, 2),
        "encoding_fps": round(actual_fps, 1),
        "real_time_factor": round(rtf, 3),
        "process_cpu_pct_mean": round(float(np.mean(cpu_percent_list)), 1) if cpu_percent_list else 0.0,
        "process_cpu_pct_max": round(float(np.max(cpu_percent_list)), 1) if cpu_percent_list else 0.0,
        "system_cpu_pct_mean": round(float(psutil.cpu_percent(interval=None)), 1) if psutil else 0.0,
        "file_size_mb": round(file_size_mb, 2),
        "actual_bitrate_mbps": round(actual_bitrate_mbps, 2),
        "out_file": out_file
    }
    stats.update(tegra_metrics)
    return stats


# ==============================================================================
# 5. 格式化核心指标对比报告
# ==============================================================================

def print_core_comparison_report(results: Dict[str, Dict[str, Any]]) -> None:
    """输出只聚焦核心指标的量化评估表格"""
    print("\n" + "=" * 92)
    print(" 📊 Jetson AGX Orin 视频编码核心性能量化评估报告 (Core Encoding Benchmark)")
    first_res = next(iter(results.values()))
    print(f" 规格: {first_res['resolution']} @ {first_res['total_frames']} 帧 (相当录像时长: {first_res['video_duration_s']} 秒)")
    print("=" * 92)

    has_nvenc = "nvenc" in results
    has_cpu = "x264" in results or "mp4v" in results

    def _calc_gain(base: float, opt: float, lower_is_better: bool = True) -> str:
        if base <= 0.0001:
            return "--"
        diff = base - opt if lower_is_better else opt - base
        pct = (diff / base) * 100.0
        if lower_is_better:
            return f"▼ {pct:.1f}%" if pct > 0 else f"▲ {-pct:.1f}%"
        else:
            return f"▲ {pct:.1f}%" if pct > 0 else f"▼ {-pct:.1f}%"

    base_key = "x264" if "x264" in results else ("mp4v" if "mp4v" in results else None)
    base_res = results.get(base_key) if base_key else None
    nvenc_res = results.get("nvenc")

    headers = ["核心评估指标项 (Core Metrics)"]
    for k in results.keys():
        headers.append(f"{k.upper()}")
    if nvenc_res and base_res:
        headers.append("NVENC 硬编收益")

    print(f"| {headers[0]:<36} | " + " | ".join([f"{h:<14}" for h in headers[1:]]) + " |")
    sep_line = f"|:{'-'*36}-|-" + "-|-".join([f"{'-'*14}:" for _ in headers[1:]]) + "-|"
    print(sep_line)

    metric_defs = [
        ("总编码耗时 (Wall Duration)", "wall_duration_s", "{:.2f} s", True),
        ("编码吞吐帧率 (Encoding FPS)", "encoding_fps", "{:.1f} FPS", False),
        ("实时比率 (RTF, <1为超实时)", "real_time_factor", "{:.3f}", True),
        ("进程平均 CPU 占用率", "process_cpu_pct_mean", "{:.1f} %", True),
        ("进程峰值 CPU 占用率", "process_cpu_pct_max", "{:.1f} %", True),
        ("整机系统 CPU 占用率", "system_cpu_pct_mean", "{:.1f} %", True),
        ("NVENC 专用硬件负荷 (Tegra)", "nvenc_avg_pct", "{:.1f} %", False),
        ("生成视频体积 (File Size)", "file_size_mb", "{:.2f} MB", True),
        ("压缩后实际码率 (Bitrate)", "actual_bitrate_mbps", "{:.2f} Mbps", True),
        ("整机功耗预估 (Power W)", "power_avg_w", "{:.2f} W", True),
        ("芯片最高温度 (Temp °C)", "temp_avg_c", "{:.1f} °C", True),
    ]

    for label, key, fmt, lower_better in metric_defs:
        row_vals = []
        for b_name in results.keys():
            val = results[b_name].get(key, 0.0)
            if val == 0.0 and ("Tegra" in label or "Power" in label or "Temp" in label) and not results[b_name].get(key):
                row_vals.append("--")
            else:
                row_vals.append(fmt.format(val))

        gain_str = ""
        if nvenc_res and base_res:
            v_base = base_res.get(key, 0.0)
            v_nvenc = nvenc_res.get(key, 0.0)
            if v_base > 0.0 and v_nvenc > 0.0:
                gain_str = _calc_gain(v_base, v_nvenc, lower_is_better=lower_better)
            elif "NVENC" in label and v_nvenc > 0:
                gain_str = "硬件卸载生效"
            else:
                gain_str = "--"

        out_row = f"| {label:<36} | " + " | ".join([f"{v:<14}" for v in row_vals])
        if nvenc_res and base_res:
            out_row += f" | {gain_str:<14}"
        out_row += " |"
        print(out_row)

    print("=" * 92)


# ==============================================================================
# 6. 主程序入口
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Jetson AGX Orin 视频编码核心性能量化评测工具")
    parser.add_argument("--backend", type=str, default="compare", choices=["compare", "nvenc", "x264", "mp4v"], help="评测模式: compare(对比全部), nvenc, x264, mp4v")
    parser.add_argument("--frames", type=int, default=600, help="编码测试总帧数 (默认 600 帧，约 20 秒)")
    parser.add_argument("--fps", type=float, default=30.0, help="视频标称帧率 (默认 30.0)")
    parser.add_argument("--width", type=int, default=1920, help="视频宽度 (默认 1920)")
    parser.add_argument("--height", type=int, default=1080, help="视频高度 (默认 1080)")
    parser.add_argument("--bitrate", type=int, default=4000000, help="编码目标码率 bps (默认 4000000，即 4Mbps)")
    parser.add_argument("--video", type=str, default=None, help="现有测试视频路径 (可选，未指定则自动探测或生成动态帧)")
    parser.add_argument("--output-dir", type=str, default="./local_cache/benchmark_encoder", help="临时编码文件输出目录")
    parser.add_argument("--report", type=str, default=None, help="JSON 评测结果输出保存路径 (可选)")
    args = parser.parse_args()

    # 1. 准备帧池
    frames_pool = prepare_frame_pool(args.video, args.width, args.height, pool_size=60)

    results = {}
    backends_to_test = []
    if args.backend == "compare":
        backends_to_test = ["nvenc", "x264", "mp4v"]
    else:
        backends_to_test = [args.backend]

    # 2. 依次测试
    for b in backends_to_test:
        try:
            res = run_single_encode_benchmark(
                backend=b,
                frames_pool=frames_pool,
                total_frames=args.frames,
                fps=args.fps,
                width=args.width,
                height=args.height,
                bitrate=args.bitrate,
                output_dir=args.output_dir
            )
            results[b] = res
            time.sleep(1.0)  # 缓冲降温
        except Exception as e:
            print(f"⚠️ [跳过/失败] 编码后端 [{b}] 执行异常: {e}")

    # 3. 输出核心指标报告
    if results:
        print_core_comparison_report(results)
    else:
        print("❌ 没有成功执行的编码测试！")

    # 4. 保存 JSON
    if args.report and results:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 核心量化评测数据已保存至: {args.report}")


if __name__ == "__main__":
    main()
