#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
受电弓网视频故障监测系统 - 运行态推流性能实时监控终端 (monitor_live_perf.py).

用途:
配合【测试方式二】使用。当在 Jetson 端通过 ./start.sh 启动主程序后，
在另一个终端运行此脚本，即可毫秒级动态监视当前视频流处理的真实吞吐率 (FPS)、
全链路单帧时延 (Mean/P50/P95/Max)、GPU 显存占用、主机物理内存占用与积压状态。

使用示例:
    # 动态实时监控 (默认每 2 秒刷新一次控制台仪表盘):
    python3 tools/monitor_live_perf.py

    # 监控 60 秒后自动退出并导出性能评测汇总:
    python3 tools/monitor_live_perf.py --duration 60 --export local_cache/reports/live_stream_summary.md
"""

import os
import sys
import time
import json
import glob
import argparse
from typing import Optional, Dict, Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_latest_report_json(report_dir: str) -> Optional[str]:
    """寻找最新的实时推流性能 JSON 归档文件"""
    candidates = glob.glob(os.path.join(report_dir, "live_perf_report_*.json"))
    if not candidates:
        # 尝试查找 final 或通用 json
        candidates = glob.glob(os.path.join(report_dir, "*.json"))
    if not candidates:
        return None
    # 按修改时间倒序
    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return candidates[0]


def render_dashboard(data: dict, json_path: str, interval_sec: float) -> None:
    """在终端渲染工业仪表盘"""
    # 清屏 (跨平台兼容)
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

    e2e = data.get("latency_e2e_ms", {})
    wall = data.get("latency_main_wall_ms", {})
    ingest = data.get("latency_ingest_ms", {})

    fps = data.get("throughput_fps", 0.0)
    nominal_fps = data.get("source_nominal_fps", 25.0)
    rt_ratio = data.get("realtime_ratio_pct", 0.0)
    total_frames = data.get("total_frames", 0)
    elapsed = data.get("total_elapsed_sec", 0.0)

    vram_alloc = data.get("gpu_vram_allocated_mb", 0.0)
    vram_peak = data.get("gpu_vram_peak_mb", 0.0)
    ram_rss = data.get("host_ram_rss_mb", 0.0)
    ram_peak = data.get("host_ram_peak_mb", 0.0)

    detectors = data.get("detectors", {})

    term_width = 72
    sep_double = "═" * term_width
    sep_single = "─" * term_width

    print(sep_double)
    print(" 🚅 动车组受电弓网视频故障监测 - 运行态推流性能实时大盘")
    print(f" 数据来源: {os.path.basename(json_path)} | 采样刷新: {interval_sec:.1f}s")
    print(sep_double)

    print(f"【运行状态】 累计处理帧数: {total_frames:<8d} | 持续运行时长: {elapsed:.1f}s")
    fps_status = "✅ 实时达标 (无积压)" if fps >= (nominal_fps * 0.95) else "⚠️ 算法过载 (推流积压)"
    print(f"【吞吐能力】 当前吞吐: {fps:>6.2f} FPS | 源标称: {nominal_fps:>4.1f} FPS | 实时比: {rt_ratio:>5.1f}% [{fps_status}]")
    print(sep_single)

    print("【全链路单帧处理时延 (Latency Profiling)】")
    print(f"  • 全链路单帧总延迟 (E2E): 均值={e2e.get('mean', 0.0):>5.1f}ms | P50={e2e.get('p50', 0.0):>5.1f}ms | P95={e2e.get('p95', 0.0):>5.1f}ms | 峰值={e2e.get('max', 0.0):>5.1f}ms")
    print(f"    ├─ 视频拉流解码 (Ingest): 均值={ingest.get('mean', 0.0):>5.1f}ms | P95={ingest.get('p95', 0.0):>5.1f}ms")
    print(f"    └─ 算法与主循环 (Wall)  : 均值={wall.get('mean', 0.0):>5.1f}ms | P95={wall.get('p95', 0.0):>5.1f}ms")

    budget_pct = (e2e.get('mean', 0.0) / 33.33 * 100.0) if e2e.get('mean') else 0.0
    budget_bar_len = min(int(budget_pct / 5), 20)
    budget_bar = "█" * budget_bar_len + "░" * (20 - budget_bar_len)
    print(f"  • 33.3ms 单帧预算占用率: [{budget_bar}] {budget_pct:.1f}% ({'安全' if budget_pct <= 100 else '超标'})")
    print(sep_single)

    print("【硬件资源动态占用 (Hardware Footprint)】")
    print(f"  • GPU 显存 (VRAM): 当前分配={vram_alloc:>6.1f} MB | 运行峰值={vram_peak:>6.1f} MB")
    print(f"  • 系统内存 (RSS) : 当前常驻={ram_rss:>6.1f} MB | 运行峰值={ram_peak:>6.1f} MB")
    print(sep_single)

    if detectors:
        print("【各故障检测器单体推理延迟明细】")
        for d_name, d_info in sorted(detectors.items()):
            print(f"  • {d_name:<18s}: 均值={d_info.get('avg_ms', 0.0):>5.1f}ms | 峰值={d_info.get('max_ms', 0.0):>5.1f}ms | 运行={d_info.get('runs', 0):>4d} 帧")
        print(sep_double)

    print(" 💡 提示: 按 Ctrl+C 可停止监控 (后台 main.py 监测服务将不受影响持续运行)")


def main():
    parser = argparse.ArgumentParser(description="动车组弓网监测系统 - 运行态性能动态监视器")
    parser.add_argument("--dir", type=str, default="local_cache/reports", help="实时性能报告目录")
    parser.add_argument("--interval", type=float, default=2.0, help="终端刷新间隔 (秒)")
    parser.add_argument("--duration", type=int, default=0, help="监视总时长 (秒，0表示持续监控直至Ctrl+C)")
    parser.add_argument("--export", type=str, default=None, help="退出时复制一份最新的Markdown报告至指定文件")
    args = parser.parse_args()

    report_dir = os.path.abspath(args.dir)
    print(f"🔍 正在连接弓网监测运行态性能数据源: {report_dir} ...")

    start_t = time.time()
    last_mtime = 0

    try:
        while True:
            json_file = find_latest_report_json(report_dir)
            if json_file and os.path.exists(json_file):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    render_dashboard(data, json_file, args.interval)
                except Exception as e:
                    print(f"⚠️ 读取性能数据文件解析中: {e}")
            else:
                print(f"⏳ 正在等待 main.py 首次输出性能报告数据 (请确认已执行 ./start.sh)... [重试中]")

            if args.duration > 0 and (time.time() - start_t) >= args.duration:
                print(f"\n✅ 已达到设定的监控时长 ({args.duration} 秒)，正常退出。")
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\n⏹ 监控已平稳停止。")

    if args.export:
        latest_json = find_latest_report_json(report_dir)
        if latest_json:
            md_candidate = os.path.splitext(latest_json)[0] + ".md"
            if os.path.exists(md_candidate):
                import shutil
                os.makedirs(os.path.dirname(os.path.abspath(args.export)), exist_ok=True)
                shutil.copyfile(md_candidate, args.export)
                print(f"📄 已成功将最新性能评测报告归档至: {args.export}")


if __name__ == "__main__":
    main()
