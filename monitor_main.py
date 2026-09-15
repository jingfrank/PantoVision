#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动车组弓网监测子系统 - AI 监测服务主入口 (monitor_main.py)
整合：
1. YOLOv11 受电弓 AI 实时故障识别引擎 (ai_detector.py)
2. 车载 UDP 8080 端口 0xFF 0x20 标准故障告警上报
3. PHM 心跳上报 + 故障结果上报 (算法 ID 10008, phm_service.py)
"""

import os
import sys
import time
import yaml
import socket
import signal
import logging
import threading
import logging.handlers

src_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(src_dir, "config_prod.yaml")) or os.path.exists(os.path.join(src_dir, "AppConfig.xlsx")):
    app_dir = src_dir
else:
    app_dir = os.path.dirname(src_dir)
sys.path.insert(0, app_dir)
sys.path.insert(0, src_dir)

from ai_detector import PantographFaultDetector
from phm_service import create_monitor_phm_reporter
from ftp_uploader import init_ftp_uploader

stop_event = threading.Event()


def signal_handler(signum, frame):
    print("\n[监测服务] 收到终止信号，正在安全停止 AI 故障识别引擎...")
    stop_event.set()


def load_config():
    candidates = [
        os.path.join(app_dir, "config", "config_prod.yaml"),
        os.path.join(app_dir, "config_prod.yaml"),
        os.path.join(app_dir, "config", "config_dev.yaml"),
        os.path.join(app_dir, "config_dev.yaml"),
    ]
    default_cfg_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
    config_file = sys.argv[1] if len(sys.argv) > 1 and os.path.exists(sys.argv[1]) else default_cfg_path

    if not os.path.exists(config_file):
        print(f"[CRITICAL] 无法找到配置文件: {config_file}", file=sys.stderr)
        sys.exit(1)

    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f), config_file


def setup_logging(log_cfg: dict, app_dir: str) -> logging.Logger:
    """
    根据配置初始化 Python logging 系统。

    log_config 示例:
        log_config:
          enabled: true           # 总开关：false 时仅打印到控制台
          level: "INFO"           # DEBUG / INFO / WARNING / ERROR
          log_dir: ""             # 留空则自动使用 local_cache/logs（板卡用 /hirain/hi_opt/algo_log）
          max_bytes: 10485760     # 单个文件上限 10 MB
          backup_count: 5         # 保留最多 5 个轮转备份

    返回配置好的根 logger。
    """
    enabled     = log_cfg.get("enabled", True)
    level_str   = log_cfg.get("level", "INFO").upper()
    log_dir_cfg = log_cfg.get("log_dir", "")
    max_bytes   = int(log_cfg.get("max_bytes", 10 * 1024 * 1024))   # 默认 10 MB
    backup_cnt  = int(log_cfg.get("backup_count", 5))

    level = getattr(logging, level_str, logging.INFO)

    # 格式：时间 [级别] logger名 - 消息
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    # 清除可能已存在的 handlers（避免重复）
    root_logger.handlers.clear()

    # ── 控制台 Handler（始终添加）──────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    # ── 文件 Handler（仅在 enabled=true 时添加）────────────────────────
    if enabled:
        # 解析日志目录：优先使用配置值，其次自动探测板卡/开发机路径
        if log_dir_cfg:
            log_dir = os.path.join(app_dir, log_dir_cfg) if not os.path.isabs(log_dir_cfg) else log_dir_cfg
        else:
            try:
                from common.config import resolve_log_root
                log_dir = os.path.join(resolve_log_root(), "monitor")
            except ImportError:
                log_dir = os.path.join(app_dir, "local_cache", "logs", "monitor")

        try:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "monitor.log")

            # 按大小轮转（RotatingFileHandler），防止单文件过大
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_cnt,
                encoding="utf-8",
            )
            file_handler.setFormatter(fmt)
            file_handler.setLevel(level)
            root_logger.addHandler(file_handler)

            print(f"[日志] 文件日志已启用: {log_file}  (上限 {max_bytes//1024//1024}MB × {backup_cnt} 份)")
        except OSError as e:
            print(f"[日志] 文件日志初始化失败（回退到仅控制台）: {e}")
    else:
        print("[日志] 文件日志已关闭（log_config.enabled=false）")

    return root_logger


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    global_cfg, config_file = load_config()

    # ── 初始化日志系统（最先执行，确保后续所有模块的日志都被捕获）──────
    log_cfg = global_cfg.get("log_config", {})
    logger = setup_logging(log_cfg, app_dir)

    logger.info("=" * 80)
    logger.info(" 动车组弓网监测子系统 - AI 监测服务启动 (Monitor) ")
    logger.info(f" 载入配置文件: {config_file}")
    logger.info("=" * 80)

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    init_ftp_uploader(global_cfg.get("ftp_config", {}))

    phm_reporter = None
    ai_threads = []
    detectors = []
    vs_cfg = global_cfg.get("video_stream_config", {})
    streams = vs_cfg.get("streams", [])

    engine_type = global_cfg.get("base_algorithm_config", {}).get("engine_type", "")
    shared_vlm_engine = None
    if "vlm" in str(engine_type).lower():
        try:
            from twostage_vlm_engine import TwoStageVLMEngine
            shared_vlm_engine = TwoStageVLMEngine(global_cfg.get("base_algorithm_config", {}))
        except Exception as e:
            logger.warning(f"VLM engine load skipped: {e}")

    for stream_cfg in streams:
        detector = PantographFaultDetector(global_cfg, stream_cfg, udp_sock, stop_event, vlm_engine=shared_vlm_engine)
        detectors.append(detector)
        t = threading.Thread(
            target=detector.run,
            daemon=True,
            name=f"AI-Detector-CAR{detector.car_number}-CAM{detector.camera_id}"
        )
        ai_threads.append(t)
        t.start()
        logger.info(f"[AI-识别引擎已拉起] 流地址: {stream_cfg.get('rtsp_url')} -> 算法结果路径: {stream_cfg.get('result_base_dir')}")

    if not ai_threads:
        logger.warning("未配置任何 video_stream_config.streams，监测服务无工作流。")

    phm_reporter = create_monitor_phm_reporter(global_cfg, detectors, package_dir=app_dir)
    if phm_reporter:
        phm_reporter.start()

    logger.info("✅ [监测服务] AI 故障识别引擎与 PHM 心跳/故障上报已成功运行！")

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    finally:
        logger.info("[监测服务] 正在关闭 AI 识别引擎...")
        stop_event.set()
        if phm_reporter:
            phm_reporter.stop()
        udp_sock.close()
        logger.info("[监测服务] 安全退出完成。")


if __name__ == "__main__":
    main()
