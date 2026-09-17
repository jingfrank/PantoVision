#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Ingestion Layer: Stream Reader.
Provides asynchronous RTSP capture with automatic reconnection and offline video file reading.
"""

import os
import time
import cv2
import queue
import threading
import multiprocessing as mp
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any
from core.data_types import FramePacket

logger = logging.getLogger(__name__)

mp_ctx = mp.get_context("spawn")


def _rtsp_capture_worker(
    rtsp_url: str,
    car_number: int,
    camera_id: int,
    train_id: str,
    sample_interval: float,
    frame_queue: Any,
    stop_event: Any,
    shared_heartbeat: Any,
) -> None:
    """
    RTSP 独立拉流子进程入口函数：
    1. 完全物理隔离底层 C/C++ (FFmpeg) 的潜在 SIGSEGV 段错误；
    2. 开局单次注入 TCP 传输与超时选项，彻底杜绝重连循环内重复赋值；
    3. 子进程独立规范化日志输出，清晰感知流连通状态；
    4. 5 秒标准重试周期，并在循环中持续刷新 shared_heartbeat，彻底消除误杀死循环。
    """
    # ── 初始化子进程独立日志 ──────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    worker_logger = logging.getLogger("core.ingestion.stream_reader.worker")

    # ── 1. 在子进程入口单次配置底层环境变量 ─────────────────────────
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;3000000|max_delay;500000"

    def _interruptible_sleep(seconds: float, step: float = 0.5) -> None:
        """分段可中断休眠并持续刷新心跳，防止休眠期间心跳过时且支持秒级快速响应退出"""
        elapsed = 0.0
        while elapsed < seconds and not stop_event.is_set():
            shared_heartbeat.value = time.time()
            sleep_duration = min(step, seconds - elapsed)
            time.sleep(sleep_duration)
            shared_heartbeat.value = time.time()
            elapsed += sleep_duration

    frame_idx = 0
    while not stop_event.is_set():
        worker_logger.info(f"[RTSP-Worker] 正在连接 RTSP 流: {rtsp_url} ...")

        # ── 阻塞点 1: 建连探测双侧刷新心跳 ──────────────────────────────
        shared_heartbeat.value = time.time()
        # 显式声明使用 cv2.CAP_FFMPEG 后端，绕过 GStreamer 管道探测，确保 TCP 传输生效
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        shared_heartbeat.value = time.time()

        if not cap.isOpened():
            # ── 阻塞点 2: 释放资源双侧刷新心跳 ──────────────────────────
            shared_heartbeat.value = time.time()
            try:
                cap.release()
            except Exception:
                pass
            shared_heartbeat.value = time.time()

            worker_logger.warning(f"[RTSP-Worker] 无法打开 RTSP 流: {rtsp_url} (推流未就绪)，5秒后重试...")
            _interruptible_sleep(5.0, step=0.5)
            continue

        worker_logger.info(f"[RTSP-Worker] ✅ 成功连接 RTSP 流: {rtsp_url}，进入实时读帧")
        last_sample = 0.0
        last_valid_frame_time = time.time()
        consecutive_read_fails = 0

        while not stop_event.is_set() and cap.isOpened():
            # ── 阻塞点 3: 核心读帧双侧刷新心跳 ──────────────────────────
            shared_heartbeat.value = time.time()
            ret, frame = cap.read()
            shared_heartbeat.value = time.time()
            now = time.time()

            # ── 软看门狗：读帧失败检测与有效帧超时控制 ───────────────────
            if not ret or frame is None:
                consecutive_read_fails += 1
                if consecutive_read_fails >= 5 or (now - last_valid_frame_time) >= 5.0:
                    worker_logger.warning("[RTSP-Worker] 超过 5 秒无有效帧或连续读取失败，主动重新建立连接...")
                    break
                _interruptible_sleep(0.01, step=0.01)
                continue

            # 校验图像完整性，防止截断的残破帧流向下游引发检测器异常
            if frame.ndim != 3 or frame.shape[0] < 100 or frame.shape[1] < 100:
                continue

            consecutive_read_fails = 0
            last_valid_frame_time = now

            if (now - last_sample) >= sample_interval:
                packet = FramePacket(
                    frame=frame,
                    timestamp=now,
                    frame_idx=frame_idx,
                    car_number=car_number,
                    camera_id=camera_id,
                    train_id=train_id
                )
                frame_idx += 1
                last_sample = now

                # 队列满时丢弃最旧数据以保持生产者通畅
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except Exception:
                        pass
                try:
                    frame_queue.put_nowait(packet)
                except Exception:
                    pass

        # ── 阻塞点 4: 断开释放双侧刷新心跳 ──────────────────────────────
        shared_heartbeat.value = time.time()
        try:
            cap.release()
        except Exception:
            pass
        shared_heartbeat.value = time.time()

        if not stop_event.is_set():
            _interruptible_sleep(5.0, step=0.5)


class BaseStreamSource(ABC):
    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def get_packet(self, timeout: float = 1.0) -> Optional[FramePacket]: ...
    @abstractmethod
    def stop(self) -> None: ...


class RTSPStreamReader(BaseStreamSource):
    """
    子进程原生隔离与心跳守护自愈的 RTSP 视频流拉流采样器。
    特性：
    1. 子进程隔离 (multiprocessing spawn)：底层 FFmpeg SIGSEGV 段错误绝不拖垮主进程；
    2. 共享心跳守护 (shared_heartbeat)：子进程 5s 周期重试正常刷新心跳，Supervisor 绝不误杀；
    3. 死锁强杀 (30s 门限)：仅当子进程陷入底层 C++ 死锁导致超过 30 秒无心跳时才强杀自愈；
    4. 零延迟汲干：get_packet 保证主循环始终处理最新鲜的一帧；
    5. 独立日志体系：子进程连接状态全透明，告别黑盒排查。
    """

    def __init__(
        self,
        rtsp_url: str,
        car_number: int = 6,
        camera_id: int = 0,
        train_id: str = "CR400BF",
        sample_interval: float = 0.033,
        hang_timeout_sec: float = 30.0
    ):
        self.rtsp_url = rtsp_url
        self.car_number = car_number
        self.camera_id = camera_id
        self.train_id = train_id
        self.sample_interval = sample_interval
        self.hang_timeout_sec = hang_timeout_sec
        self.running = False
        self.frame_queue = mp_ctx.Queue(maxsize=10)
        self.stop_event = mp_ctx.Event()
        self.shared_heartbeat = mp_ctx.Value("d", time.time())
        self.process: Optional[mp.Process] = None
        self.supervisor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _start_child_process(self) -> None:
        with self._lock:
            # 清理残留进程句柄
            if self.process is not None and self.process.is_alive():
                try:
                    self.process.terminate()
                    self.process.join(timeout=1.0)
                    if self.process.is_alive():
                        self.process.kill()
                        self.process.join(timeout=0.5)
                except Exception:
                    pass

            self.shared_heartbeat.value = time.time()
            self.process = mp_ctx.Process(
                target=_rtsp_capture_worker,
                args=(
                    self.rtsp_url,
                    self.car_number,
                    self.camera_id,
                    self.train_id,
                    self.sample_interval,
                    self.frame_queue,
                    self.stop_event,
                    self.shared_heartbeat,
                ),
                name=f"RTSPWorker-CAR{self.car_number}-CAM{self.camera_id}",
                daemon=True,
            )
            self.process.start()
            logger.info(
                f"[RTSPStreamReader] 成功拉起拉流子进程: PID={self.process.pid} "
                f"(流地址: {self.rtsp_url}, 独立原生隔离保护生效)"
            )

    def start(self) -> None:
        self.running = True
        self.stop_event.clear()
        self._start_child_process()
        self.supervisor_thread = threading.Thread(
            target=self._supervisor_loop,
            name=f"RTSPSupervisor-CAR{self.car_number}-CAM{self.camera_id}",
            daemon=True
        )
        self.supervisor_thread.start()
        logger.info(f"[RTSPStreamReader] 启动拉流管理与看门狗守护线程 (死锁门限: {self.hang_timeout_sec}s): {self.rtsp_url}")

    def _supervisor_loop(self) -> None:
        """主进程看门狗守护循环：检测子进程原生崩溃或网络底层卡死"""
        while self.running:
            time.sleep(1.0)
            if not self.running:
                break

            # ── 检查 1: 子进程是否异常退出 (SIGSEGV / OOM / 意外退出) ────────
            if self.process is not None and not self.process.is_alive():
                exitcode = self.process.exitcode
                logger.critical(
                    f"[RTSPStreamReader-Supervisor] 探测到拉流子进程异常终止 (exitcode={exitcode})，"
                    f"正在执行秒级自愈重启新拉流子进程..."
                )
                self._start_child_process()
                continue

            # ── 检查 2: 心跳硬看门狗 - 超过 30 秒无心跳 (确凿死锁于底层系统调用) ──
            now = time.time()
            if (now - self.shared_heartbeat.value) > self.hang_timeout_sec:
                logger.warning(
                    f"[RTSPStreamReader-Supervisor] 探测到拉流子进程超过 {self.hang_timeout_sec}s 无心跳刷新（疑似底层 C 库死锁），"
                    f"强杀当前子进程并自愈重启..."
                )
                try:
                    if self.process and self.process.is_alive():
                        self.process.terminate()
                        self.process.join(timeout=1.0)
                        if self.process.is_alive():
                            self.process.kill()
                            self.process.join(timeout=0.5)
                except Exception as kill_e:
                    logger.error(f"[RTSPStreamReader-Supervisor] 终止卡死子进程异常: {kill_e}")
                self._start_child_process()

    def get_packet(self, timeout: float = 1.0) -> Optional[FramePacket]:
        try:
            packet = self.frame_queue.get(timeout=timeout)
            # 零延迟汲干机制：汲干所有积压的旧帧，确保主循环始终处理最新鲜的一帧
            while True:
                try:
                    packet = self.frame_queue.get_nowait()
                except (queue.Empty, Exception):
                    break
            return packet
        except (queue.Empty, Exception):
            return None

    def stop(self) -> None:
        self.running = False
        self.stop_event.set()
        if self.supervisor_thread and self.supervisor_thread.is_alive():
            self.supervisor_thread.join(timeout=2.0)
        with self._lock:
            if self.process and self.process.is_alive():
                self.process.join(timeout=2.0)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=1.0)
        logger.info(f"[RTSPStreamReader] 拉流子进程与守护线程已平稳停止: {self.rtsp_url}")


class VideoFileReader(BaseStreamSource):
    """离线测试视频源 (支持逐帧精确控制与回放)"""

    def __init__(self, file_path: str, car_number: int = 6, camera_id: int = 0, train_id: str = "CR400BF", sample_interval_s: float = 0.04):
        self.file_path = file_path
        self.car_number = car_number
        self.camera_id = camera_id
        self.train_id = train_id
        self.sample_interval_s = sample_interval_s
        self.cap = cv2.VideoCapture(file_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0

    def start(self) -> None:
        pass

    def get_packet(self, timeout: float = 1.0) -> Optional[FramePacket]:
        if not self.cap.isOpened() or self.current_frame >= self.total_frames:
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None

        timestamp = self.current_frame / self.fps
        packet = FramePacket(
            frame=frame,
            timestamp=timestamp,
            frame_idx=self.current_frame,
            car_number=self.car_number,
            camera_id=self.camera_id,
            train_id=self.train_id
        )
        self.current_frame += 1
        return packet

    def stop(self) -> None:
        if self.cap:
            self.cap.release()
