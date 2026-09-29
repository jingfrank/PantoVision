#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Ingestion Layer: Stream Reader.
Provides asynchronous RTSP capture with automatic reconnection and offline video file reading.
"""

import os
import re
import socket
import subprocess
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


# ==============================================================================
# RTSP 视频流编码探测与 Jetson GStreamer 管道构建
# ==============================================================================

def probe_rtsp_codec_socket(rtsp_url: str, timeout: float = 1.5) -> Optional[str]:
    """
    通过原生 Socket 发送 RTSP DESCRIBE 请求解析 SDP 媒体行。
    零依赖、毫秒级响应，适用于无密码的模拟推流 (EasyDarwin / FFmpeg / 部分相机)。
    """
    try:
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
                f"User-Agent: GongwangMonitor/1.0\r\n\r\n"
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
            return None

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


def detect_rtsp_codec(rtsp_url: str, fallback: str = "h264") -> str:
    """
    综合探测 RTSP 流编码格式，优先 Socket 毫秒探测，失败则回退至 ffprobe 鉴权探测。
    """
    codec = probe_rtsp_codec_socket(rtsp_url, timeout=1.0)
    if codec:
        return codec
    codec = probe_rtsp_codec_ffprobe(rtsp_url, timeout=3.0)
    if codec:
        return codec
    return fallback


def build_jetson_gst_pipeline(rtsp_url: str, codec: str = "h264", latency_ms: int = 0) -> str:
    """
    构造 Jetson AGX Orin 专用的硬件解码与格式转换管道。
    关键优化点:
    1. latency={latency_ms} + drop-on-latency=true 消除网络抖动积压；
    2. nvv4l2decoder enable-max-performance=1 drop-frame-interval=0 极速硬解；
    3. nvvidconv 直接硬件输出 BGRx 规避 CPU videoconvert 瓶颈；
    4. appsink drop=1 max-buffers=1 保持实时单帧吞吐。
    """
    depay = "rtph264depay ! h264parse" if codec == "h264" else "rtph265depay ! h265parse"
    return (
        f"rtspsrc location={rtsp_url} latency={latency_ms} protocols=tcp drop-on-latency=true ! "
        f"{depay} ! "
        f"nvv4l2decoder enable-max-performance=1 drop-frame-interval=0 ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )


def _open_capture(
    rtsp_url: str,
    decoder_backend: str,
    codec_type: str,
    gstreamer_latency_ms: int,
    worker_logger: logging.Logger
) -> Tuple[Optional[cv2.VideoCapture], str]:
    """
    建立视频连接，优先尝试 Jetson GStreamer NVDEC 硬件解码，
    若失败则自动尝试倒换编码或平滑降级至 FFmpeg 软件解码。
    """
    if decoder_backend in ("gstreamer_nvdec", "nvdec", "auto"):
        target_codec = codec_type
        if target_codec == "auto":
            worker_logger.info(f"[RTSP-Worker] 正在自适应探测视频编码: {rtsp_url} ...")
            target_codec = detect_rtsp_codec(rtsp_url, fallback="h264")
            worker_logger.info(f"[RTSP-Worker] 视频编码探测命中: {target_codec.upper()}")

        gst_pipe = build_jetson_gst_pipeline(rtsp_url, codec=target_codec, latency_ms=gstreamer_latency_ms)
        worker_logger.info(f"[RTSP-Worker] 尝试启动 Jetson NVDEC 硬解管道 ({target_codec.upper()}): {gst_pipe}")
        cap = cv2.VideoCapture(gst_pipe, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            worker_logger.info(f"[RTSP-Worker] ✅ 成功开启 Jetson NVDEC 硬件解码 ({target_codec.upper()})")
            return cap, "nvdec"

        # 若失败且为 auto，尝试倒换编码 (h264 <-> h265)
        if codec_type == "auto":
            alt_codec = "h265" if target_codec == "h264" else "h264"
            worker_logger.warning(f"[RTSP-Worker] NVDEC {target_codec.upper()} 开启失败，尝试倒换至 {alt_codec.upper()} 硬解...")
            gst_pipe_alt = build_jetson_gst_pipeline(rtsp_url, codec=alt_codec, latency_ms=gstreamer_latency_ms)
            cap_alt = cv2.VideoCapture(gst_pipe_alt, cv2.CAP_GSTREAMER)
            if cap_alt.isOpened():
                worker_logger.info(f"[RTSP-Worker] ✅ 倒换成功，已开启 Jetson NVDEC 硬件解码 ({alt_codec.upper()})")
                return cap_alt, "nvdec"

        worker_logger.warning("[RTSP-Worker] Jetson NVDEC 硬件解码不可用或初始化失败，平滑降级至 FFmpeg 软件解码...")

    # FFmpeg 软件解码回退/原生模式
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;3000000|max_delay;500000"
    cap_ff = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if cap_ff.isOpened():
        worker_logger.info("[RTSP-Worker] ✅ 成功开启 FFmpeg 软件解码连接")
        return cap_ff, "ffmpeg"

    return None, "none"


def _rtsp_capture_worker(
    rtsp_url: str,
    car_number: int,
    camera_id: int,
    train_id: str,
    sample_interval: float,
    frame_queue: Any,
    stop_event: Any,
    shared_heartbeat: Any,
    decoder_backend: str = "gstreamer_nvdec",
    codec_type: str = "auto",
    gstreamer_latency_ms: int = 0,
) -> None:
    """
    RTSP 独立拉流子进程入口函数：
    1. 完全物理隔离底层 C/C++ 动态链接库的潜在 SIGSEGV 段错误；
    2. 支持 Jetson AGX Orin 专用 GStreamer NVDEC 硬件解码 (H.264/H.265 自适应)；
    3. 若硬件解码异常自动平滑回退至 FFmpeg 软件解码；
    4. 子进程独立规范化日志输出，清晰感知流连通状态；
    5. 5 秒标准重试周期，并在循环中持续刷新 shared_heartbeat，彻底消除误杀死循环。
    """
    # ── 初始化子进程独立日志 ──────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    worker_logger = logging.getLogger("core.ingestion.stream_reader.worker")

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
        cap, active_backend = _open_capture(
            rtsp_url=rtsp_url,
            decoder_backend=decoder_backend,
            codec_type=codec_type,
            gstreamer_latency_ms=gstreamer_latency_ms,
            worker_logger=worker_logger
        )
        shared_heartbeat.value = time.time()

        if cap is None or not cap.isOpened():
            # ── 阻塞点 2: 释放资源双侧刷新心跳 ──────────────────────────
            shared_heartbeat.value = time.time()
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            shared_heartbeat.value = time.time()

            worker_logger.warning(f"[RTSP-Worker] 无法打开 RTSP 流: {rtsp_url} (推流未就绪)，5秒后重试...")
            _interruptible_sleep(5.0, step=0.5)
            continue

        worker_logger.info(f"[RTSP-Worker] ✅ 成功建立流连接 (后端: {active_backend})，进入实时读帧")
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

            # 处理 BGRx 4 通道格式（nvvidconv 硬件转换输出），快速转换为标准 3 通道 BGR
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

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
        hang_timeout_sec: float = 30.0,
        decoder_backend: str = "gstreamer_nvdec",
        codec_type: str = "auto",
        gstreamer_latency_ms: int = 0,
    ):
        self.rtsp_url = rtsp_url
        self.car_number = car_number
        self.camera_id = camera_id
        self.train_id = train_id
        self.sample_interval = sample_interval
        self.hang_timeout_sec = hang_timeout_sec
        self.decoder_backend = decoder_backend
        self.codec_type = codec_type
        self.gstreamer_latency_ms = gstreamer_latency_ms
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
                    self.decoder_backend,
                    self.codec_type,
                    self.gstreamer_latency_ms,
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
