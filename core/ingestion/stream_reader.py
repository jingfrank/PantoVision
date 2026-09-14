#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Ingestion Layer: Stream Reader.
Provides asynchronous RTSP capture with automatic reconnection and offline video file reading.
"""

import time
import cv2
import threading
import queue
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from core.data_types import FramePacket

logger = logging.getLogger(__name__)


class BaseStreamSource(ABC):
    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def get_packet(self, timeout: float = 1.0) -> Optional[FramePacket]: ...
    @abstractmethod
    def stop(self) -> None: ...


class RTSPStreamReader(BaseStreamSource):
    """异步多线程 RTSP 视频流拉流采样器 (带断线自动重连)"""

    def __init__(self, rtsp_url: str, car_number: int = 6, camera_id: int = 0, train_id: str = "CR400BF", sample_interval: float = 0.033):
        self.rtsp_url = rtsp_url
        self.car_number = car_number
        self.camera_id = camera_id
        self.train_id = train_id
        self.sample_interval = sample_interval
        self.running = False
        self.frame_queue = queue.Queue(maxsize=30)
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        logger.info(f"[RTSPStreamReader] 启动拉流线程: {self.rtsp_url}")

    def _worker(self) -> None:
        frame_idx = 0
        while self.running:
            cap = cv2.VideoCapture(self.rtsp_url)
            if not cap.isOpened():
                logger.warning(f"[RTSPStreamReader] 无法打开 RTSP 流: {self.rtsp_url}，5秒后重试...")
                time.sleep(5)
                continue

            logger.info(f"[RTSPStreamReader] 成功连接 RTSP 流: {self.rtsp_url}")
            last_sample = 0.0

            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning("[RTSPStreamReader] 读取帧失败，重新连接...")
                    break

                now = time.time()
                if (now - last_sample) >= self.sample_interval:
                    packet = FramePacket(
                        frame=frame,
                        timestamp=now,
                        frame_idx=frame_idx,
                        car_number=self.car_number,
                        camera_id=self.camera_id,
                        train_id=self.train_id
                    )
                    frame_idx += 1
                    last_sample = now

                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.frame_queue.put(packet)

            cap.release()
            time.sleep(2)

    def get_packet(self, timeout: float = 1.0) -> Optional[FramePacket]:
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)


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
