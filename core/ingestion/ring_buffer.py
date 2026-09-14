#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ring Frame Buffer & Video Recording Engine.
Buffers JPEG frames for pre-60s/post-60s incident video generation.
"""

import os
import cv2
import time
import logging
from collections import deque
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class RingFrameBuffer:
    """JPEG 压缩环形缓冲与前后 60 秒录像引擎"""

    def __init__(self, fps: float = 25.0, pre_record_sec: int = 60, post_record_sec: int = 60, jpeg_quality: int = 80):
        self.fps = fps
        self.pre_record_sec = pre_record_sec
        self.post_record_sec = post_record_sec
        self.jpeg_quality = jpeg_quality

        # 最大缓存容量：前60s + 后60s + 30s 冗余
        max_capacity = int((pre_record_sec + post_record_sec + 30) * fps)
        self.buffer = deque(maxlen=max_capacity)
        self.pending_tasks: List[Dict[str, Any]] = []

    def append(self, timestamp: float, frame: Any) -> None:
        """编码并存入环形队列"""
        success, jpeg_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if success:
            self.buffer.append((timestamp, jpeg_bytes))

    def trigger_recording(self, event_name: str, fault_code: int, fault_ts: float, save_dir: str, base_name: str) -> None:
        """故障发生时注册前后视频合成任务"""
        start_ts = fault_ts - self.pre_record_sec
        end_ts = fault_ts + self.post_record_sec

        # 抽取缓存中 >= start_ts 的帧作为 pre_frames
        pre_frames = [buf for ts, buf in self.buffer if ts >= start_ts]

        task = {
            "event_name": event_name,
            "fault_code": fault_code,
            "fault_ts": fault_ts,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "save_dir": save_dir,
            "base_name": base_name,
            "pre_frames": pre_frames,
            "post_frames": [],
        }
        self.pending_tasks.append(task)
        logger.info(f"[RingBuffer] 启动录像任务: {base_name} (覆盖前{len(pre_frames)}帧，持续录制后{self.post_record_sec}秒)")

    def collect_post_frame(self, current_ts: float, frame: Any) -> None:
        """为所有激活的录像任务收集故障后帧"""
        if not self.pending_tasks:
            return

        success, jpeg_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not success:
            return

        for task in self.pending_tasks:
            if task["fault_ts"] < current_ts <= task["end_ts"]:
                task["post_frames"].append(jpeg_bytes)

    def finalize_recordings(self, current_time: float) -> List[str]:
        """检查并合成已达到 end_ts 的完整前后视频"""
        completed_videos = []
        still_pending = []

        for task in self.pending_tasks:
            if current_time >= task["end_ts"]:
                all_jpegs = task["pre_frames"] + task["post_frames"]
                if all_jpegs:
                    try:
                        first_img = cv2.imdecode(all_jpegs[0], cv2.IMREAD_COLOR)
                        h, w = first_img.shape[:2]
                        os.makedirs(task["save_dir"], exist_ok=True)
                        video_path = os.path.join(task["save_dir"], f"{task['base_name']}.mp4")

                        # 动态计算有效帧率保证播放时长 ≈ 真实经历时长
                        total_span = max(1.0, task["end_ts"] - task["start_ts"])
                        effective_fps = len(all_jpegs) / total_span

                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(video_path, fourcc, float(effective_fps), (w, h))

                        for j_bytes in all_jpegs:
                            img = cv2.imdecode(j_bytes, cv2.IMREAD_COLOR)
                            if img is not None:
                                writer.write(img)
                        writer.release()

                        logger.info(f"[RingBuffer] 完整前后视频已成功合成落盘: {video_path} ({len(all_jpegs)}帧, {effective_fps:.1f} FPS)")
                        completed_videos.append(video_path)

                    except Exception as e:
                        logger.error(f"[RingBuffer] 合成完整视频失败: {e}", exc_info=True)
            else:
                still_pending.append(task)

        self.pending_tasks = still_pending
        return completed_videos
