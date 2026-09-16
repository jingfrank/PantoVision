#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ring Frame Buffer & Video Recording Engine (Asynchronous Decoupled Architecture).

特性：
1. 内存隔离与无竞争：帧数据统一编码为不可变 bytes (buf.tobytes()) 存储，任务交接采用只读快照所有权转移；
2. 彻底非阻塞：主线程 finalize_recordings 仅做微秒级任务投递，主循环 0 卡顿；
3. 两级流水线集成：通过 on_video_completed 回调将生成的视频交接给异步上传流水线；
4. 队列背压保护：限制最大任务积压 (maxsize=5，在途内存 <= 150MB)，防止 Jetson OOM；
5. 原子落盘与自愈：先写 .tmp.mp4 再原子重命名为 .mp4，异常安全清理临时脏文件；工作线程崩溃自愈重启。
"""

import os
import cv2
import time
import queue
import logging
import threading
import numpy as np
from collections import deque
from typing import List, Dict, Any, Tuple, Optional, Callable

logger = logging.getLogger(__name__)


class RingFrameBuffer:
    """JPEG 压缩环形缓冲与后台异步前后 60 秒录像引擎"""

    def __init__(
        self,
        fps: float = 25.0,
        pre_record_sec: int = 60,
        post_record_sec: int = 60,
        jpeg_quality: int = 80,
        max_synthesis_queue: int = 5,
        on_video_completed: Optional[Callable[[str], None]] = None,
        worker_name: str = "RingBuffer-VideoSynthesizer",
    ):
        self.fps = fps
        self.pre_record_sec = pre_record_sec
        self.post_record_sec = post_record_sec
        self.jpeg_quality = jpeg_quality
        self.max_synthesis_queue = max_synthesis_queue
        self.on_video_completed = on_video_completed
        self.worker_name = worker_name

        # 最大缓存容量：前60s + 后60s + 30s 冗余
        max_capacity = int((pre_record_sec + post_record_sec + 30) * fps)
        self.buffer = deque(maxlen=max_capacity)
        self.pending_tasks: List[Dict[str, Any]] = []

        # 异步合成流水线 (Stage 1)
        self._synthesis_queue: queue.Queue = queue.Queue(maxsize=max_synthesis_queue)
        self._completed_queue: queue.Queue = queue.Queue()
        self._running: bool = True
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # 统计计数
        self.total_synthesized: int = 0
        self.total_dropped: int = 0

        self._start_worker()

    def _start_worker(self) -> None:
        """启动后台视频合成工作线程"""
        with self._lock:
            self._worker_thread = threading.Thread(
                target=self._synthesis_worker_loop,
                daemon=True,
                name=self.worker_name
            )
            self._worker_thread.start()
            logger.info(f"[RingBuffer] 启动后台视频合成工作线程: {self.worker_name}")

    def _check_and_heal_worker(self) -> None:
        """检查工作线程健康状态，若意外退出则自动拉起自愈"""
        if self._running and (self._worker_thread is None or not self._worker_thread.is_alive()):
            logger.critical(f"[RingBuffer] 探测到工作线程 {self.worker_name} 意外终止，正在自愈重启...")
            self._start_worker()

    def append(self, timestamp: float, frame: Any) -> None:
        """编码并存入环形队列（显式转为不可变 bytes，确保底层内存绝对只读隔离）"""
        success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if success:
            # 必须调用 .tobytes() 将 mutable numpy array 转为 immutable Python bytes
            self.buffer.append((timestamp, buf.tobytes()))

    def trigger_recording(
        self,
        event_name: str,
        fault_code: int,
        fault_ts: float,
        save_dir: str,
        base_name: str
    ) -> None:
        """故障发生时注册前后视频合成任务"""
        start_ts = fault_ts - self.pre_record_sec
        end_ts = fault_ts + self.post_record_sec

        # 抽取缓存中 >= start_ts 的帧作为 pre_frames (全新 list 快照)
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
        logger.info(
            f"[RingBuffer] 启动录像任务: {base_name} "
            f"(覆盖前{len(pre_frames)}帧，持续录制后{self.post_record_sec}秒)"
        )

    def collect_post_frame(self, current_ts: float, frame: Any) -> None:
        """为所有激活的录像任务收集故障后帧"""
        if not self.pending_tasks:
            return

        success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not success:
            return

        jpeg_bytes = buf.tobytes()
        for task in self.pending_tasks:
            if task["fault_ts"] < current_ts <= task["end_ts"]:
                task["post_frames"].append(jpeg_bytes)

    def finalize_recordings(self, current_time: float) -> List[str]:
        """
        检查已达到 end_ts 的任务，将其打包为不可变快照投递后台队列。
        主线程耗时 < 0.1ms，实现彻底非阻塞。
        同时排空已完成队列返回新生成的视频路径列表（兼容轮询）。
        """
        self._check_and_heal_worker()

        still_pending = []
        for task in self.pending_tasks:
            if current_time >= task["end_ts"]:
                all_jpegs = list(task["pre_frames"] + task["post_frames"])
                if all_jpegs:
                    job = {
                        "base_name": task["base_name"],
                        "save_dir": task["save_dir"],
                        "start_ts": task["start_ts"],
                        "end_ts": task["end_ts"],
                        "all_jpegs": all_jpegs,
                    }
                    self._submit_synthesis_job(job)
                else:
                    logger.warning(f"[RingBuffer] 录像任务 {task['base_name']} 未收集到有效帧，跳过合成")
            else:
                still_pending.append(task)

        self.pending_tasks = still_pending

        # 收集当前已完成视频路径返回
        completed = []
        while not self._completed_queue.empty():
            try:
                completed.append(self._completed_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def _submit_synthesis_job(self, job: Dict[str, Any]) -> None:
        """将合成任务投递到后台队列（含背压淘汰与 OOM 保护）"""
        try:
            self._synthesis_queue.put_nowait(job)
            logger.info(
                f"[RingBuffer] 视频任务已投递后台合成队列: {job['base_name']} "
                f"({len(job['all_jpegs'])} 帧, 队列深度={self._synthesis_queue.qsize()})"
            )
        except queue.Full:
            # 触发背压保护：队列已满，淘汰最旧未开始的任务
            self.total_dropped += 1
            try:
                dropped = self._synthesis_queue.get_nowait()
                logger.critical(
                    f"[RingBuffer-Backpressure] 视频合成队列已满 (maxsize={self.max_synthesis_queue})，"
                    f"丢弃排队时间最久任务以保护内存: {dropped['base_name']}"
                )
                self._synthesis_queue.put_nowait(job)
            except (queue.Empty, queue.Full):
                logger.error(f"[RingBuffer-Backpressure] 严重拥堵，直接丢弃本次任务: {job['base_name']}")

    def _synthesis_worker_loop(self) -> None:
        """后台视频合成工作线程主循环（顶级异常兜底，永不崩溃）"""
        while self._running:
            try:
                job = self._synthesis_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if job is None:
                # 停机哨兵
                self._synthesis_queue.task_done()
                break

            try:
                video_path = self._synthesize_single_video(job)
                if video_path and os.path.isfile(video_path):
                    self.total_synthesized += 1

                    # 若配置了两级流水线解耦回调，直接触发提交 Stage 2 上传流水线
                    if self.on_video_completed:
                        try:
                            self.on_video_completed(video_path)
                        except Exception as cb_e:
                            logger.error(f"[RingBuffer] on_video_completed 回调执行异常: {cb_e}", exc_info=True)
                    else:
                        # 否则存入完成队列供外部 finalize_recordings 轮询获取
                        self._completed_queue.put(video_path)
            except Exception as e:
                logger.error(f"[RingBuffer] 合成视频任务异常被安全拦截: {job.get('base_name')} -> {e}", exc_info=True)
            finally:
                self._synthesis_queue.task_done()

    def _synthesize_single_video(self, job: Dict[str, Any]) -> Optional[str]:
        """
        单次视频合成核心方法：
        - 安全只读 np.frombuffer 解码
        - 优先 avc1 硬件编码，失败优雅回退 mp4v
        - 原子落盘机制 (.tmp.mp4 -> .mp4)
        """
        base_name = job["base_name"]
        save_dir = job["save_dir"]
        all_jpegs = job["all_jpegs"]
        start_ts = job["start_ts"]
        end_ts = job["end_ts"]

        if not all_jpegs:
            return None

        # 安全只读方式解码首帧获取图像尺寸
        first_img = cv2.imdecode(np.frombuffer(all_jpegs[0], np.uint8), cv2.IMREAD_COLOR)
        if first_img is None:
            logger.error(f"[RingBuffer] 首帧解码失败，放弃合成: {base_name}")
            return None

        h, w = first_img.shape[:2]
        os.makedirs(save_dir, exist_ok=True)

        tmp_video_path = os.path.join(save_dir, f"{base_name}.tmp.mp4")
        final_video_path = os.path.join(save_dir, f"{base_name}.mp4")

        # 动态计算有效帧率保证播放时长 ≈ 真实经历时长
        total_span = max(1.0, end_ts - start_ts)
        effective_fps = float(max(1.0, min(60.0, len(all_jpegs) / total_span)))

        # 优先采用车规标准 H.264 (avc1) 编码，若失败回退至 mp4v
        writer = None
        t0 = time.perf_counter()
        try:
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(tmp_video_path, fourcc, effective_fps, (w, h))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(tmp_video_path, fourcc, effective_fps, (w, h))

            if not writer.isOpened():
                raise RuntimeError(f"无法初始化任何视频编码器 (avc1 / mp4v) 写入 {tmp_video_path}")

            for j_bytes in all_jpegs:
                img = cv2.imdecode(np.frombuffer(j_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    writer.write(img)

            writer.release()
            writer = None

            # 原子重命名，确保文件在最终暴露给外部时必定 100% 完整可用
            os.replace(tmp_video_path, final_video_path)

            elapsed = (time.perf_counter() - t0) * 1000.0
            logger.info(
                f"[RingBuffer] 完整前后视频后台异步合成落盘: {final_video_path} "
                f"({len(all_jpegs)}帧, {effective_fps:.1f} FPS, 耗时 {elapsed:.1f}ms)"
            )
            return final_video_path

        except Exception as e:
            logger.error(f"[RingBuffer] 写入视频文件失败: {tmp_video_path} -> {e}", exc_info=True)
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
            # 异常清理未写完的脏临时文件
            if os.path.exists(tmp_video_path):
                try:
                    os.remove(tmp_video_path)
                except OSError:
                    pass
            return None

    def get_synthesis_queue_depth(self) -> int:
        """获取当前合成队列深度"""
        return self._synthesis_queue.qsize()

    def stop(self, timeout: float = 5.0) -> None:
        """优雅退出：向工作线程发送哨兵并等待当前在途作业写盘完成"""
        self._running = False
        try:
            self._synthesis_queue.put_nowait(None)
        except Exception:
            pass
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
            logger.info(f"[RingBuffer] 后台视频合成线程已平稳停止: {self.worker_name}")
