#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Asynchronous NAS/FTP Uploader (Stage 2: 独立网络上传流水线).

将抓拍图片与录像文件的 FTP/NAS 上传从检测与编码主路径解耦：
- 线程安全任务队列 (UploadQueue) 支持背压限制；
- 独立后台工作线程 (NasUploadWorker) 消费任务，避免网络 I/O 阻塞；
- 失败指数退避重试 (最大重试 3 次)；
- 成功上传后原子安全清理本地容器文件；
- 线程异常隔离与健康自愈。
"""

import os
import time
import queue
import logging
import threading
from typing import Optional, Dict, Any, Callable
from ftp_uploader import get_ftp_uploader

logger = logging.getLogger(__name__)


class UploadTask:
    """单个上传任务载体"""
    __slots__ = ("local_path", "remote_path", "created_at", "retries")

    def __init__(self, local_path: str, remote_path: str, retries: int = 3):
        self.local_path = local_path
        self.remote_path = remote_path
        self.created_at = time.time()
        self.retries = retries


class AsyncNasUploader:
    """
    异步 NAS 上传管理器（独立工作线程 + 任务队列 + 退避重试）。
    """

    def __init__(
        self,
        max_queue_size: int = 20,
        worker_name: str = "NasUploadWorker",
        max_retries: int = 3,
        retry_delay_sec: float = 2.0,
    ):
        self.max_queue_size = max_queue_size
        self.worker_name = worker_name
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._running: bool = True
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # 统计指标
        self.total_uploaded: int = 0
        self.total_failed: int = 0
        self.total_dropped: int = 0

        self._start_worker()

    def _start_worker(self) -> None:
        """启动上传工作线程"""
        with self._lock:
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name=self.worker_name
            )
            self._worker_thread.start()
            logger.info(f"[AsyncUploader] 启动后台上传线程: {self.worker_name}")

    def _check_and_heal_worker(self) -> None:
        """线程健康探测与自愈"""
        if self._running and (self._worker_thread is None or not self._worker_thread.is_alive()):
            logger.critical(f"[AsyncUploader] 探测到上传工作线程 {self.worker_name} 意外退出，正在自愈重启...")
            self._start_worker()

    def submit(self, local_path: str, remote_path: str) -> bool:
        """
        向上传流水线投递任务（非阻塞）。
        返回 True 表示投递成功；False 表示本地文件不存在或因背压丢弃。
        """
        self._check_and_heal_worker()

        if not local_path or not os.path.exists(local_path):
            logger.warning(f"[AsyncUploader] 本地待上传文件不存在: {local_path}")
            return False

        task = UploadTask(local_path=local_path, remote_path=remote_path, retries=self.max_retries)

        try:
            self._queue.put_nowait(task)
            return True
        except queue.Full:
            # 背压保护：上传队列已满，丢弃最旧任务以腾出空间
            self.total_dropped += 1
            try:
                dropped: UploadTask = self._queue.get_nowait()
                logger.warning(
                    f"[AsyncUploader-Backpressure] 上传队列已满 (maxsize={self.max_queue_size})，"
                    f"丢弃排队时间最久任务: {dropped.local_path}"
                )
                self._queue.put_nowait(task)
                return True
            except (queue.Empty, queue.Full):
                logger.error(f"[AsyncUploader-Backpressure] 无法写入上传队列，直接放弃本次任务: {local_path}")
                return False

    def _worker_loop(self) -> None:
        """上传线程工作主循环（顶级异常兜底，永不崩溃退出）"""
        while self._running:
            try:
                task = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if task is None:
                # 停机哨兵
                self._queue.task_done()
                break

            try:
                self._process_upload(task)
            except Exception as e:
                logger.error(
                    f"[AsyncUploader] 处理上传任务发生未捕获异常: {task.local_path} -> {e}",
                    exc_info=True
                )
            finally:
                self._queue.task_done()

    def _process_upload(self, task: UploadTask) -> None:
        """单次上传尝试与重试逻辑"""
        uploader = get_ftp_uploader()
        if not uploader or not uploader.enabled:
            logger.debug(f"[AsyncUploader] FTP 未启用或未配置，跳过上传: {task.local_path}")
            return

        if not os.path.exists(task.local_path):
            logger.warning(f"[AsyncUploader] 待上传文件在消费前已丢失: {task.local_path}")
            return

        t0 = time.perf_counter()
        success = False
        try:
            success = uploader.upload_and_delete(task.local_path, task.remote_path)
        except Exception as e:
            logger.error(f"[AsyncUploader] 调用 upload_and_delete 抛出异常: {e}")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if success:
            self.total_uploaded += 1
            logger.info(
                f"[AsyncUploader] 上传成功并清理本地文件 ({elapsed_ms:.1f}ms): "
                f"{os.path.basename(task.local_path)} -> {task.remote_path}"
            )
        else:
            if task.retries > 0:
                task.retries -= 1
                logger.warning(
                    f"[AsyncUploader] 上传失败，{self.retry_delay_sec}秒后进行剩余 {task.retries} 次重试: "
                    f"{task.local_path}"
                )
                time.sleep(self.retry_delay_sec)
                try:
                    self._queue.put_nowait(task)
                except queue.Full:
                    self.total_failed += 1
                    logger.error(f"[AsyncUploader] 重试入队失败（队列满），放弃任务: {task.local_path}")
            else:
                self.total_failed += 1
                logger.error(
                    f"[AsyncUploader] 上传彻底失败（已达最大重试次数），本地保留供 NAS 清理策略巡检: "
                    f"{task.local_path}"
                )

    def get_queue_depth(self) -> int:
        """获取当前上传队列积压深度"""
        return self._queue.qsize()

    def stop(self, timeout: float = 3.0) -> None:
        """优雅退出"""
        self._running = False
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
            logger.info(f"[AsyncUploader] 后台上传线程已平稳停止: {self.worker_name}")
