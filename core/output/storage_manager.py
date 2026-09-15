#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Output Layer: Storage Manager.
Manages daily directories and file naming standards for snapshots and incident recordings.
"""

import os
import cv2
import time
import datetime
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class StorageManager:
    """落盘存储管理引擎，遵循车载规范化落盘路径与命名规则"""

    def __init__(
        self,
        base_result_dir: str = "./Algo-result",
        pre_sec: int = 60,
        post_sec: int = 60,
        train_id: str = "CR400BF"
    ):
        self.base_dir = os.path.abspath(base_result_dir)
        self.pre_sec = pre_sec
        self.post_sec = post_sec
        self.train_id = train_id

    def get_daily_directory(self, timestamp: float) -> str:
        """根据故障时间戳返回 base_dir/月-日 文件夹"""
        dt = datetime.datetime.fromtimestamp(timestamp)
        daily_dir = os.path.join(self.base_dir, dt.strftime("%m-%d"))
        os.makedirs(daily_dir, exist_ok=True)
        return daily_dir

    def generate_base_name(self, timestamp: float, fault_code: int) -> Tuple[str, str]:
        """
        按车载规范生成文件基名与独立事件目录:
        基名格式: `起始时间-终止时间-车次-PCM_Algo_result-故障代码`
        事件目录: `base_dir/月-日/基名/`
        返回: (base_name, event_dir)
        """
        daily_dir = self.get_daily_directory(timestamp)
        start_ts = timestamp - self.pre_sec
        end_ts = timestamp + self.post_sec
        start_str = time.strftime("%H%M%S", time.localtime(start_ts))
        end_str = time.strftime("%H%M%S", time.localtime(end_ts))
        
        # 严格遵守车载规范: CR400BF-PCM_Algo_result-{fault_code} (下划线连接，禁止空格)
        base_name = f"{start_str}-{end_str}-{self.train_id}-PCM_Algo_result-{fault_code}"
        event_dir = os.path.join(daily_dir, base_name)
        os.makedirs(event_dir, exist_ok=True)
        return base_name, event_dir

    def save_snapshot(
        self,
        frame,
        timestamp: float,
        fault_code: int,
        base_name: Optional[str] = None,
        save_dir: Optional[str] = None
    ) -> str:
        """保存故障时刻高质量图片至事件独立子目录"""
        if base_name is None or save_dir is None:
            base_name, save_dir = self.generate_base_name(timestamp, fault_code)
        
        pic_path = os.path.join(save_dir, f"{base_name}.jpg")
        cv2.imwrite(pic_path, frame)
        logger.info(f"[StorageManager] 故障抓拍图片已规范化落盘: {pic_path}")
        return pic_path

    def clean_local_storage(self, max_storage_gb: float = 30.0, retention_days: int = 7) -> None:
        """本地磁盘保护：当本地缓存超出容量或超出保存天数时淘汰最旧事件"""
        try:
            if not os.path.exists(self.base_dir):
                return
            now = time.time()
            cutoff_ts = now - retention_days * 86400

            # 遍历所有事件目录
            event_dirs = []
            total_bytes = 0
            for root, dirs, files in os.walk(self.base_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        sz = os.path.getsize(fp)
                        total_bytes += sz
                    except OSError:
                        pass
                for d in dirs:
                    dp = os.path.join(root, d)
                    if "PCM_Algo_result" in d:
                        try:
                            mtime = os.path.getmtime(dp)
                            event_dirs.append((mtime, dp))
                        except OSError:
                            pass

            max_bytes = int(max_storage_gb * (1024 ** 3))
            # 1. 淘汰超期文件
            for mtime, ed in sorted(event_dirs, key=lambda x: x[0]):
                if mtime < cutoff_ts or total_bytes > max_bytes:
                    import shutil
                    try:
                        shutil.rmtree(ed, ignore_errors=True)
                        logger.info(f"[StorageManager] 本地磁盘容量维护: 已清理过期/超限事件目录: {ed}")
                    except Exception as e:
                        logger.error(f"[StorageManager] 清理事件目录失败 {ed}: {e}")
        except Exception as e:
            logger.error(f"[StorageManager] 本地磁盘清理异常: {e}")

