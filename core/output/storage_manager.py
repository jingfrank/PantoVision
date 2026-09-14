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
from typing import Tuple

logger = logging.getLogger(__name__)


class StorageManager:
    """落盘存储管理引擎"""

    def __init__(self, base_result_dir: str = "./Algo-result", pre_sec: int = 60, post_sec: int = 60):
        self.base_dir = base_result_dir
        self.pre_sec = pre_sec
        self.post_sec = post_sec

    def get_daily_directory(self, timestamp: float) -> str:
        """根据故障时间戳返回 base_dir/月-日 文件夹"""
        dt = datetime.datetime.fromtimestamp(timestamp)
        daily_dir = os.path.join(self.base_dir, dt.strftime("%m-%d"))
        os.makedirs(daily_dir, exist_ok=True)
        return daily_dir

    def generate_base_name(self, timestamp: float, fault_code: int) -> Tuple[str, str]:
        """
        按规范生成文件基名:
        `起始时间-终止时间-CR400BF-PCM Algo result-故障代码`
        """
        save_dir = self.get_daily_directory(timestamp)
        start_ts = timestamp - self.pre_sec
        end_ts = timestamp + self.post_sec
        start_str = time.strftime("%H%M%S", time.localtime(start_ts))
        end_str = time.strftime("%H%M%S", time.localtime(end_ts))
        base_name = f"{start_str}-{end_str}-CR400BF-PCM Algo result-{fault_code}"
        return base_name, save_dir

    def save_snapshot(self, frame, timestamp: float, fault_code: int) -> str:
        """保存故障时刻高质量图片"""
        base_name, save_dir = self.generate_base_name(timestamp, fault_code)
        pic_path = os.path.join(save_dir, f"{base_name}.jpg")
        cv2.imwrite(pic_path, frame)
        logger.info(f"[StorageManager] 故障抓拍图片已落盘: {pic_path}")
        return pic_path
