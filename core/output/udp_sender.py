#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Output Layer: 0xFF 0x20 Standard Binary UDP Alarm Sender.
Conforms to CR400BF train-level pantograph fault alarm packet specifications.
"""

import socket
import struct
import os
import sys
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

FAULT_TYPE_MAP = {
    "大火花": 3,
    "big_spark": 3,
    "受电弓结构异常": 4,
    "structure_abnormal": 4,
    "画面脏污": 10,
    "dirty": 10,
    "受电弓倾斜": 12,
    "tilt": 12,
    "高压故障": 13,
    "悬挂异物": 14,
    "foreign_object": 14,
    "异常飘弓": 15,
    "abnormal_float": 15,
}


class UDPAlarmSender:
    """车载标准二进制 0xFF 0x20 报文发射器"""

    def __init__(self, udp_ip: str = "127.0.0.1", udp_port: int = 8080):
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    def send_alarm(
        self,
        event_name: str,
        car_number: int,
        camera_id: int,
        picture_path: str,
        level: int = 1,
        fault_code: int = 0
    ) -> bool:
        """
        组装并广播 0xFF 0x20 协议报文
        Data数据域: 1B故障级别 + 2B故障类型(1B) + 3B~4B故障代码(2B) + 5B~10B车体号(6B) + 11B~NB路径名
        """
        try:
            type_code = FAULT_TYPE_MAP.get(event_name, 10)
            fault_level = 0x01 if level == 1 else 0x03

            fault_type_byte = struct.pack("B", type_code)
            fault_code_bytes = struct.pack(">H", fault_code)
            dev_info_bytes = struct.pack(">HI", car_number, camera_id)
            path_bytes = os.path.abspath(picture_path).encode("utf-8")

            data_field = (
                struct.pack("B", fault_level)
                + fault_type_byte
                + fault_code_bytes
                + dev_info_bytes
                + path_bytes
            )
            data_len = len(data_field)

            packet_header = struct.pack(">B B H", 0xFF, 0x20, data_len)
            packet_base = packet_header + data_field
            checksum = sum(packet_base) & 0xFF
            final_packet = packet_base + struct.pack("B", checksum)

            self.sock.sendto(final_packet, (self.udp_ip, self.udp_port))
            logger.info(
                f"[UDP-Alarm] 成功发射 0xFF 0x20 报文 -> 事件:【{event_name}】 CAR{car_number}-CAM{camera_id} "
                f"| 字节数: {len(final_packet)} | 目标: {self.udp_ip}:{self.udp_port}"
            )
            return True

        except Exception as e:
            logger.error(f"[UDP-Alarm] 发射报文失败: {e}", exc_info=True)
            return False

    def close(self) -> None:
        if self.sock:
            self.sock.close()
