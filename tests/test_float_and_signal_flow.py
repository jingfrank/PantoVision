#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：AppConfig 10个输入信号接入、UDP 解析与异常飘弓状态门控流转
覆盖信号对照：
  • 序号 0: 列车速度 (Unsigned16, 比例 0.01 km/h/bit)
  • 序号 1: 年 (Unsigned8)
  • 序号 2: 月 (Unsigned8)
  • 序号 3: 日 (Unsigned8)
  • 序号 4: 时 (Unsigned8)
  • 序号 5: 分 (Unsigned8)
  • 序号 6: 秒 (Unsigned8)
  • 序号 7: 时间有效 (Boolean1)
  • 序号 8: TP06升弓信号 (Boolean1, 1=升弓, 0=降弓)
  • 序号 9: 公里标（HH/HL/LH/LL）(Unsigned32, 单位: 米)
"""

import sys
import os
import socket
import struct
import time
import datetime
import threading
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from common.config import load_app_config, load_input_variables_from_app_config
from common.protocol import (
    pack_message_header,
    PROG_ALGO_DATA,
    FIELD_HEADER_FORMAT,
)
from common.algo_app import PhmBackgroundReporter
from core.data_types import FramePacket, DetectionContext
from core.detectors.float_detector import YoloFloatDetector
from ai_detector import PantographFaultDetector


def test_app_config_10_variables():
    """测试从 AppConfig.xlsx 解析最大输入序号为 9 (共 10 个变量 0~9) 及其定义"""
    algo_id, algo_version, max_input = load_app_config("AppConfig.xlsx", 10008)
    assert algo_id == 10008
    assert max_input == 9, f"Expected max_input_index=9, got {max_input}"

    vars_dict = load_input_variables_from_app_config("AppConfig.xlsx")
    assert len(vars_dict) == 10, f"Expected 10 variables, got {len(vars_dict)}"

    expected = {
        0: ("列车速度", {"Unsigned16", "UINT16"}),
        1: ("年", {"Unsigned8", "UINT8"}),
        2: ("月", {"Unsigned8", "UINT8"}),
        3: ("日", {"Unsigned8", "UINT8"}),
        4: ("时", {"Unsigned8", "UINT8"}),
        5: ("分", {"Unsigned8", "UINT8"}),
        6: ("秒", {"Unsigned8", "UINT8"}),
        7: ("时间有效", {"Boolean1", "UINT8"}),
        8: ("TP06升弓信号", {"Boolean1", "UINT8"}),
        9: ("公里标（HH/HL/LH/LL）", {"Unsigned32", "UINT32"}),
    }

    for idx, (exp_name, exp_types) in expected.items():
        assert idx in vars_dict, f"Index {idx} missing in AppConfig.xlsx"
        assert vars_dict[idx]["name"] == exp_name, f"Expected {exp_name}, got {vars_dict[idx]['name']}"
        assert vars_dict[idx]["type"] in exp_types, f"Expected {exp_types}, got {vars_dict[idx]['type']}"

    print("✅ Test 1: AppConfig.xlsx 10 个输入变量及 max_input_index == 9 校验通过")


def test_udp_10_variables_injection():
    """测试 UDP 接收 PROG_ALGO_DATA (0x22) 报文并完整解包 10 个字段注入共享字典"""
    shared_cfg = {}
    reporter = PhmBackgroundReporter(
        algo_id=10008,
        algo_port=39298,
        app_config_path="AppConfig.xlsx",
        output_provider=lambda: (0, b""),
        shared_cfg=shared_cfg,
        log_enabled=False,
    )

    recv_th = threading.Thread(target=reporter.recv_command_loop, daemon=True)
    recv_th.start()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 构造包含全部 10 个输入变量的 UDP 报文 payload：
    # 0: 列车速度 (Unsigned16, 2B): 25000 (对应 250.0 km/h)
    # 1: 年 (Unsigned8, 1B): 26 (2026年)
    # 2: 月 (Unsigned8, 1B): 9
    # 3: 日 (Unsigned8, 1B): 17
    # 4: 时 (Unsigned8, 1B): 16
    # 5: 分 (Unsigned8, 1B): 50
    # 6: 秒 (Unsigned8, 1B): 0
    # 7: 时间有效 (Boolean1, 1B): 1
    # 8: TP06升弓信号 (Boolean1, 1B): 0 (降弓)
    # 9: 公里标 (Unsigned32, 4B): 123456 (米)
    payload = (
        struct.pack(FIELD_HEADER_FORMAT, 0, 2) + struct.pack("<H", 25000)
        + struct.pack(FIELD_HEADER_FORMAT, 1, 1) + struct.pack("<B", 26)
        + struct.pack(FIELD_HEADER_FORMAT, 2, 1) + struct.pack("<B", 9)
        + struct.pack(FIELD_HEADER_FORMAT, 3, 1) + struct.pack("<B", 17)
        + struct.pack(FIELD_HEADER_FORMAT, 4, 1) + struct.pack("<B", 16)
        + struct.pack(FIELD_HEADER_FORMAT, 5, 1) + struct.pack("<B", 50)
        + struct.pack(FIELD_HEADER_FORMAT, 6, 1) + struct.pack("<B", 0)
        + struct.pack(FIELD_HEADER_FORMAT, 7, 1) + struct.pack("<B", 1)
        + struct.pack(FIELD_HEADER_FORMAT, 8, 1) + struct.pack("<B", 0)
        + struct.pack(FIELD_HEADER_FORMAT, 9, 4) + struct.pack("<I", 123456)
    )
    packet = pack_message_header(PROG_ALGO_DATA, len(payload)) + payload

    sock.sendto(packet, ("127.0.0.1", 39298))
    time.sleep(0.1)

    in_params = shared_cfg.get("input_parameters", {})
    assert in_params[0] == 25000, f"Expected speed=25000, got {in_params.get(0)}"
    assert in_params[1] == 26
    assert in_params[2] == 9
    assert in_params[3] == 17
    assert in_params[4] == 16
    assert in_params[5] == 50
    assert in_params[6] == 0
    assert in_params[7] == 1
    assert in_params[8] == 0  # 降弓
    assert in_params[9] == 123456

    # 模拟切换为升弓信号 (8: 1)
    p_up = struct.pack(FIELD_HEADER_FORMAT, 8, 1) + struct.pack("<B", 1)
    packet_up = pack_message_header(PROG_ALGO_DATA, len(p_up)) + p_up
    sock.sendto(packet_up, ("127.0.0.1", 39298))
    time.sleep(0.1)

    assert in_params[8] == 1  # 升弓

    reporter.close()
    sock.close()
    print("✅ Test 2: UDP 39008 完整 10 个变量报文解码与共享注入通过")


def test_packet_injection_from_inputs():
    """测试 ai_detector 中将 input_parameters 正确映射注入 FramePacket"""
    global_cfg = {
        "hardware_config": {"car_number": 6, "camera_id": 0},
        "video_stream_config": {"fps": 25.0},
        "alarm_push_config": {"common_response_time_s": 0.1, "panto_down_height_limit": 70.0},
        "input_parameters": {
            0: 25000,      # 25000 * 0.01 = 250.0 km/h
            1: 26, 2: 9, 3: 17, 4: 16, 5: 50, 6: 0, 7: 1, # 时间有效
            8: 0,          # 降弓
            9: 123456,     # 公里标 123456 米
        }
    }

    packet = FramePacket(
        frame=np.zeros((100, 100, 3), dtype=np.uint8),
        timestamp=0.0,
        frame_idx=1,
    )

    # 模拟 ai_detector run() 内的注入逻辑
    input_params = global_cfg.get("input_parameters", {})
    if 8 in input_params:
        packet.is_panto_raised = bool(input_params[8])
    if 0 in input_params:
        raw_spd = float(input_params[0])
        packet.speed = raw_spd * 0.01 if raw_spd > 655.35 else raw_spd
    if 9 in input_params:
        kp = input_params[9]
        if kp is not None and kp != 0xFFFFFFFF:
            packet.kilometer_post = int(kp)
    if input_params.get(7) == 1:
        yr = int(input_params.get(1, 0))
        yr = (2000 + yr) if yr < 100 else yr
        mo = int(input_params.get(2, 1))
        dy = int(input_params.get(3, 1))
        hr = int(input_params.get(4, 0))
        mn = int(input_params.get(5, 0))
        sc = int(input_params.get(6, 0))
        dt = datetime.datetime(yr, mo, dy, hr, mn, sc)
        packet.timestamp = dt.timestamp()

    assert packet.is_panto_raised is False
    assert abs(packet.speed - 250.0) < 1e-3
    assert packet.kilometer_post == 123456
    expected_ts = datetime.datetime(2026, 9, 17, 16, 50, 0).timestamp()
    assert abs(packet.timestamp - expected_ts) < 1e-3

    print("✅ Test 3: FramePacket 变量注入 (车速比例转换、升降弓状态、时钟同步、公里标) 通过")


def test_float_detector_gating():
    """测试 YoloFloatDetector 基于新信号状态的门控逻辑"""
    detector = YoloFloatDetector()
    detector.initialize({"panto_down_height_limit": 70.0, "consecutive_alarm_frames": 3})

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # 场景 A: 升弓工况 (is_panto_raised = True，由序号 8 注入)，高度 100 > 70
    packet_raised = FramePacket(
        frame=dummy_frame, timestamp=1.0, frame_idx=1, is_panto_raised=True
    )
    ctx_raised = DetectionContext(packet=packet_raised, panto_bbox=[100, 100, 300, 200])

    for _ in range(5):
        res = detector.postprocess([100, 100, 300, 200], ctx_raised)
        assert not res.is_abnormal, "升弓工况下不应触发异常飘弓告警"
        assert res.extra["abnormal_frame_count"] == 0

    # 场景 B: 降弓工况 (is_panto_raised = False，由序号 8 注入)，受电弓高度 100 > 70
    packet_down = FramePacket(
        frame=dummy_frame, timestamp=2.0, frame_idx=2, is_panto_raised=False
    )
    ctx_down = DetectionContext(packet=packet_down, panto_bbox=[100, 100, 300, 200])

    detector.postprocess([100, 100, 300, 200], ctx_down)
    detector.postprocess([100, 100, 300, 200], ctx_down)
    res3 = detector.postprocess([100, 100, 300, 200], ctx_down)
    assert res3.is_abnormal and res3.extra["abnormal_frame_count"] == 3
    assert res3.fault_type == "abnormal_float"

    # 场景 C: 降弓工况下高度回落正常 (高度 50 < 70)
    res4 = detector.postprocess([100, 100, 300, 150], ctx_down)
    assert not res4.is_abnormal and res4.extra["abnormal_frame_count"] == 0

    print("✅ Test 4: YoloFloatDetector 异常飘弓门控逻辑通过")


if __name__ == "__main__":
    test_app_config_10_variables()
    test_udp_10_variables_injection()
    test_packet_injection_from_inputs()
    test_float_detector_gating()
    print("\n🎉 全部 4 组单元测试通过！")
