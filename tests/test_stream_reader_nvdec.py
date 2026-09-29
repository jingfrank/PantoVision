#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元与集成回归测试: RTSP GStreamer NVDEC 硬件解码流水线与自适应探测.
"""

import unittest
import time
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ingestion.stream_reader import (
    build_jetson_gst_pipeline,
    detect_rtsp_codec,
    RTSPStreamReader,
    _open_capture
)
from core.perf_monitor import PerformanceMonitor
from core.data_types import FramePacket, DetectionResult


class TestRTSPStreamReaderNVDEC(unittest.TestCase):
    def test_pipeline_builder_h264(self):
        url = "rtsp://172.10.1.126/live/test"
        pipe = build_jetson_gst_pipeline(url, codec="h264", latency_ms=0)
        self.assertIn("rtph264depay", pipe)
        self.assertIn("h264parse", pipe)
        self.assertIn("nvv4l2decoder", pipe)
        self.assertIn("nvvidconv", pipe)
        self.assertIn("format=BGRx", pipe)
        self.assertIn("appsink", pipe)
        self.assertIn("latency=0", pipe)
        self.assertIn("protocols=tcp", pipe)

    def test_pipeline_builder_h265(self):
        url = "rtsp://172.10.1.126/live/test"
        pipe = build_jetson_gst_pipeline(url, codec="h265", latency_ms=50)
        self.assertIn("rtph265depay", pipe)
        self.assertIn("h265parse", pipe)
        self.assertIn("nvv4l2decoder", pipe)
        self.assertIn("latency=50", pipe)

    def test_codec_probe_fallback(self):
        # 针对不可达地址，应快速超时并安全回退为默认 h264
        codec = detect_rtsp_codec("rtsp://127.0.0.1:59999/dummy", fallback="h264")
        self.assertEqual(codec, "h264")

    def test_rtsp_stream_reader_init(self):
        reader = RTSPStreamReader(
            rtsp_url="rtsp://172.10.1.126/live/test",
            car_number=6,
            camera_id=0,
            decoder_backend="gstreamer_nvdec",
            codec_type="auto",
            gstreamer_latency_ms=10
        )
        self.assertEqual(reader.decoder_backend, "gstreamer_nvdec")
        self.assertEqual(reader.codec_type, "auto")
        self.assertEqual(reader.gstreamer_latency_ms, 10)

    def test_perf_monitor_ingest_ms(self):
        pm = PerformanceMonitor(source_fps=30.0, report_interval_sec=0.1)
        packet = FramePacket(
            frame=MagicMock(),
            timestamp=time.time(),
            frame_idx=0,
            car_number=6,
            camera_id=0,
            train_id="CR400BF"
        )
        results = {
            "spark_detector": DetectionResult(
                fault_type="spark",
                is_abnormal=False,
                score=0.0,
                extra={"latency_ms": 12.5}
            )
        }
        # 测试包含 ingest_ms 字段的 tick
        pm.tick(packet, results, wall_ms=25.0, queue_depth=1, ingest_ms=3.5)
        self.assertEqual(len(pm._ingest_ms_samples), 1)
        self.assertEqual(pm._ingest_ms_samples[0], 3.5)


if __name__ == "__main__":
    unittest.main()
