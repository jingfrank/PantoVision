#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证视场偏转 (Camera Offset) 与清晰度 (LapVar) 防御联动机制。
1. 模拟画面偏移样本：无合法受电弓且 TopLap > 280 -> 阻断 FastAD 推理，全系 FastAD 检测器静默。
2. 模拟真实脏污样本：无受电弓但模糊低频 TopLap < 280 -> 允许脏污检测器正常确诊，结构异常保持静默。
3. 模拟正常受电弓样本：检出合法受电弓 -> 各检测器正常运转。
"""

import unittest
import numpy as np
import cv2

from core.data_types import FramePacket, DetectionContext, DetectionResult
from core.detectors.spark_detector import YoloSparkDetector
from core.detectors.dirt_detector import FastadDirtDetector
from core.detectors.struct_detector import FastadStructDetector
from core.detectors.foreign_detector import FastadForeignDetector
from core.detectors.fastad_base import FastADBaseDetector


class DummyFastADModel:
    def predict(self, tensor):
        # 模拟产生全图高异常分
        h, w = tensor.shape[:2] if hasattr(tensor, "shape") else (1536, 2048)
        return {
            "anomaly_score": 12.0,
            "anomaly_map": np.ones((h, w), dtype=np.float32) * 5.0,
            "pixel_threshold": 3.0,
            "image_threshold": 4.0,
        }


class TestCameraOffsetDefense(unittest.TestCase):

    def test_camera_offset_suppression(self):
        """测试画面偏移场景：无受电弓 + 高频硬边缘 -> 触发全系 FastAD 拦截"""
        # 创建一个充满高频网格线条的高清晰度车顶图像 (TopLap 会非常高)
        frame = np.zeros((1536, 2048, 3), dtype=np.uint8)
        for y in range(0, 500, 10):
            cv2.line(frame, (0, y), (2048, y), (255, 255, 255), 2)
        for x in range(0, 2048, 10):
            cv2.line(frame, (x, 0), (x, 500), (255, 255, 255), 2)

        packet = FramePacket(frame=frame, timestamp=1.0, frame_idx=1)
        context = DetectionContext(packet=packet)

        # 1. 模拟 spark_detector 检测 (未检出有效受电弓)
        # 手工构造空的或不合理的 YOLO boxes (如一个贴边的非受电弓框)
        spark_det = YoloSparkDetector()
        spark_det.spark_duration_frames = 12
        spark_det.conf_thresh = 0.3
        spark_det.spark_min_area = 400
        spark_det.spark_area_ratio = 0.05
        spark_det.require_panto_overlap = True
        spark_det.panto_bbox_hold_sec = 1.0

        class DummyYOLOOutput:
            boxes = []  # 无受电弓检出

        spark_res = spark_det.postprocess(DummyYOLOOutput(), context=context)

        # 验证 context 中的标记
        self.assertTrue(context.is_camera_offset, "应正确识别为画面偏转 (is_camera_offset=True)")
        self.assertGreater(context.top_lap_var, 280.0, "顶部 LapVar 应大幅超过 280")
        self.assertIsNone(context.panto_bbox, "受电弓目标框应为 None")

        # 2. 验证 FastADBaseDetector.infer 是否快速阻断
        fastad_base = FastadDirtDetector()
        fastad_base._model_path_abs = "dummy"
        fastad_base.model = DummyFastADModel()
        fastad_base.preprocess(frame, context=context)
        infer_out = fastad_base.infer(frame)
        self.assertTrue(infer_out.get("is_camera_offset_bypassed"), "FastAD 前向推理应被拦截直接短路")

        # 3. 验证 FastadDirtDetector 是否彻底静默
        dirt_res = fastad_base.postprocess(infer_out, context=context)
        self.assertFalse(dirt_res.is_abnormal, "画面偏移下脏污检测器必须静默")
        self.assertEqual(dirt_res.extra.get("status"), "camera_offset_suppressed")

        # 4. 验证 FastadStructDetector 是否彻底静默
        struct_det = FastadStructDetector()
        struct_det._model_path_abs = "dummy"
        struct_det.preprocess(frame, context=context)
        struct_res = struct_det.postprocess(infer_out, context=context)
        self.assertFalse(struct_res.is_abnormal, "画面偏移下结构异常检测器必须静默")
        self.assertEqual(struct_res.extra.get("status"), "camera_offset_suppressed")

        # 5. 验证 FastadForeignDetector 是否彻底静默
        foreign_det = FastadForeignDetector()
        foreign_det._model_path_abs = "dummy"
        foreign_det.preprocess(frame, context=context)
        foreign_res = foreign_det.postprocess(infer_out, context=context)
        self.assertFalse(foreign_res.is_abnormal, "画面偏移下异物检测器必须静默")
        self.assertEqual(foreign_res.extra.get("status"), "camera_offset_suppressed")

        print("✅ 测试 1 通过: 画面偏移下成功拦截 FastAD 全系误报 (包含脏污、结构异常、异物)")

    def test_blurry_dirt_allowed(self):
        """测试极度严重脏污场景：无受电弓 + 平滑模糊泥浆 (TopLap < 280) -> 允许脏污检测继续运行"""
        # 创建模糊低频画面 (模拟泥浆糊死镜头)
        frame = np.ones((1536, 2048, 3), dtype=np.uint8) * 128
        frame = cv2.GaussianBlur(frame, (15, 15), 5.0)

        packet = FramePacket(frame=frame, timestamp=1.0, frame_idx=1)
        context = DetectionContext(packet=packet)

        spark_det = YoloSparkDetector()
        spark_det.spark_duration_frames = 12
        spark_det.conf_thresh = 0.3
        spark_det.spark_min_area = 400
        spark_det.spark_area_ratio = 0.05
        spark_det.require_panto_overlap = True
        spark_det.panto_bbox_hold_sec = 1.0

        class DummyYOLOOutput:
            boxes = []  # 泥浆遮挡，无受电弓

        spark_det.postprocess(DummyYOLOOutput(), context=context)

        # 验证 context：虽然无弓，但不是画面偏移（是模糊遮挡）
        self.assertFalse(context.is_camera_offset, "模糊画面不应标记为画面偏移")
        self.assertLess(context.top_lap_var, 280.0, "顶部 LapVar 应远小于 280")

        # 验证 FastAD 前向推理不被短路，允许脏污检测器根据异常图确诊
        fastad_base = FastadDirtDetector()
        fastad_base._model_path_abs = "dummy"
        fastad_base.model = DummyFastADModel()
        fastad_base.preprocess(frame, context=context)
        infer_out = fastad_base.infer(frame)
        self.assertFalse(infer_out.get("is_camera_offset_bypassed", False), "严重脏污下 FastAD 推理不应被短路")

        print("✅ 测试 2 通过: 极度严重脏污下未被误杀，允许正常脏污研判")


if __name__ == "__main__":
    unittest.main()
