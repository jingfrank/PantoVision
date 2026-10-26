"""Check spark filtering at the pantograph and during brief box dropouts."""

import os
import sys
import unittest
from types import SimpleNamespace

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np

from core.data_types import DetectionContext, FramePacket
from core.detectors.spark_detector import YoloSparkDetector


def make_box(cls_id, xyxy, confidence=0.8):
    return SimpleNamespace(
        cls=np.array(cls_id), conf=np.array(confidence), xyxy=np.array([xyxy])
    )


def make_context(timestamp):
    packet = FramePacket(
        frame=np.zeros((100, 200, 3), dtype=np.uint8),
        timestamp=timestamp,
        frame_idx=int(timestamp * 25) + 1,
    )
    return DetectionContext(packet=packet)


class SparkOverlapGateTest(unittest.TestCase):
    def setUp(self):
        # Postprocess tests use synthetic YOLO boxes without loading model weights.
        self.detector = YoloSparkDetector.__new__(YoloSparkDetector)
        self.detector.conf_thresh = 0.30
        self.detector.spark_area_ratio = 0.05
        self.detector.spark_min_area = 400
        self.detector.spark_duration_frames = 2
        self.detector.require_panto_overlap = True
        self.detector.panto_bbox_hold_sec = 1.0
        self.detector.reset()

    def evaluate(self, timestamp, panto=None, sparks=()):
        boxes = []
        if panto is not None:
            boxes.append(make_box(1, panto))
        boxes.extend(make_box(2, spark) for spark in sparks)
        context = make_context(timestamp)
        result = self.detector.postprocess(SimpleNamespace(boxes=boxes), context)
        return result, context

    def test_disjoint_predictions_do_not_accumulate_or_mask_other_detectors(self):
        panto = [40, 30, 160, 60]
        spark = [170, 20, 199, 75]
        for timestamp in (0.0, 0.04, 0.08):
            result, context = self.evaluate(timestamp, panto, [spark])
            self.assertFalse(result.is_abnormal)
            self.assertEqual(result.extra["duration_frames"], 0)
            self.assertEqual(result.extra["raw_spark_count"], 1)
            self.assertEqual(result.extra["spark_count"], 0)
            self.assertEqual(context.spark_bboxes, [])

    def test_intersecting_predictions_confirm_alarm(self):
        panto = [40, 30, 160, 60]
        spark = [80, 15, 110, 45]
        first, _ = self.evaluate(0.0, panto, [spark])
        second, context = self.evaluate(0.04, panto, [spark])
        self.assertFalse(first.is_abnormal)
        self.assertTrue(second.is_abnormal)
        self.assertEqual(second.bbox, spark)
        self.assertEqual(context.spark_bboxes, [spark])

    def test_recent_pantograph_survives_brief_glare_but_expires(self):
        panto = [40, 30, 160, 60]
        spark = [80, 15, 110, 45]
        self.evaluate(0.0, panto)
        recent, _ = self.evaluate(0.5, None, [spark])
        expired, _ = self.evaluate(1.1, None, [spark])
        self.assertEqual(recent.extra["spark_count"], 1)
        self.assertEqual(expired.extra["spark_count"], 0)


if __name__ == "__main__":
    unittest.main()
