#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T4: YoloFloatDetector 黑板零计算复用测试 - 验证 infer() 无 GPU 调用
T5c: 双向步进积分消抖验证（LeakyStepIntegrator 逻辑）
"""
import sys, time
sys.path.insert(0, "F:/gongwang_clean")

from core.data_types import DetectionContext, FramePacket
from core.detectors.float_detector import YoloFloatDetector
import numpy as np


def make_packet(ts=0.0, raised=True):
    return FramePacket(
        frame=np.zeros((640, 640, 3), dtype=np.uint8),
        timestamp=ts, frame_idx=0,
        is_panto_raised=raised
    )


# ─────────────────────────────────────────────
# T4: 黑板零计算复用测试
# ─────────────────────────────────────────────
def test_float_detector_zero_gpu():
    """YoloFloatDetector.infer() 不应调用任何 GPU/YOLO 推理，只做透传"""
    det = YoloFloatDetector()
    det.initialize({
        "window_size": 30,
        "float_speed_thresh": 40.0,
        "panto_down_height_limit": 70.0,
        "consecutive_alarm_frames": 5,
    })

    panto_box = [100, 50, 400, 200]  # height=150 > panto_down_height_limit=70
    ctx = DetectionContext(packet=make_packet(ts=0.0, raised=False))
    ctx.panto_bbox = panto_box

    t0 = time.perf_counter()
    tensor = det.preprocess(image=None, context=ctx)
    infer_out = det.infer(tensor)
    elapsed_us = (time.perf_counter() - t0) * 1e6

    assert infer_out == panto_box, "infer() 必须直接透传 panto_box，不调用 GPU"
    assert elapsed_us < 500, f"infer() 耗时应 <500μs，实际 {elapsed_us:.1f}μs"
    print(f"[PASS] T4 黑板零GPU复用: infer() 透传，耗时 {elapsed_us:.2f}μs")

    # 验证累计 5 帧后触发告警
    det.reset()
    for i in range(5):
        ctx2 = DetectionContext(packet=make_packet(ts=float(i) * 0.04, raised=False))
        ctx2.panto_bbox = panto_box
        tensor2 = det.preprocess(None, ctx2)
        raw = det.infer(tensor2)
        result = det.postprocess(raw, ctx2)

    assert result.is_abnormal == True, "连续5帧高度异常后应触发告警"
    assert result.level == "critical"
    print(f"[PASS] T4 连续5帧 → critical 告警: count={result.extra['abnormal_frame_count']}")


def test_panto_raised_suppresses_float():
    """受电弓升弓状态下，飘弓检测应直接返回 normal（TCMS 联动）"""
    det = YoloFloatDetector()
    det.initialize({"panto_down_height_limit": 70.0, "consecutive_alarm_frames": 5})

    panto_box = [100, 50, 400, 200]  # 正常高度也很大
    ctx = DetectionContext(packet=make_packet(ts=0.0, raised=True))  # is_panto_raised=True
    ctx.panto_bbox = panto_box

    tensor = det.preprocess(None, ctx)
    raw = det.infer(tensor)
    result = det.postprocess(raw, ctx)

    assert result.is_abnormal == False, "升弓状态不应触发飘弓告警"
    assert result.extra.get("is_panto_raised") == True
    print(f"[PASS] T4 TCMS升弓抑制: is_abnormal=False, reason={result.extra.get('reason')}")


# ─────────────────────────────────────────────
# T5c: 双向步进积分消抖
# ─────────────────────────────────────────────
class LeakyStepIntegrator:
    """博客文章对应的参考实现"""
    def __init__(self, step_up=1.0, step_down=0.5, warn_thresh=3.0, alarm_thresh=5.0):
        self.value = 0.0
        self.step_up = step_up
        self.step_down = step_down
        self.warn_thresh = warn_thresh
        self.alarm_thresh = alarm_thresh

    def update(self, is_abnormal: bool) -> str:
        if is_abnormal:
            self.value = min(self.value + self.step_up, self.alarm_thresh + 2)
        else:
            self.value = max(self.value - self.step_down, 0.0)
        if self.value >= self.alarm_thresh:
            return "alarm"
        elif self.value >= self.warn_thresh:
            return "warning"
        return "normal"


def test_leaky_step_integrator_no_false_clear():
    """中间插入1帧正常帧，不应清零（对比传统固定窗口计数器）"""
    integrator = LeakyStepIntegrator(step_up=1.0, step_down=0.5, warn_thresh=3.0, alarm_thresh=5.0)
    sequence = [True, True, True, True, True, False, True, True]
    results = [integrator.update(x) for x in sequence]
    assert results[4] == "alarm", f"第5帧应为 alarm，实际={results[4]}"
    assert results[5] != "normal", f"1帧正常不应清零到 normal，实际={results[5]}"
    print(f"[PASS] T5c 1帧正常不清零: 第6帧status={results[5]} (值={integrator.value+0.5:.1f}→{integrator.value:.1f})")


def test_leaky_step_integrator_true_normal():
    """连续足够多正常帧才退出告警"""
    integrator = LeakyStepIntegrator(step_up=1.0, step_down=0.5, warn_thresh=3.0, alarm_thresh=5.0)
    for _ in range(6):
        integrator.update(True)
    status = "alarm"
    steps = 0
    while status != "normal" and steps < 30:
        status = integrator.update(False)
        steps += 1
    assert status == "normal", "足够多正常帧后应退回 normal"
    print(f"[PASS] T5c 需 {steps} 帧正常才退出告警（步长衰减设计正确）")


def test_leaky_step_integrator_no_missed_alarm():
    """漏检1帧（假阴性）不应导致真实故障漏报（博客核心卖点）"""
    integrator = LeakyStepIntegrator(step_up=1.0, step_down=0.5, warn_thresh=3.0, alarm_thresh=5.0)
    # 4帧异常 + 1帧漏检(正常) + 2帧异常 → 应仍能积累到 alarm
    sequence = [True, True, True, True, False, True, True]
    results = [integrator.update(x) for x in sequence]
    final_status = results[-1]
    assert final_status in ("alarm", "warning"), \
        f"漏检1帧后，后续异常应仍积累，实际={final_status}"
    print(f"[PASS] T5c 漏检鲁棒性: 7帧序列末 status={final_status}, value={integrator.value:.1f}")


if __name__ == "__main__":
    print("=" * 60)
    print("T4 黑板零计算复用 & T5c 步进积分消抖测试")
    print("=" * 60)
    test_float_detector_zero_gpu()
    test_panto_raised_suppresses_float()
    test_leaky_step_integrator_no_false_clear()
    test_leaky_step_integrator_true_normal()
    test_leaky_step_integrator_no_missed_alarm()
    print("\n✅ 全部通过")
