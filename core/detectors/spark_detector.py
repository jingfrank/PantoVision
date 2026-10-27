#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO-based Big Spark Detector Plugin.
Implements BaseDetector for high-speed (30 FPS) pantograph arc discharge detection.
"""

import os
from typing import Any, Optional, Dict, List
import numpy as np
import logging
from ultralytics import YOLO

from core.data_types import DetectionResult, DetectionContext
from core.detectors.base import BaseDetector
from core.detectors.registry import DetectorRegistry

logger = logging.getLogger(__name__)


@DetectorRegistry.register("YoloSparkDetector")
class YoloSparkDetector(BaseDetector):
    name: str = "spark_yolo"
    fault_type: str = "big_spark"
    input_size: tuple = (640, 640)

    def initialize(self, config: Dict[str, Any]) -> None:
        model_path = config.get("model_path", "models/best.pt")
        engine_path = config.get("engine_path", None)
        prefer_trt = config.get("prefer_tensorrt", True)
        self.conf_thresh = config.get("conf_thresh", 0.30)
        self.panto_min_conf = float(config.get("panto_min_conf", 0.05))
        self.panto_incomplete_hold_sec = float(config.get("panto_incomplete_hold_sec", 10.0))
        self.spark_area_ratio = config.get("spark_area_ratio", 0.05)
        self.spark_min_area = config.get("spark_min_area", 400)
        self.spark_duration_frames = config.get("spark_duration_frames", 12)  # ~500-700ms at 25fps
        self.require_panto_overlap = config.get("require_panto_overlap", True)
        self.panto_bbox_hold_sec = max(0.0, float(config.get("panto_bbox_hold_sec", 1.0)))

        # 优先探测并加载 TensorRT 引擎
        self.is_trt = False
        if prefer_trt:
            candidate_engine = engine_path
            if not candidate_engine and model_path:
                candidate = os.path.splitext(model_path)[0] + ".engine"
                if os.path.exists(candidate):
                    candidate_engine = candidate

            if candidate_engine and os.path.exists(candidate_engine):
                try:
                    logger.info(f"[{self.name}] 命中 TensorRT 加速引擎: {candidate_engine}")
                    self.model = YOLO(candidate_engine, task="detect")
                    self.is_trt = True
                    logger.info(f"[{self.name}] YOLO TensorRT 引擎加载成功, 类别: {self.model.names}")
                except Exception as e:
                    logger.warning(
                        f"[{self.name}] 加载 TensorRT 引擎失败 ({e})，将平滑回退至 PyTorch 权重: {model_path}"
                    )
                    self.is_trt = False

        if not self.is_trt:
            logger.info(f"[{self.name}] 正在加载 YOLO PyTorch 权重: {model_path} ...")
            self.model = YOLO(model_path)
            logger.info(f"[{self.name}] YOLO PyTorch 模型加载成功, 类别: {self.model.names}")

        self.current_duration_count = 0
        self.alarmed = False
        self.last_panto_bbox = None
        self.last_panto_timestamp = None
        self.last_incomplete_timestamp = None

    @staticmethod
    def _boxes_intersect(a: List[int], b: List[int]) -> bool:
        """Require positive-area overlap, rather than merely touching box edges."""
        return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])

    def preprocess(self, image: Any, context: Optional[DetectionContext] = None) -> Any:
        return image

    def infer(self, tensor: Any) -> Any:
        # Ultralytics 直接支持 numpy 图像输入，推理置信度使用 min(conf_thresh, panto_min_conf) 保证边缘残损弓可检出
        infer_conf = min(self.conf_thresh, self.panto_min_conf)
        return self.model(tensor, conf=infer_conf, verbose=False)

    def postprocess(self, raw_output: Any, context: Optional[DetectionContext] = None) -> DetectionResult:
        res = raw_output[0] if isinstance(raw_output, list) else raw_output
        panto_boxes = []
        spark_boxes = []
        max_spark_conf = 0.0

        for box in res.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = [int(v) for v in box.xyxy[0].tolist()]

            if cls_id == 1:  # pantograph
                panto_boxes.append((xyxy, conf))
            elif cls_id == 2:  # spark
                if conf >= self.conf_thresh:
                    spark_boxes.append((xyxy, conf))
                    if conf > max_spark_conf:
                        max_spark_conf = conf

        # 1. 提取并进行受电弓目标几何合法性校验 (Geometric Sanity Check)
        best_panto_box = None
        best_panto_conf = 0.0
        frame_ref = context.packet.frame if (context and context.packet is not None) else None
        img_h, img_w = frame_ref.shape[:2] if frame_ref is not None else (1536, 2048)

        high_panto_boxes = [item for item in panto_boxes if item[1] >= self.conf_thresh]
        low_panto_boxes = [item for item in panto_boxes if item[1] < self.conf_thresh]

        if high_panto_boxes:
            high_panto_boxes.sort(key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]), reverse=True)
            for pbox, pconf in high_panto_boxes:
                bw = pbox[2] - pbox[0]
                bh = pbox[3] - pbox[1]
                aspect_ratio = bw / max(bh, 1)

                # ① 宽高比检查：受电弓横梁呈扁平水平分布，宽高比通常 >= 1.5
                if aspect_ratio < 1.5:
                    logger.debug("[spark_detector] 剔除宽高比异常虚警框: W/H=%.2f < 1.5, box=%s", aspect_ratio, pbox)
                    continue

                # ② 极端边缘截断检查：同时贴合两条及以上物理边缘的目标多为车体侧壁/导流板
                edge_margin = 15
                edge_count = sum([
                    pbox[0] <= edge_margin,         # 紧贴左边界
                    pbox[2] >= img_w - edge_margin, # 紧贴右边界
                    pbox[3] >= img_h - edge_margin, # 紧贴底部边界
                ])
                if edge_count >= 2:
                    logger.debug("[spark_detector] 剔除贴边截断虚警框: edge_count=%d, box=%s", edge_count, pbox)
                    continue

                # ③ 面积下限检查：受电弓本体至少占画面一定面积
                area_fraction = (bw * bh) / float(img_w * img_h)
                if area_fraction < 0.005:
                    logger.debug("[spark_detector] 剔除过小虚警框: area_frac=%.4f < 0.005, box=%s", area_fraction, pbox)
                    continue

                best_panto_box = pbox
                best_panto_conf = pconf
                break

        # 受电弓显示完整度检查 (监控不完整/边缘截断识别与时序锁存)
        timestamp = context.packet.timestamp if context and context.packet else None
        is_panto_incomplete = False
        margin_x = max(30, int(img_w * 0.02))
        margin_y = max(25, int(img_h * 0.025))

        if best_panto_box is not None:
            px1, py1, px2, py2 = best_panto_box
            touches_edge = (
                px1 <= margin_x or
                px2 >= img_w - margin_x or
                py1 <= margin_y or
                py2 >= img_h - margin_y
            )
            pw = px2 - px1
            too_narrow = (pw < img_w * 0.45) and (px1 <= margin_x * 2 or px2 >= img_w - margin_x * 2)
            if touches_edge or too_narrow:
                is_panto_incomplete = True
                self.last_incomplete_timestamp = timestamp
                logger.info(
                    "[spark_detector] 标记受电弓显示不全 (is_panto_incomplete=True): box=%s, w=%d, h=%d",
                    best_panto_box, img_w, img_h
                )
            else:
                # 画面中心清晰检出完整正常受电弓，重置显示不全时序锁存
                is_panto_incomplete = False
                self.last_incomplete_timestamp = None
        else:
            # 1. 检查是否存在贴边的低置信度受电弓候选（因严重出画导致置信度下降）
            edge_low_candidate = False
            for lpbox, lpconf in low_panto_boxes:
                lpx1, lpy1, lpx2, lpy2 = lpbox
                if (lpx1 <= margin_x or lpx2 >= img_w - margin_x or
                        lpy1 <= margin_y or lpy2 >= img_h - margin_y):
                    edge_low_candidate = True
                    break
            if edge_low_candidate:
                is_panto_incomplete = True
                self.last_incomplete_timestamp = timestamp
                logger.info(
                    "[spark_detector] 召回贴边低置信度受电弓候选, 维持显示不全状态 (is_panto_incomplete=True)"
                )
            elif (self.last_incomplete_timestamp is not None and timestamp is not None
                  and 0 <= timestamp - self.last_incomplete_timestamp <= self.panto_incomplete_hold_sec):
                # 2. 时序平滑锁存：若近期受电弓贴边残缺且当前无弓，说明受电弓已完全滑出画幅外，维持不全抑制
                is_panto_incomplete = True

        # 视场状态与清晰度辅助判定 (LapVar Analysis)
        top_lap_var = 0.0
        full_lap_var = 0.0
        is_camera_offset = False
        if frame_ref is not None:
            import cv2
            h_valid = int(img_h * 0.82)
            gray = cv2.cvtColor(frame_ref[:h_valid, :], cv2.COLOR_BGR2GRAY)
            top_h = max(10, int(h_valid * 0.30))
            top_gray = gray[:top_h, :]
            top_lap_var = float(cv2.Laplacian(top_gray, cv2.CV_64F).var())
            full_lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # 若未检出合法受电弓，且画面包含大量锋利锐利硬边缘（顶部 TopLap > 160 或全图 FullLap > 300），
            # 真实脏污无弓样本的 TopLap 均值仅 22.6、95% 分位数在 119 以下；
            # 只有拍摄到车顶金属/绝缘子/钣金构件等非受电弓视场时才会产生极高清晰度。
            if best_panto_box is None and (top_lap_var > 160.0 or full_lap_var > 300.0):
                is_camera_offset = True
                logger.info(
                    "[spark_detector] 触发视场偏转标记 (is_camera_offset=True): 无合法受电弓且 (TopLap=%.1f > 160 或 FullLap=%.1f > 300)",
                    top_lap_var, full_lap_var
                )

        if context is not None:
            context.panto_bbox = best_panto_box
            context.panto_confidence = best_panto_conf
            context.is_camera_offset = is_camera_offset
            context.is_panto_incomplete = is_panto_incomplete
            context.top_lap_var = top_lap_var

        # Intense real arcs can temporarily hide the pantograph from YOLO. Use
        # its last detected box briefly, but prefer the current box whenever present.
        timestamp = context.packet.timestamp if context and context.packet else None
        if best_panto_box is not None:
            self.last_panto_bbox = best_panto_box
            self.last_panto_timestamp = timestamp
        spatial_panto_box = best_panto_box
        if (spatial_panto_box is None and self.last_panto_bbox is not None
                and timestamp is not None and self.last_panto_timestamp is not None
                and 0 <= timestamp - self.last_panto_timestamp <= self.panto_bbox_hold_sec):
            spatial_panto_box = self.last_panto_bbox

        # 降弓工况门控：副弓脱网无受流电流，清空火花时序计数器，静默大火花告警；
        # 但保留 YOLO 提取的受电弓目标框供后续 YoloFloatDetector 复用，避免重复推理
        is_panto_raised = context.packet.is_panto_raised if (context and context.packet) else True
        if not is_panto_raised:
            self.current_duration_count = 0
            self.alarmed = False
            self.last_panto_bbox = None
            self.last_panto_timestamp = None
            return DetectionResult(
                fault_type=self.fault_type,
                is_abnormal=False,
                score=0.0,
                level="normal",
                bbox=best_panto_box,
                extra={
                    "panto_bbox": best_panto_box,
                    "is_panto_raised": False,
                    "status": "panto_lowered_muted",
                    "spark_count": 0,
                }
            )

        # 2. 火花空间尺度与持续时间判决
        target_spark_bbox = None
        target_spark_conf = 0.0
        qualifying_spark_boxes = []

        if spark_boxes:
            area_reference_box = spatial_panto_box if self.require_panto_overlap else best_panto_box
            panto_area = (
                (area_reference_box[2] - area_reference_box[0]) * (area_reference_box[3] - area_reference_box[1])
                if area_reference_box else 1.0
            )

            for s_box, s_conf in spark_boxes:
                s_area = (s_box[2] - s_box[0]) * (s_box[3] - s_box[1])
                area_ok = s_area >= self.spark_min_area or (s_area / panto_area) >= self.spark_area_ratio
                overlap_ok = not self.require_panto_overlap or (
                    spatial_panto_box is not None and self._boxes_intersect(s_box, spatial_panto_box)
                )
                if area_ok and overlap_ok:
                    qualifying_spark_boxes.append(s_box)
                    if s_conf > target_spark_conf:
                        target_spark_bbox = s_box
                        target_spark_conf = s_conf

        has_spark_target = bool(qualifying_spark_boxes)

        if has_spark_target:
            self.current_duration_count += 1
        else:
            self.current_duration_count = max(0, self.current_duration_count - 1)

        single_img_mode = bool(context and context.shared_data and context.shared_data.get("single_image_mode", False))
        if single_img_mode:
            is_abnormal = has_spark_target
            level = "critical" if is_abnormal else "normal"
        else:
            is_abnormal = self.current_duration_count >= self.spark_duration_frames
            level = "critical" if is_abnormal else ("warning" if self.current_duration_count >= 5 else "normal")

        if context is not None:
            context.spark_bboxes = qualifying_spark_boxes

        return DetectionResult(
            fault_type=self.fault_type,
            is_abnormal=is_abnormal,
            score=round(target_spark_conf, 3),
            level=level,
            bbox=target_spark_bbox,
            extra={
                "duration_frames": self.current_duration_count,
                "duration_thresh": self.spark_duration_frames,
                "panto_bbox": best_panto_box,
                "spatial_reference_bbox": spatial_panto_box,
                "spark_count": len(qualifying_spark_boxes),
                "raw_spark_count": len(spark_boxes),
                "raw_spark_bboxes": [box for box, _ in spark_boxes],
                "raw_max_spark_conf": round(max_spark_conf, 3),
            }
        )

    def reset(self) -> None:
        self.current_duration_count = 0
        self.alarmed = False
        self.last_panto_bbox = None
        self.last_panto_timestamp = None
        self.last_incomplete_timestamp = None
