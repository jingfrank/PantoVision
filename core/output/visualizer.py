#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Output Layer: Industrial Visualizer.
3+2 OSD Watermark rendering and comprehensive HUD diagnostic canvas.
"""

import os
import cv2
import time
import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Dict, Any, List
from core.data_types import FusionResult, FramePacket


class Visualizer:
    """工业级 HUD 诊断画板与 3+2 双侧 OSD 水印渲染器"""

    def __init__(self):
        # 探测系统中文字体
        candidate_fonts = [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
        ]
        self.font_path = next((f for f in candidate_fonts if os.path.exists(f)), None)

    def draw_osd_watermark(self, frame: np.ndarray, packet: FramePacket) -> np.ndarray:
        """绘制 3+2 双侧 OSD 水印"""
        try:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font_size = 24
            font = ImageFont.truetype(self.font_path, font_size) if self.font_path else ImageFont.load_default()

            h, w = frame.shape[:2]
            dt_str = datetime.datetime.fromtimestamp(packet.timestamp).strftime("%Y-%m-%d %H:%M:%S")
            car_str = str(packet.car_number).zfill(2)
            cam_str = str(packet.camera_id)

            left_lines = [f"车次: {packet.train_id}", f"车厢号: {car_str}", f"速度: {packet.speed:.0f}km/h"]
            right_lines = [f"{dt_str}", f"位置号: {cam_str}"]

            # 左下角 (3行)
            y_start_left = h - 105
            for i, line in enumerate(left_lines):
                draw.text((30, y_start_left + i * 30), line, font=font, fill=(255, 255, 255))

            # 右下角 (2行)
            y_start_right = h - 75
            for i, line in enumerate(right_lines):
                if hasattr(font, "getbbox"):
                    bbox = font.getbbox(line)
                    text_w = bbox[2] - bbox[0]
                else:
                    text_w = len(line) * 14
                draw.text((max(10, w - text_w - 30), y_start_right + i * 30), line, font=font, fill=(255, 255, 255))

            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception:
            return frame

    def render_hud(self, frame: np.ndarray, fusion_res: FusionResult, packet: FramePacket) -> np.ndarray:
        """渲染工业 HUD 诊断画板"""
        out = frame.copy()
        h, w = out.shape[:2]

        # 1. 顶部状态横幅 (Header Banner)
        color = (0, 0, 255) if fusion_res.has_fault else (0, 200, 0)
        cv2.rectangle(out, (10, 10), (520, 85), (20, 20, 20), -1)
        cv2.rectangle(out, (10, 10), (520, 85), color, 2)

        # 2. 绘制各检测器目标框与特征
        results = fusion_res.detector_results

        # (a) 大火花框 (红闪)
        spark_res = results.get("spark")
        if spark_res and spark_res.bbox and spark_res.is_abnormal:
            x1, y1, x2, y2 = spark_res.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(out, f"SPARK ({spark_res.score:.2f})", (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # (b) 悬挂异物框 (黄)
        foreign_res = results.get("foreign_object")
        if foreign_res and foreign_res.bbox and foreign_res.is_abnormal:
            x1, y1, x2, y2 = foreign_res.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(out, "FOREIGN OBJ", (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # (c) 结构异常缺陷框 (橙红)
        struct_res = results.get("structural")
        if struct_res and struct_res.is_abnormal:
            s_boxes = struct_res.extra.get("bboxes") or ([struct_res.bbox] if struct_res.bbox else [])
            for sb in s_boxes:
                sx1, sy1, sx2, sy2 = sb[:4]
                cv2.rectangle(out, (sx1, sy1), (sx2, sy2), (0, 140, 255), 3)
                cv2.putText(out, "STRUCT DEFECT", (sx1, max(15, sy1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

        # (c) HRNet 4 关键点与姿态线
        tilt_res = results.get("tilt")
        if tilt_res and tilt_res.keypoints:
            kpts = tilt_res.keypoints
            if len(kpts) >= 4:
                p1, p2, p3, p4 = [(int(pt[0]), int(pt[1])) for pt in kpts[:4]]
                # 绘制关键点
                for pt in [p1, p2, p3, p4]:
                    cv2.circle(out, pt, 6, (0, 255, 255), -1)
                # 绘制下横梁基线 (蓝色基准)
                cv2.line(out, p2, p3, (255, 200, 0), 2)
                # 绘制上滑板工作线 (绿色或红色)
                horn_color = (0, 0, 255) if tilt_res.is_abnormal else (0, 255, 0)
                cv2.line(out, p1, p4, horn_color, 3)

        # 3. 渲染中文字符 (通过 PIL)
        try:
            img_pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font_title = ImageFont.truetype(self.font_path, 22) if self.font_path else ImageFont.load_default()
            font_sub = ImageFont.truetype(self.font_path, 16) if self.font_path else ImageFont.load_default()

            title_color = (255, 60, 60) if fusion_res.has_fault else (60, 230, 60)
            draw.text((20, 20), f"受电弓故障监测: {fusion_res.primary_fault}", fill=title_color, font=font_title)

            tilt_deg_str = ""
            if tilt_res and "relative_tilt_deg" in tilt_res.extra:
                tilt_deg_str = f" | 相对偏角: {tilt_res.extra['relative_tilt_deg']:+.2f}°"

            draw.text((20, 52), f"时间: {packet.timestamp:.1f}s | 告警数: {len(fusion_res.active_alarms)}{tilt_deg_str}",
                      fill=(230, 230, 230), font=font_sub)

            out = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception:
            pass

        # 4. 叠加 3+2 OSD
        final = self.draw_osd_watermark(out, packet)
        return final
