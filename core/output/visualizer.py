#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Output Layer: Industrial Visualizer.
3+2 OSD Watermark rendering and comprehensive HUD diagnostic canvas.

故障专属渲染架构 (Per-Fault Context-Aware Renderer):
  render_hud() 根据 primary_fault 分发到对应的 _render_xxx() 私有方法，
  每种故障类型拥有独立的图层逻辑，保证故障时刻图与视频帧语义严格对应。

  支持故障类型:
    big_spark         -> _render_spark       (大火花：红框 + 四角红色角标)
    dirty             -> _render_dirt        (脏污：蓝色遮罩 + 检测屏蔽提示)
    foreign_object    -> _render_foreign     (悬挂异物：黄色多框 + 数量标注)
    structure_abnormal-> _render_structural  (结构异常：橙红多框 + ROI 辅助框)
    tilt              -> _render_tilt        (倾斜：姿态线 + 偏角大字)
    abnormal_float    -> _render_float       (飘弓：橙色框 + 垂向速度标注)
    正常              -> _render_normal      (正常：绿色姿态线)
"""

import os
import cv2
import time
import logging
import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Dict, Any, List, Tuple
from core.data_types import FusionResult, FramePacket, DetectionResult

logger = logging.getLogger("visualizer")


# ---------------------------------------------------------------------------
# 模块级外部直接调用接口
# ---------------------------------------------------------------------------

def draw_osd_watermark(
    frame: np.ndarray,
    train_id: str = "CR400BF",
    car_number: Any = "06",
    speed: float = 0.0,
    camera_id: Any = "0",
    ts: Optional[float] = None,
    font_path: Optional[str] = None,
) -> np.ndarray:
    """
    在 OpenCV 图像帧上叠加 3+2 双侧车规工业 OSD 水印:
    - 左下角 (3行): 车次、车厢号、速度
    - 右下角 (2行): 时间戳 (YYYY-MM-DD HH:MM:SS)、位置号 (相机号)
    """
    try:
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        if not font_path:
            proj_fonts = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "fonts"))
            candidate_fonts = [
                os.path.join(proj_fonts, "simhei.ttf"),
                os.path.join(proj_fonts, "wqy-zenhei.ttc"),
                "/app/fonts/simhei.ttf",
                "/app/fonts/wqy-zenhei.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                "/home/hirain/.local/share/fonts/simhei.ttf",
                os.path.expanduser("~/.local/share/fonts/simhei.ttf"),
                os.path.expanduser("~/.local/share/fonts/msyh.ttc"),
                os.path.expanduser("~/.local/share/fonts/wqy-zenhei.ttc"),
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simhei.ttf",
            ]
            font_path = next((f for f in candidate_fonts if os.path.exists(f)), None)

        font_size = 24
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

        h, w = frame.shape[:2]
        dt_str = datetime.datetime.fromtimestamp(ts if ts is not None else time.time()).strftime("%Y-%m-%d %H:%M:%S")
        car_str = str(car_number).zfill(2)
        cam_str = str(camera_id)

        left_lines = [f"车次: {train_id}", f"车厢号: {car_str}", f"速度: {speed:.0f}km/h"]
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


# ---------------------------------------------------------------------------
# 故障专属颜色与横幅配色常量
# ---------------------------------------------------------------------------

# BGR
_COLOR_RED       = (0,   0,   255)
_COLOR_GREEN     = (0,   200,  0)
_COLOR_BLUE      = (200, 100,  0)
_COLOR_YELLOW    = (0,   255, 255)
_COLOR_ORANGE    = (0,   140, 255)
_COLOR_PURPLE    = (180,  0,  180)
_COLOR_CYAN      = (255, 255,  0)
_COLOR_WHITE     = (255, 255, 255)
_COLOR_DARK      = (20,   20,  20)

# 横幅背景色（BGR）
_BANNER_BG = {
    "big_spark":          (0,   0,  180),   # 深红
    "dirty":              (140, 80,   0),   # 深蓝/青
    "foreign_object":     (0,  100,  200),  # 橙
    "structure_abnormal": (0,   70,  180),  # 深橙红
    "tilt_alarm":         (0,   0,  180),   # 红（告警）
    "tilt_warning":       (0,  100,  200),  # 橙（预警）
    "tilt_normal":        (0,  140,   0),   # 绿（正常）
    "abnormal_float":     (160,  0,  160),  # 紫
    "normal":             (0,  140,   0),   # 绿
}


# ---------------------------------------------------------------------------
# Visualizer 主类
# ---------------------------------------------------------------------------

class Visualizer:
    """工业级 HUD 诊断画板与 3+2 双侧 OSD 水印渲染器（故障专属渲染架构）"""

    # 故障类型 -> 专属渲染方法名映射
    _FAULT_RENDER_MAP: Dict[str, str] = {
        "big_spark":          "_render_spark",
        "dirty":              "_render_dirt",
        "foreign_object":     "_render_foreign",
        "structure_abnormal": "_render_structural",
        "tilt":               "_render_tilt",
        "abnormal_float":     "_render_float",
    }

    def __init__(self):
        # 探测系统中文字体
        proj_fonts = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "fonts"))
        candidate_fonts = [
            os.path.join(proj_fonts, "simhei.ttf"),
            os.path.join(proj_fonts, "wqy-zenhei.ttc"),
            "/app/fonts/simhei.ttf",
            "/app/fonts/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/home/hirain/.local/share/fonts/simhei.ttf",
            os.path.expanduser("~/.local/share/fonts/simhei.ttf"),
            os.path.expanduser("~/.local/share/fonts/msyh.ttc"),
            os.path.expanduser("~/.local/share/fonts/wqy-zenhei.ttc"),
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
        ]
        self.font_path = next((f for f in candidate_fonts if os.path.exists(f)), None)
        if self.font_path:
            logger.info(f"[Visualizer] 成功挂载中文字体: {self.font_path}")
        else:
            logger.warning("[Visualizer] ⚠️ 未探测到有效中文字体文件，文字可能退化为方块乱码！")

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def draw_osd_watermark(self, frame: np.ndarray, packet: FramePacket) -> np.ndarray:
        """绘制 3+2 双侧 OSD 水印"""
        return draw_osd_watermark(
            frame=frame,
            train_id=packet.train_id,
            car_number=packet.car_number,
            speed=packet.speed,
            camera_id=packet.camera_id,
            ts=packet.timestamp,
            font_path=self.font_path,
        )

    def render_hud(self, frame: np.ndarray, fusion_res: FusionResult, packet: FramePacket) -> np.ndarray:
        """
        渲染工业 HUD 诊断画板（故障专属渲染分发器）。

        根据 fusion_res.active_alarms[0].fault_type 决定渲染策略：
        每种故障类型调用对应的 _render_xxx 私有方法，
        所有模式共享顶部横幅与 3+2 OSD 水印。
        """
        out = frame.copy()
        results = fusion_res.detector_results

        # 确定主告警类型（FusionEngine 已按优先级排序，[0] 即最高优先级）
        fault_type: Optional[str] = None
        if fusion_res.active_alarms:
            fault_type = fusion_res.active_alarms[0].get("fault_type")

        # 分发到故障专属渲染器
        render_method_name = self._FAULT_RENDER_MAP.get(fault_type, "_render_normal")
        renderer = getattr(self, render_method_name)
        out = renderer(out, results, fusion_res, packet)

        # 共享：顶部状态横幅（PIL 中文文字）
        out = self._draw_common_banner(out, fusion_res, packet, fault_type)

        # 共享：3+2 OSD 水印
        return self.draw_osd_watermark(out, packet)

    # ------------------------------------------------------------------
    # 故障专属渲染器
    # ------------------------------------------------------------------

    def _render_spark(
        self,
        out: np.ndarray,
        results: Dict[str, "DetectionResult"],
        fusion_res: FusionResult,
        packet: FramePacket,
    ) -> np.ndarray:
        """大火花专属渲染：红色 bbox + 受电弓蓝色辅助框 + 四角红色角标"""
        spark_res = results.get("spark") or results.get("big_spark")

        if spark_res and spark_res.bbox and spark_res.is_abnormal:
            x1, y1, x2, y2 = spark_res.bbox
            # 火花框：红色 3px
            cv2.rectangle(out, (x1, y1), (x2, y2), _COLOR_RED, 3)
            cv2.putText(
                out, f"SPARK ({spark_res.score:.2f})",
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, _COLOR_WHITE, 2,
            )

        # 受电弓主框（蓝色细框，来自 spark_res.extra.panto_bbox）
        panto_box = (spark_res.extra or {}).get("panto_bbox") if spark_res else None
        if panto_box and len(panto_box) == 4:
            cv2.rectangle(out, (panto_box[0], panto_box[1]), (panto_box[2], panto_box[3]),
                          (255, 100, 0), 1)

        # 四角红色角标（强警示感）
        self._draw_corner_markers(out, color=_COLOR_RED, length=50, thickness=5)
        return out

    def _render_dirt(
        self,
        out: np.ndarray,
        results: Dict[str, "DetectionResult"],
        fusion_res: FusionResult,
        packet: FramePacket,
    ) -> np.ndarray:
        """脏污专属渲染：蓝色半透明遮罩 + 四角蓝色角标。不渲染 foreign/tilt/struct。"""
        h, w = out.shape[:2]

        # 半透明蓝色遮罩（alpha=0.18）
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 100, 200), -1)
        cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)

        # 四角蓝色角标
        self._draw_corner_markers(out, color=(0, 180, 255), length=40, thickness=4)
        return out

    def _render_foreign(
        self,
        out: np.ndarray,
        results: Dict[str, "DetectionResult"],
        fusion_res: FusionResult,
        packet: FramePacket,
    ) -> np.ndarray:
        """悬挂异物专属渲染：黄色多框 + 异物数量标注"""
        foreign_res = results.get("foreign_object")
        if not (foreign_res and foreign_res.is_abnormal):
            return out

        bboxes: List = foreign_res.extra.get("bboxes") or (
            [foreign_res.bbox] if foreign_res.bbox else []
        )

        for fb in bboxes:
            x1, y1, x2, y2 = fb[:4]
            cv2.rectangle(out, (x1, y1), (x2, y2), _COLOR_YELLOW, 2)

        count = len(bboxes)
        if bboxes:
            x1, y1 = bboxes[0][0], bboxes[0][1]
            cv2.putText(
                out, f"x{count} FOREIGN OBJ",
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, _COLOR_YELLOW, 2,
            )
        return out

    def _render_structural(
        self,
        out: np.ndarray,
        results: Dict[str, "DetectionResult"],
        fusion_res: FusionResult,
        packet: FramePacket,
    ) -> np.ndarray:
        """结构异常专属渲染：橙红多框 + 缺陷数量标注"""
        struct_res = results.get("structural") or results.get("structure_abnormal")
        if not (struct_res and struct_res.is_abnormal):
            return out

        bboxes: List = struct_res.extra.get("bboxes") or (
            [struct_res.bbox] if struct_res.bbox else []
        )

        for sb in bboxes:
            sx1, sy1, sx2, sy2 = sb[:4]
            cv2.rectangle(out, (sx1, sy1), (sx2, sy2), _COLOR_ORANGE, 3)

        count = len(bboxes)
        if bboxes:
            sx1, sy1 = bboxes[0][0], bboxes[0][1]
            cv2.putText(
                out, f"x{count} STRUCT DEFECT",
                (sx1, max(15, sy1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _COLOR_ORANGE, 2,
            )
        return out

    def _render_tilt(
        self,
        out: np.ndarray,
        results: Dict[str, "DetectionResult"],
        fusion_res: FusionResult,
        packet: FramePacket,
    ) -> np.ndarray:
        """
        倾斜专属渲染：
        - 4 关键点（青色）
        - 下横梁基线（蓝色）
        - 上滑板工作线（告警红 / 正常绿）
        - 偏角大字标注（画面中央）
        """
        tilt_res = results.get("tilt")
        if not (tilt_res and tilt_res.keypoints and len(tilt_res.keypoints) >= 4):
            return out

        kpts = tilt_res.keypoints
        p1, p2, p3, p4 = [(int(pt[0]), int(pt[1])) for pt in kpts[:4]]

        # 关键点（青色实心圆）
        for pt in [p1, p2, p3, p4]:
            cv2.circle(out, pt, 6, _COLOR_YELLOW, -1)

        # 下横梁基线（蓝色）
        cv2.line(out, p2, p3, (255, 200, 0), 2)

        # 上滑板工作线（异常红 / 正常绿）
        horn_color = _COLOR_RED if tilt_res.is_abnormal else _COLOR_GREEN
        cv2.line(out, p1, p4, horn_color, 3)

        # 偏角大字标注（画面中央）
        rel_tilt = tilt_res.extra.get("relative_tilt_deg", 0.0)
        h, w = out.shape[:2]
        tilt_text = f"{rel_tilt:+.2f} deg"
        text_x = max(10, w // 2 - 120)
        text_y = max(60, h // 2)
        # 黑色阴影
        cv2.putText(out, tilt_text, (text_x + 2, text_y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 6)
        # 彩色主字
        cv2.putText(out, tilt_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, horn_color, 4)

        return out

    def _render_float(
        self,
        out: np.ndarray,
        results: Dict[str, "DetectionResult"],
        fusion_res: FusionResult,
        packet: FramePacket,
    ) -> np.ndarray:
        """飘弓专属渲染：橙色受电弓框 + 垂向速度标注"""
        float_res = results.get("float") or results.get("abnormal_float")
        if not float_res:
            return out

        # 受电弓框（橙色）
        if float_res.bbox:
            x1, y1, x2, y2 = float_res.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), 2)

        # 垂向速度标注
        v_speed = float_res.extra.get("vertical_speed")
        reason  = float_res.extra.get("reason", "")
        if v_speed is not None:
            speed_text = f"V {v_speed:.1f} px/s  OVER LIMIT"
            h, w = out.shape[:2]
            label_x = 20
            label_y = h // 2
            cv2.putText(out, speed_text, (label_x + 2, label_y + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
            cv2.putText(out, speed_text, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, _COLOR_RED, 3)
        elif reason == "down_state_raised":
            h, _ = out.shape[:2]
            cv2.putText(out, "PANTO RAISED (DOWN CMD)",
                        (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, _COLOR_RED, 2)

        return out

    def _render_normal(
        self,
        out: np.ndarray,
        results: Dict[str, "DetectionResult"],
        fusion_res: FusionResult,
        packet: FramePacket,
    ) -> np.ndarray:
        """正常帧渲染：绘制 tilt 关键点（绿色），帮助运维确认受电弓姿态正常"""
        tilt_res = results.get("tilt")
        if tilt_res and tilt_res.keypoints and len(tilt_res.keypoints) >= 4:
            kpts = tilt_res.keypoints
            p1, p2, p3, p4 = [(int(pt[0]), int(pt[1])) for pt in kpts[:4]]
            for pt in [p1, p2, p3, p4]:
                cv2.circle(out, pt, 5, _COLOR_YELLOW, -1)
            cv2.line(out, p2, p3, (255, 200, 0), 2)
            cv2.line(out, p1, p4, _COLOR_GREEN, 2)
        return out

    # ------------------------------------------------------------------
    # 共享辅助方法
    # ------------------------------------------------------------------

    def _draw_corner_markers(
        self,
        out: np.ndarray,
        color: Tuple[int, int, int],
        length: int = 40,
        thickness: int = 4,
    ) -> None:
        """在画面四角绘制 L 形角标（高亮视觉警示）"""
        h, w = out.shape[:2]
        corners  = [(0, 0), (w, 0), (0, h), (w, h)]
        x_dirs   = [1, -1,  1, -1]
        y_dirs   = [1,  1, -1, -1]
        for (cx, cy), xd, yd in zip(corners, x_dirs, y_dirs):
            cv2.line(out, (cx, cy), (cx + xd * length, cy), color, thickness)
            cv2.line(out, (cx, cy), (cx, cy + yd * length), color, thickness)

    def _draw_common_banner(
        self,
        out: np.ndarray,
        fusion_res: FusionResult,
        packet: FramePacket,
        fault_type: Optional[str],
    ) -> np.ndarray:
        """
        共享顶部状态横幅（PIL 中文文字渲染）：
          第 1 行（大字）：故障类型中文名 + 主告警
          第 2 行（小字）：时间戳 + 告警数 + 故障专属补充信息
          第 3 行（小字，仅脏污）：检测屏蔽提示
        """
        # ── OpenCV 横幅底框 ──────────────────────────────────────────
        banner_color_bgr = (0, 0, 255) if fusion_res.has_fault else (0, 140, 0)
        if fault_type and fault_type in _BANNER_BG:
            banner_color_bgr = _BANNER_BG[fault_type]
        elif fault_type == "tilt":
            # tilt 根据 level 细分
            tilt_res = fusion_res.detector_results.get("tilt")
            if tilt_res:
                if tilt_res.level == "critical":
                    banner_color_bgr = _BANNER_BG["tilt_alarm"]
                elif tilt_res.level == "warning":
                    banner_color_bgr = _BANNER_BG["tilt_warning"]
                else:
                    banner_color_bgr = _BANNER_BG["tilt_normal"]

        cv2.rectangle(out, (10, 10), (700, 100), _COLOR_DARK, -1)
        cv2.rectangle(out, (10, 10), (700, 100), banner_color_bgr, 2)

        # ── PIL 中文文字叠加 ──────────────────────────────────────────
        try:
            img_pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            font_title = (
                ImageFont.truetype(self.font_path, 22) if self.font_path
                else ImageFont.load_default()
            )
            font_sub = (
                ImageFont.truetype(self.font_path, 16) if self.font_path
                else ImageFont.load_default()
            )

            # 第 1 行：故障名称（主告警）
            title_rgb = (255, 60, 60) if fusion_res.has_fault else (60, 230, 60)
            draw.text((20, 18), f"受电弓故障监测: {fusion_res.primary_fault}",
                      fill=title_rgb, font=font_title)

            # 第 2 行：时间戳 + 告警数 + 故障专属补充
            extra_info = self._build_extra_info(fault_type, fusion_res)
            sub_line = (
                f"时间: {packet.timestamp:.1f}s"
                f" | 告警数: {len(fusion_res.active_alarms)}"
                f"{extra_info}"
            )
            draw.text((20, 48), sub_line, fill=(230, 230, 230), font=font_sub)

            # 第 3 行（仅脏污）：检测屏蔽提示
            if fault_type == "dirty":
                dirt_res = (
                    fusion_res.detector_results.get("dirt")
                    or fusion_res.detector_results.get("dirty")
                )
                score_str = f"{dirt_res.score:.2f}" if dirt_res else "?"
                draw.text(
                    (20, 72),
                    f"⚠ 镜头脏污({score_str}) — 外物/结构检测已屏蔽",
                    fill=(0, 200, 255),
                    font=font_sub,
                )

            out = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception:
            pass

        return out

    def _build_extra_info(
        self,
        fault_type: Optional[str],
        fusion_res: FusionResult,
    ) -> str:
        """根据当前故障类型构建顶部横幅第 2 行的专属补充信息字符串"""
        results = fusion_res.detector_results
        try:
            if fault_type == "big_spark":
                spark_res = results.get("spark") or results.get("big_spark")
                if spark_res:
                    dur = spark_res.extra.get("duration_frames", 0)
                    thr = spark_res.extra.get("duration_thresh", 0)
                    return f" | 持续帧数: {dur}/{thr}"

            elif fault_type == "dirty":
                dirt_res = results.get("dirt") or results.get("dirty")
                if dirt_res:
                    cells = dirt_res.extra.get("persistent_cells", 0)
                    img_score = dirt_res.extra.get("image_score", 0.0)
                    return f" | 持续格子: {cells} | 分数: {img_score:.2f}"

            elif fault_type == "foreign_object":
                foreign_res = results.get("foreign_object")
                if foreign_res:
                    count = foreign_res.extra.get("foreign_count", 0)
                    img_score = foreign_res.extra.get("image_score", 0.0)
                    return f" | 异物数: {count} | 区域分: {img_score:.2f}"

            elif fault_type == "structure_abnormal":
                struct_res = results.get("structural") or results.get("structure_abnormal")
                if struct_res:
                    count = struct_res.extra.get("defect_count", 0)
                    img_score = struct_res.extra.get("image_score", 0.0)
                    return f" | 缺陷区域: {count} | 异常分: {img_score:.2f}"

            elif fault_type == "tilt":
                tilt_res = results.get("tilt")
                if tilt_res:
                    rel_tilt = tilt_res.extra.get("relative_tilt_deg", 0.0)
                    status   = tilt_res.extra.get("status", "")
                    sustained = tilt_res.extra.get("sustained_alarm", False)
                    return (
                        f" | 偏角: {rel_tilt:+.2f}°"
                        f" | 状态: {status}"
                        f"{' | 持续告警' if sustained else ''}"
                    )

            elif fault_type == "abnormal_float":
                float_res = results.get("float") or results.get("abnormal_float")
                if float_res:
                    v_speed = float_res.extra.get("vertical_speed")
                    abn_cnt = float_res.extra.get("abnormal_frame_count", 0)
                    raised  = float_res.extra.get("is_panto_raised", True)
                    state_str = "受流中" if raised else "降弓"
                    spd_str = f"{v_speed:.1f}px/s" if v_speed is not None else "-"
                    return f" | 垂向速度: {spd_str} | 连续异常帧: {abn_cnt} | 弓态: {state_str}"

        except Exception:
            pass
        return ""
