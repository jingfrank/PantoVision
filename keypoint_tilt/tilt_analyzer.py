"""
Geometric Pose Evaluation and Tilt Anomaly Detection for Pantographs (4-Keypoint Scheme).
Based on Paper Fig. 3: P1(left_horn), P2(left_support), P3(right_support), P4(right_horn).
"""

from collections import deque
import cv2
import numpy as np
from .dataset import PANTOGRAPH_KEYPOINTS, SKELETON_PAIRS


class PantographTiltAnalyzer:
    def __init__(
        self,
        structural_warning_thresh=1.2,  # degrees: relative non-parallel angle (Condition A)
        structural_alarm_thresh=1.8,    # degrees: relative non-parallel angle (Condition A)
        install_bias_deg=0.0,           # camera mounting roll bias angle (for logging only)
        rigid_alarm_thresh=None,        # disabled by default to prevent camera shift false alarms
        use_rigid_alarm=False,          # strictly False: only Condition A is used
        sliding_window_size=5,
        min_consecutive_alarms=3,
    ):
        self.warning_thresh = structural_warning_thresh
        self.alarm_thresh = structural_alarm_thresh
        self.install_bias_deg = install_bias_deg
        self.rigid_alarm_thresh = rigid_alarm_thresh
        self.use_rigid_alarm = use_rigid_alarm
        self.sliding_window_size = sliding_window_size
        self.min_consecutive_alarms = min_consecutive_alarms
        self.history = deque(maxlen=sliding_window_size)
        self.alarmed = False

    @staticmethod
    def calc_angle(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0
        return float(np.degrees(np.arctan2(dy, dx)))

    @staticmethod
    def validate_topology(coords, relative_tilt=0.0, max_physical_deg=20.0):
        """
        校验受电弓 4 关键点空间拓扑与物理姿态角有效性:
        P1(左羊角), P2(左支架), P3(右支架), P4(右羊角)
        """
        if coords is None or len(coords) < 4:
            return False, "KEYPOINTS_COUNT_LESS_THAN_4"
        p1, p2, p3, p4 = coords[0], coords[1], coords[2], coords[3]

        # 1. X 轴严格递增拓扑: P1_x < P2_x < P3_x < P4_x (从左至右展开)
        if not (p1[0] < p2[0] < p3[0] < p4[0]):
            return False, f"X_ORDER_VIOLATED: ({p1[0]:.1f}, {p2[0]:.1f}, {p3[0]:.1f}, {p4[0]:.1f})"

        # 2. 跨度尺度校验
        w_horn = p4[0] - p1[0]
        w_support = p3[0] - p2[0]
        if w_horn < 150.0:
            return False, f"HORN_SPAN_TOO_SMALL: {w_horn:.1f}px"
        if w_support < 50.0:
            return False, f"SUPPORT_SPAN_TOO_SMALL: {w_support:.1f}px"

        # 3. 支架基线与总跨度比例 (正常约为 0.4~0.6)
        ratio = w_support / max(w_horn, 1.0)
        if ratio < 0.20 or ratio > 0.85:
            return False, f"SPAN_RATIO_ANOMALOUS: {ratio:.2f}"

        # 4. 物理偏角合理性校验 (正常受电弓物理倾斜绝不可能出现翻转/垂直)
        if abs(relative_tilt) > max_physical_deg:
            return False, f"PHYSICAL_ANGLE_EXCEEDED: {relative_tilt:.2f}deg"

        return True, "OK"

    @staticmethod
    def calc_relative_angle(p1, p4, p2, p3):
        """
        Calculate relative angle of horn vector P1->P4 in the local coordinate frame
        of support baseline P2->P3 using 2D vector dot and cross product.
        Invariant to camera roll, camera displacement, and vehicle vibration!
        """
        v_base = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=np.float64)
        v_horn = np.array([p4[0] - p1[0], p4[1] - p1[1]], dtype=np.float64)

        # In image coords (y down), cross product > 0 means clockwise (right drop relative to base)
        dot = v_base[0] * v_horn[0] + v_base[1] * v_horn[1]
        cross = v_base[0] * v_horn[1] - v_base[1] * v_horn[0]

        rel_rad = np.arctan2(cross, dot)
        return float(np.degrees(rel_rad))

    def analyze(self, keypoints):
        """
        keypoints: array-like of shape (4, 2) or (4, 3)
          P1 (0): left_horn
          P2 (1): left_support
          P3 (2): right_support
          P4 (3): right_horn
        """
        kpts = np.array(keypoints, dtype=np.float32)
        if kpts.shape[-1] == 3:
            coords = kpts[:, :2]
            confs = kpts[:, 2]
        else:
            coords = kpts
            confs = np.ones(len(kpts))

        # 1. Image-coordinate absolute angles
        angle_horn = self.calc_angle(coords[0], coords[3])
        dy_horn = float(coords[3, 1] - coords[0, 1])

        angle_support = self.calc_angle(coords[1], coords[2])
        dy_support = float(coords[2, 1] - coords[1, 1])

        # 2. Relative angle (P1-P4 relative to P2-P3 local baseline, Condition A)
        relative_tilt = self.calc_relative_angle(coords[0], coords[3], coords[1], coords[2])

        # 3. Rigid overall tilt (for baseline logging)
        rigid_tilt = angle_support - self.install_bias_deg

        # Legacy fused angle for backward compatibility
        fused_angle = float(np.mean([angle_horn, angle_support]))

        abs_rel = abs(relative_tilt)
        abs_rigid = abs(rigid_tilt)

        # 0. 拓扑合理性与物理角度上限校验
        is_valid_topo, topo_reason = self.validate_topology(coords, relative_tilt)

        if not is_valid_topo:
            current_status = "INVALID_TOPOLOGY"
            is_anomaly = False
            severity = "NORMAL"
        else:
            # PURE CONDITION A: Structural parallelism fault detection only
            # Completely immune to camera mounting displacement, vehicle roll, or vibration!
            is_structural_alarm = abs_rel >= self.alarm_thresh
            is_rigid_alarm = self.use_rigid_alarm and (self.rigid_alarm_thresh is not None) and (abs_rigid >= self.rigid_alarm_thresh)

            if is_structural_alarm or is_rigid_alarm:
                current_status = "ALARM_TILT"
                is_anomaly = True
                severity = "CRITICAL"
            elif abs_rel >= self.warning_thresh:
                current_status = "WARNING_TILT"
                is_anomaly = False
                severity = "WARNING"
            else:
                current_status = "NORMAL"
                is_anomaly = False
                severity = "NORMAL"

        self.history.append(is_anomaly)
        sustained_alarm = sum(self.history) >= self.min_consecutive_alarms
        self.alarmed = sustained_alarm

        direction = "RIGHT_DROP" if relative_tilt > 0 else "LEFT_DROP"
        if abs_rel < self.warning_thresh:
            direction = "PARALLEL"

        return {
            "is_anomaly": is_anomaly,
            "sustained_alarm": sustained_alarm,
            "is_valid_topology": is_valid_topo,
            "topology_reason": topo_reason,
            "status": current_status,
            "severity": severity,
            "tilt_direction": direction,
            "relative_tilt_deg": round(relative_tilt, 2),
            "rigid_tilt_deg": round(rigid_tilt, 2),
            "fused_angle_deg": round(fused_angle, 2),
            "horn_angle_deg": round(angle_horn, 2),
            "support_angle_deg": round(angle_support, 2),
            "horn_drop_px": round(dy_horn, 1),
            "support_drop_px": round(dy_support, 1),
            "keypoints": coords.tolist(),
            "confidences": confs.tolist(),
        }

    def render_visualization(self, image, result):
        vis = image.copy()
        kpts = np.array(result["keypoints"], dtype=np.int32)
        status = result["status"]
        is_alarm = result["is_anomaly"]
        fused_angle = result["fused_angle_deg"]
        direction = result["tilt_direction"]

        alarm_color = (0, 0, 255) if is_alarm else (0, 220, 0)
        point_colors = [(255, 120, 0), (0, 255, 255), (0, 255, 255), (255, 120, 0)]
        point_names = ["P1 (Left Horn)", "P2 (Left Support)", "P3 (Right Support)", "P4 (Right Horn)"]

        # 1. Draw Support Line (P2-P3) and Horn Line (P1-P4)
        line_color = (0, 0, 255) if is_alarm else (0, 255, 100)
        # Inner support crossbar
        cv2.line(vis, tuple(kpts[1]), tuple(kpts[2]), line_color, 4, cv2.LINE_AA)
        # Outer horn span virtual reference
        cv2.line(vis, tuple(kpts[0]), tuple(kpts[3]), (0, 180, 255) if not is_alarm else (0, 0, 255), 2, cv2.LINE_AA)
        # Horn to support connects
        cv2.line(vis, tuple(kpts[0]), tuple(kpts[1]), (180, 220, 100), 2, cv2.LINE_AA)
        cv2.line(vis, tuple(kpts[2]), tuple(kpts[3]), (180, 220, 100), 2, cv2.LINE_AA)

        # 2. Draw 4 Keypoints & Labels
        for i, pt in enumerate(kpts):
            c = point_colors[i]
            cv2.circle(vis, tuple(pt), 8, c, -1, cv2.LINE_AA)
            cv2.circle(vis, tuple(pt), 10, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                vis, f"P{i+1}", (pt[0] - 15, pt[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
            )

        # 3. Top HUD Dashboard
        hud_h = 135
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (vis.shape[1], hud_h), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.85, vis, 0.15, 0, vis)

        cv2.putText(vis, "PANTOGRAPH TILT MONITOR (4-KEYPOINT)", (30, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"STATUS: [{status}]", (30, 100),
                    cv2.FONT_HERSHEY_DUPLEX, 1.25, alarm_color, 2, cv2.LINE_AA)

        col_x = 880
        rel_angle = result.get("relative_tilt_deg", 0.0)
        rigid_angle = result.get("rigid_tilt_deg", 0.0)
        line1 = f"Parallelism Tilt: {rel_angle:+.2f} deg ({direction})  [Alarm: >={self.alarm_thresh:.1f} deg]"
        line2 = f"Horn Line (P1-P4): {result['horn_angle_deg']:+.2f} deg | Drop: {result['horn_drop_px']:+.1f} px"
        line3 = f"Support Base (P2-P3): {result['support_angle_deg']:+.2f} deg | Base Tilt: {rigid_angle:+.2f} deg"

        cv2.putText(vis, line1, (col_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, line2, (col_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (200, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, line3, (col_x, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (170, 255, 170) if not is_alarm else (180, 180, 255), 2, cv2.LINE_AA)

        return vis
