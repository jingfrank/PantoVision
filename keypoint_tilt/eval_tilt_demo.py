"""
Evaluation and Demonstration Script using REAL TRAINED HRNet-W32 weights.
Evaluates both normal frame and target tilt anomaly image (/home/ibd/jyb/pantograph_tilt.png).
"""

import os
import sys
import cv2

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from keypoint_tilt.detector import PantographTiltDetector


def main():
    tilt_img_path = "/home/ibd/jyb/pantograph_tilt.png"
    normal_img_path = "/mnt/T4/jyb/gongwang_fastad/dataset/train/train_0161_spark_1_t000.0s_f00000.jpg"

    artifact_dir = "/home/ibd/.gemini/antigravity-cli/brain/11c8624d-8ce9-4ad0-b2b3-4509751bf72d"

    detector = PantographTiltDetector(
        width=32,
        num_joints=4,
        tilt_warning_thresh=1.2,
        tilt_alarm_thresh=1.8,
    )

    print("=" * 60)
    print("1. EVALUATING NORMAL FRAME WITH TRAINED HRNet-W32")
    print("=" * 60)
    norm_result, norm_vis = detector.predict(normal_img_path)
    norm_out_path = os.path.join(artifact_dir, "pantograph_normal_detected.jpg")
    cv2.imwrite(norm_out_path, norm_vis)
    cv2.imwrite("/mnt/T4/jyb/gongwang_hrnet_tilt/pantograph_normal_detected.jpg", norm_vis)

    print(f"Normal Frame Status: [{norm_result['status']}] (Severity: {norm_result['severity']})")
    print(f"  Fused Tilt Angle : {norm_result['fused_angle_deg']:+.2f}°")
    print(f"  Horn Line (P1-P4): {norm_result['horn_angle_deg']:+.2f}° (Drop: {norm_result['horn_drop_px']:+.1f} px)")
    print(f"  Support (P2-P3)  : {norm_result['support_angle_deg']:+.2f}° (Drop: {norm_result['support_drop_px']:+.1f} px)")
    print(f"  Saved Image      : {norm_out_path}")

    print("\n" + "=" * 60)
    print(f"2. EVALUATING TILT ANOMALY IMAGE WITH TRAINED HRNet-W32: {tilt_img_path}")
    print("=" * 60)
    tilt_result, tilt_vis = detector.predict(tilt_img_path)
    tilt_out_path = os.path.join(artifact_dir, "pantograph_tilt_detected.jpg")
    cv2.imwrite(tilt_out_path, tilt_vis)
    cv2.imwrite("/home/ibd/jyb/pantograph_tilt_detected.jpg", tilt_vis)
    cv2.imwrite("/mnt/T4/jyb/gongwang_hrnet_tilt/pantograph_tilt_detected.jpg", tilt_vis)

    print(f"Target Image Status: [{tilt_result['status']}] (Severity: {tilt_result['severity']})")
    print(f"  Anomaly Flag     : {tilt_result['is_anomaly']}")
    print(f"  Tilt Direction   : {tilt_result['tilt_direction']}")
    print(f"  Fused Tilt Angle : {tilt_result['fused_angle_deg']:+.2f}°")
    print(f"  Horn Line (P1-P4): {tilt_result['horn_angle_deg']:+.2f}° (Drop: {tilt_result['horn_drop_px']:+.1f} px)")
    print(f"  Support (P2-P3)  : {tilt_result['support_angle_deg']:+.2f}° (Drop: {tilt_result['support_drop_px']:+.1f} px)")
    print(f"  Saved Image      : {tilt_out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
