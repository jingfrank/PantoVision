"""
End-to-End Pantograph Keypoint & Tilt Anomaly Detector (4-Keypoint HRNet).
Loads trained HRNet weights, infers 4 keypoints via Gaussian heatmap decoding,
and evaluates geometric tilt angles.
"""

import os
import cv2
import numpy as np
import torch
try:
    from .hrnet import HRNetKeypoint, decode_heatmaps
    from .tilt_analyzer import PantographTiltAnalyzer
    from .dataset import PANTOGRAPH_KEYPOINTS
except (ImportError, ValueError):
    import sys
    from pathlib import Path
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from keypoint_tilt.hrnet import HRNetKeypoint, decode_heatmaps
    from keypoint_tilt.tilt_analyzer import PantographTiltAnalyzer
    from keypoint_tilt.dataset import PANTOGRAPH_KEYPOINTS


class PantographTiltDetector:
    def __init__(
        self,
        weights_path=None,
        width=32,
        num_joints=4,
        target_size=(288, 384),
        heatmap_size=(72, 96),
        structural_warning_thresh=1.2,
        structural_alarm_thresh=1.8,
        install_bias_deg=2.0,
        rigid_alarm_thresh=None,
        use_rigid_alarm=False,
        device=None,
    ):
        self.num_joints = num_joints
        self.target_size = target_size
        self.heatmap_size = heatmap_size
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        if weights_path is None:
            default_weights = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "weights",
                f"best_hrnet_w{width}_panto.pth",
            )
            if os.path.exists(default_weights):
                weights_path = default_weights

        self.model = HRNetKeypoint(num_joints=num_joints, width=width)
        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"[Detector] Successfully loaded trained HRNet weights: {weights_path}")
        else:
            print("[Detector] Warning: No trained weights found. Running in uninitialized mode.")

        self.model.to(self.device)
        self.model.eval()

        self.analyzer = PantographTiltAnalyzer(
            structural_warning_thresh=structural_warning_thresh,
            structural_alarm_thresh=structural_alarm_thresh,
            install_bias_deg=install_bias_deg,
            rigid_alarm_thresh=rigid_alarm_thresh,
            use_rigid_alarm=use_rigid_alarm,
        )

    def predict(self, image_path_or_bgr):
        if isinstance(image_path_or_bgr, str):
            img = cv2.imread(image_path_or_bgr)
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {image_path_or_bgr}")
        else:
            img = image_path_or_bgr

        h_orig, w_orig, _ = img.shape
        target_h, target_w = self.target_size
        hm_h, hm_w = self.heatmap_size

        # Preprocess
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = ((img_tensor - mean) / std).unsqueeze(0).to(self.device)

        # Forward
        with torch.no_grad():
            heatmaps = self.model(img_tensor)

        preds_hm, maxvals = decode_heatmaps(heatmaps)

        # Rescale heatmap coordinates back to original image resolution
        scale_x = w_orig / float(hm_w)
        scale_y = h_orig / float(hm_h)

        kpts = preds_hm[0].copy()
        kpts[:, 0] *= scale_x
        kpts[:, 1] *= scale_y
        confs = maxvals[0, :, 0]

        # Geometric tilt evaluation
        kpts_with_conf = np.hstack([kpts, confs[:, np.newaxis]])
        result = self.analyzer.analyze(kpts_with_conf)

        # Render visualization
        vis_img = self.analyzer.render_visualization(img, result)

        return result, vis_img


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pantograph 4-Keypoint Tilt Detection")
    parser.add_argument("--image", "-i", type=str, default="/home/ibd/jyb/pantograph_tilt.png", help="Path to input image")
    parser.add_argument("--weights", "-w", type=str, default="weights/best_hrnet_w32_panto.pth", help="Path to trained weights")
    parser.add_argument("--output", "-o", type=str, default="/home/ibd/jyb/pantograph_tilt_detected.jpg", help="Output path for annotated image")
    parser.add_argument("--install_bias", type=float, default=2.0, help="Camera mounting roll angle bias (deg)")
    parser.add_argument("--structural_warning", type=float, default=1.2, help="Warning threshold for relative tilt (deg)")
    parser.add_argument("--structural_alarm", type=float, default=1.8, help="Alarm threshold for relative tilt (deg)")

    args = parser.parse_args()

    detector = PantographTiltDetector(
        weights_path=args.weights,
        width=32,
        num_joints=4,
        structural_warning_thresh=args.structural_warning,
        structural_alarm_thresh=args.structural_alarm,
        install_bias_deg=args.install_bias,
    )

    print(f"\nEvaluating Image: {args.image}")
    res, vis = detector.predict(args.image)

    print("-" * 50)
    print(f"Status           : [{res['status']}] (Severity: {res['severity']})")
    print(f"Parallelism Tilt : {res['relative_tilt_deg']:+.2f}° ({res['tilt_direction']})")
    print(f"Support Base     : {res['support_angle_deg']:+.2f}° (Base Tilt: {res['rigid_tilt_deg']:+.2f}°)")
    print(f"Horn Line (P1-P4): {res['horn_angle_deg']:+.2f}° (Drop: {res['horn_drop_px']:+.1f} px)")
    print(f"Support (P2-P3)  : {res['support_angle_deg']:+.2f}° (Drop: {res['support_drop_px']:+.1f} px)")
    print(f"Is Anomaly       : {res['is_anomaly']}")
    print("-" * 50)

    cv2.imwrite(args.output, vis)
    print(f"Annotated result saved to: {args.output}\n")
