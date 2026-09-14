#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pantograph-Catenary FastAD Model Wrapper.
Integrates FastAD (Fast Anomaly Detection) with Gaussian & Quantile CDF estimators.
"""

import os
import sys
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from typing import Optional, Tuple, Dict, Union, List

# Add fastad_core to sys.path
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fastad_core"
    ),
)
from fastad3 import FastAD
from fastad_common import RescaleSegmentor


class PantographFastADModel:
    """
    High-level FastAD Model wrapper for Pantograph & Catenary defect detection.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layer_to_extract: Union[int, List[int]] = [2, 3],
        input_size: Tuple[int, int] = (256, 256),
        patchsize: int = 3,
        stride: int = 1,
        n_quantiles: int = 3,
        fm_quantiles: int = 2,
        min_q: float = 0.01,
        max_q: float = 0.99,
        patch_covariance_method: str = "full",
        global_covariance_method: str = "LedoitWolf",
        regularization: float = 0.01,
        pretrained_weights_path: Optional[str] = None,
        anomaly_threshold: float = 2.0,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.backbone_name = backbone
        self.layer_to_extract = (
            [layer_to_extract] if isinstance(layer_to_extract, int) else list(layer_to_extract)
        )
        self.input_size = tuple(input_size)
        self.patchsize = patchsize
        self.stride = stride
        self.n_quantiles = n_quantiles
        self.fm_quantiles = fm_quantiles
        self.min_q = min_q
        self.max_q = max_q
        self.patch_covariance_method = patch_covariance_method
        self.global_covariance_method = global_covariance_method
        self.regularization = regularization
        self.pretrained_weights_path = pretrained_weights_path
        self.anomaly_threshold = anomaly_threshold

        self.core = FastAD(
            backbone=self.backbone_name,
            layer_to_extract=self.layer_to_extract,
            device=torch.device(self.device),
            patchsize=self.patchsize,
            stride=self.stride,
            target_size=self.input_size,
            n_quantiles=self.n_quantiles,
            fm_quantiles=self.fm_quantiles,
            min_q=self.min_q,
            max_q=self.max_q,
            fm_min_q=self.min_q,
            fm_max_q=self.max_q,
            patch_covariance_method=self.patch_covariance_method,
            global_covariance_method=self.global_covariance_method,
            regularization=self.regularization,
            pretrained_weights_path=self.pretrained_weights_path,
        )

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (rgb.shape[0], rgb.shape[1]) != (self.input_size[1], self.input_size[0]):
            resized = cv2.resize(rgb, self.input_size, interpolation=cv2.INTER_AREA)
        else:
            resized = rgb
        return self.transform(resized).unsqueeze(0)

    def fit_from_frames(
        self,
        train_frames: List[np.ndarray],
        val_frames: Optional[List[np.ndarray]] = None,
        batch_size: int = 16,
        val_ratio: float = 0.2,
    ):
        """
        Fits FastAD Gaussian estimators using normal baseline video frames.
        """
        if val_frames is None:
            N = len(train_frames)
            assert N > 0, "No training frames provided!"
            indices = np.random.RandomState(42).permutation(N)
            val_size = max(10, int(N * val_ratio))
            train_indices = indices[val_size:]
            val_indices = indices[:val_size]
            actual_train = [train_frames[i] for i in train_indices]
            actual_val = [train_frames[i] for i in val_indices]
        else:
            actual_train = train_frames
            actual_val = val_frames

        print(
            f"[FastAD] Fitting model: Train={len(actual_train)} frames, Val={len(actual_val)} frames (batch_size={batch_size})..."
        )

        class FrameDataset(torch.utils.data.Dataset):
            def __init__(self, frames, preproc_fn):
                self.frames = frames
                self.preproc_fn = preproc_fn

            def __len__(self):
                return len(self.frames)

            def __getitem__(self, idx):
                return {"image": self.preproc_fn(self.frames[idx]).squeeze(0)}

        train_loader = torch.utils.data.DataLoader(
            FrameDataset(actual_train, self._preprocess_frame),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        val_loader = torch.utils.data.DataLoader(
            FrameDataset(actual_val, self._preprocess_frame),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        # 1. Fit Gaussian Estimators
        print("[FastAD] Estimating Gaussian patch & global distributions...")
        self.core.fit(train_loader)

        # 2. Validate & Fit Z-Score Scalers
        print("[FastAD] Calibrating Z-score anomaly scalers on validation frames...")
        self.core.validate(val_loader)
        print("[FastAD] FastAD model fitting completed successfully.")

    def predict(
        self,
        frame: np.ndarray,
        return_heatmap: bool = True,
        gaussian_sigma: float = 4.0,
    ) -> Dict[str, Union[float, bool, np.ndarray, list]]:
        """
        Predicts anomaly score and generates pixel-level anomaly heatmap.
        """
        orig_h, orig_w = frame.shape[:2]
        tensor = self._preprocess_frame(frame).to(self.device)

        self.core.eval()
        with torch.no_grad():
            res = self.core.predict(tensor)
            if len(res) == 7:
                patch_raw_score, patch_score, fused_mean_score, fused_score, avg_patch_map, seg, avg_global_dist = res
            else:
                patch_raw_score, patch_score, fused_mean_score, fused_score, patch_dist, seg = res

        anomaly_map = np.squeeze(np.asarray(seg, dtype=np.float32))

        # Gaussian smoothing
        if gaussian_sigma > 0:
            anomaly_map = cv2.GaussianBlur(anomaly_map, (0, 0), sigmaX=gaussian_sigma)

        # Resize anomaly map to original resolution
        anomaly_map_resized = cv2.resize(
            anomaly_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )

        # Image-level anomaly score
        img_score = float(fused_score.item())

        img_thresh = (
            float(self.core.calibrated_image_threshold.item())
            if hasattr(self.core, "calibrated_image_threshold")
            else self.anomaly_threshold
        )
        pix_thresh = (
            float(self.core.calibrated_pixel_threshold.item())
            if hasattr(self.core, "calibrated_pixel_threshold")
            else 4.0
        )

        is_anomaly = bool(img_score > img_thresh)
        binary_mask = np.squeeze(
            (anomaly_map_resized > pix_thresh).astype(np.uint8) * 255
        )

        # Extract defect bounding boxes
        bboxes = []
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # Ignore tiny noise
                x, y, w, h = cv2.boundingRect(cnt)
                bboxes.append([x, y, x + w, y + h])

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": img_score,
            "fused_score": img_score,
            "patch_score": float(patch_score.item()),
            "image_threshold": img_thresh,
            "calibrated_image_threshold": img_thresh,
            "calibrated_pixel_threshold": pix_thresh,
            "anomaly_map": anomaly_map_resized,
            "anomaly_mask": binary_mask,
            "bboxes": bboxes,
        }

    def generate_overlay(
        self,
        frame: np.ndarray,
        anomaly_map: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
        vmax: float = 4.0,
    ) -> np.ndarray:
        """
        Overlays heatmap on original frame.
        """
        norm_map = np.clip((anomaly_map - 0.0) / (vmax + 1e-8), 0.0, 1.0)
        heatmap_u8 = (norm_map * 255.0).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap_u8, colormap)
        overlay = cv2.addWeighted(frame, 1.0 - alpha, colored_heatmap, alpha, 0)
        return overlay

    def save(self, save_path: str):
        """
        Saves FastAD model checkpoint.
        """
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        checkpoint = {
            "backbone": self.backbone_name,
            "layer_to_extract": self.layer_to_extract,
            "input_size": self.input_size,
            "patchsize": self.patchsize,
            "stride": self.stride,
            "n_quantiles": self.n_quantiles,
            "fm_quantiles": self.fm_quantiles,
            "min_q": self.min_q,
            "max_q": self.max_q,
            "patch_covariance_method": self.patch_covariance_method,
            "global_covariance_method": self.global_covariance_method,
            "regularization": self.regularization,
            "anomaly_threshold": self.anomaly_threshold,
            "fastad_state": self.core.get_model_state(),
        }
        torch.save(checkpoint, save_path)
        print(f"[FastAD] Saved model checkpoint to: {save_path}")

    @classmethod
    def load(cls, checkpoint_path: str, device: Optional[str] = None):
        """
        Loads FastAD model from checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = cls(
            backbone=checkpoint.get("backbone", "wide_resnet50_2"),
            layer_to_extract=checkpoint.get("layer_to_extract", 3),
            input_size=checkpoint.get("input_size", (320, 320)),
            patchsize=checkpoint.get("patchsize", 3),
            stride=checkpoint.get("stride", 1),
            n_quantiles=checkpoint.get("n_quantiles", 4),
            fm_quantiles=checkpoint.get("fm_quantiles", 4),
            min_q=checkpoint.get("min_q", 0.05),
            max_q=checkpoint.get("max_q", 0.95),
            patch_covariance_method=checkpoint.get(
                "patch_covariance_method", "diagonal"
            ),
            global_covariance_method=checkpoint.get(
                "global_covariance_method", "diagonal"
            ),
            regularization=checkpoint.get("regularization", 1e-4),
            anomaly_threshold=checkpoint.get("anomaly_threshold", 2.0),
            device=device,
        )
        if "fastad_state" in checkpoint:
            model.core.load_model_state(checkpoint["fastad_state"])
        model.core.eval()
        return model
