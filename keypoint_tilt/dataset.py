"""
Dataset and Heatmap Generation for Pantograph Keypoint Localization (4-Keypoint Scheme).
Based on Paper Fig. 3: P1(left_horn), P2(left_support), P3(right_support), P4(right_horn).
"""

import json
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

PANTOGRAPH_KEYPOINTS = [
    "p1_left_horn",       # 0: 左羊角顶点 (Left Horn Vertex)
    "p2_left_support",    # 1: 左支撑臂交汇点 (Left Support Arm Joint)
    "p3_right_support",   # 2: 右支撑臂交汇点 (Right Support Arm Joint)
    "p4_right_horn"       # 3: 右羊角顶点 (Right Horn Vertex)
]

KEYPOINT_FLIP_PAIRS = [
    (0, 3),  # p1_left_horn <-> p4_right_horn
    (1, 2),  # p2_left_support <-> p3_right_support
]

SKELETON_PAIRS = [
    (0, 1),  # 左羊角到左支撑
    (1, 2),  # 左右支撑横梁 (支撑跨度)
    (2, 3),  # 右支撑到右羊角
    (0, 3),  # 外侧总跨度基线 (羊角顶点跨度)
]


def generate_gaussian_heatmap(h, w, kpts, sigma=2.0):
    num_kpts = len(kpts)
    target = np.zeros((num_kpts, h, w), dtype=np.float32)
    radius = int(sigma * 3)

    for i in range(num_kpts):
        x, y, v = kpts[i]
        if v == 0:
            continue
        
        mu_x = int(round(x))
        mu_y = int(round(y))

        ul = [mu_x - radius, mu_y - radius]
        br = [mu_x + radius + 1, mu_y + radius + 1]
        if ul[0] >= w or ul[1] >= h or br[0] <= 0 or br[1] <= 0:
            continue

        g_x = np.arange(0, 2 * radius + 1, 1, np.float32)
        g_y = np.arange(0, 2 * radius + 1, 1, np.float32)
        g_y = g_y[:, np.newaxis]
        g = np.exp(-((g_x - radius) ** 2 + (g_y - radius) ** 2) / (2 * sigma ** 2))

        g_x1 = max(0, -ul[0])
        g_x2 = min(br[0], w) - ul[0]
        g_y1 = max(0, -ul[1])
        g_y2 = min(br[1], h) - ul[1]

        img_x1 = max(0, ul[0])
        img_x2 = min(br[0], w)
        img_y1 = max(0, ul[1])
        img_y2 = min(br[1], h)

        target[i, img_y1:img_y2, img_x1:img_x2] = np.maximum(
            target[i, img_y1:img_y2, img_x1:img_x2],
            g[g_y1:g_y2, g_x1:g_x2]
        )

    return target


class PantographKeypointDataset(Dataset):
    def __init__(
        self,
        anno_file,
        img_dir,
        target_size=(288, 384),
        heatmap_size=(72, 96),
        sigma=2.0,
        is_train=True,
    ):
        self.img_dir = img_dir
        self.target_size = target_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.is_train = is_train

        if os.path.exists(anno_file):
            with open(anno_file, "r") as f:
                self.data = json.load(f)
            self.images = {img["id"]: img for img in self.data.get("images", [])}
            self.annotations = self.data.get("annotations", [])
        else:
            self.data = {}
            self.images = {}
            self.annotations = []

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        anno = self.annotations[idx]
        image_info = self.images.get(anno["image_id"], {})
        img_path = os.path.join(self.img_dir, image_info.get("file_name", ""))
        
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig, _ = img.shape

        raw_kpts = np.array(anno["keypoints"], dtype=np.float32).reshape(-1, 3)
        target_h, target_w = self.target_size
        hm_h, hm_w = self.heatmap_size

        scale_x = target_w / w_orig
        scale_y = target_h / h_orig

        img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        kpts_scaled = raw_kpts.copy()
        kpts_scaled[:, 0] *= scale_x
        kpts_scaled[:, 1] *= scale_y

        kpts_heatmap = raw_kpts.copy()
        kpts_heatmap[:, 0] *= (hm_w / w_orig)
        kpts_heatmap[:, 1] *= (hm_h / h_orig)

        heatmaps = generate_gaussian_heatmap(hm_h, hm_w, kpts_heatmap, sigma=self.sigma)

        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        heatmaps_tensor = torch.from_numpy(heatmaps).float()
        kpts_tensor = torch.from_numpy(kpts_scaled).float()

        return {
            "image": img_tensor,
            "target_heatmaps": heatmaps_tensor,
            "keypoints": kpts_tensor,
            "img_path": img_path,
            "orig_size": (h_orig, w_orig),
        }
