import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy.ndimage as ndimage
from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from typing import Tuple

def resize_patch_distance(patch_distance,height, width):
    patch_distance = patch_distance.unsqueeze(1)
    # padded = F.pad(patch_distance, (4, 4, 4, 4))
    resized = F.interpolate(patch_distance,
                            size=(height, width),
                            mode='bilinear',
                            align_corners=False)
    return resized.squeeze(1)
    
def get_pdn_small(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)
    )


def get_pdn_medium(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                  kernel_size=1)
    )
def compute_mahalanobis_distance_pytorch(feature_vector, feature_mean, feature_covinv, device='cuda'):
    """
    使用PyTorch矩阵乘法版本 (通常更快)
    """
    feature_vector = feature_vector
    # 中心化
    

    center_feature_vector = feature_vector - feature_mean
    # 在计算前统一类型
    center_feature_vector = center_feature_vector
    feature_covinv = feature_covinv
    # 矩阵乘法: (B, C) @ (C, C) = (B, C)
    # temp = torch.mm(center_feature_vector, feature_covinv)
    
    # # 逐元素乘法后按行求和: (B, C) * (B, C) -> (B,)
    # mahalanobis_squared = torch.sum(temp * center_feature_vector, dim=1)
    mahalanobis_squared = torch.einsum('bi,ij,bj->b', center_feature_vector, feature_covinv, center_feature_vector)

    # 开平方根
    mahalanobis_distance = torch.sqrt(torch.clamp(mahalanobis_squared, min=0))
    
    
    return mahalanobis_distance 

def gaussian_estimation(embeddings: torch.Tensor, 
                       covariance_method: str,
                       shrinkage: float = 0.1,):
    mean = torch.mean(embeddings, dim=0)  # [C]
    X_np = embeddings.cpu().numpy() if embeddings.is_cuda else embeddings.numpy()
        
    if covariance_method == "EmpiricalCovariance":
        estimator = EmpiricalCovariance().fit(X_np)
    elif covariance_method == "ShrunkCovariance":
        estimator = ShrunkCovariance(shrinkage=shrinkage).fit(X_np)
    elif covariance_method == "LedoitWolf":
        estimator = LedoitWolf().fit(X_np)
        print(f"Ledoit-Wolf optimal shrinkage: {estimator.shrinkage_:.4f}")
    cov = torch.from_numpy(estimator.covariance_).to(device = embeddings.device, dtype=embeddings.dtype)  # [C, C]
# 计算逆协方差矩阵（添加小正则项保证数值稳定性）
    cov += torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype) * 1e-6  # 添加小正则项
    cov_inv = torch.linalg.inv(cov)
    return mean, cov_inv
    
class OnlineGaussianEstimator:
    """Helper class for incremental mean and covariance estimation."""
    
    def __init__(self,covariance_method="full"):
        self.covariance_method = covariance_method
        self.mean = None
        self.M2 = None  # Sum of squared differences from mean
        self.count = 0
    
    def update(self, batch):
        input_dtype = batch.dtype  # 记录输入数据类型
        if batch.dtype == torch.float16:
            batch = batch.float()
        """Update statistics with a new batch of data."""
        batch = batch.view(batch.shape[0], -1)  # Flatten to (N, D)
        batch_count = batch.shape[0]
        batch_mean = torch.mean(batch, dim=0)
        
        if self.mean is None:
            self.mean = batch_mean.clone()
            if self.covariance_method == "diagonal":
                self.M2 = torch.sum((batch - self.mean) ** 2, dim=0)
            elif self.covariance_method == "full" or "shrunk":
                centered = batch - self.mean
                self.M2 = centered.T @ centered
            self.count = batch_count
        else:
            delta = batch_mean - self.mean
            new_count = self.count + batch_count
            self.mean += delta * batch_count / new_count
            if self.covariance_method == "diagonal":
                batch_M2 = torch.sum((batch - batch_mean) ** 2, dim=0)
                self.M2 += batch_M2 + delta**2 * self.count * batch_count / new_count
            elif self.covariance_method == "full" or "shrunk":
                centered = batch - self.mean
                batch_cov = centered.T @ centered
                self.M2 += batch_cov + torch.outer(delta, delta) * self.count * batch_count / new_count
            
            self.count = new_count
    
    def finalize(self, regularization=1e-2):
        """Compute final mean and inverse covariance."""
        if self.count < 2:
            raise ValueError("Need at least 2 samples for covariance estimation")
        cov = self.M2 / (self.count - 1)
        # Compute covariance matrix
        if self.covariance_method == "full":
            # Full covariance matrix
            
            cov += regularization * torch.eye(cov.shape[0], device=cov.device)
            inv_cov = torch.inverse(cov)
        elif self.covariance_method == "diagonal":
            
            cov += regularization  # Regularization
            cov = torch.diag(cov)
            inv_cov = torch.inverse(cov)
        elif self.covariance_method == "shrunk":
            cov = self._ledoit_wolf_shrinkage(cov, shrinkage = None, regularization = regularization)
            
            inv_cov = torch.inverse(cov)
        else:
            raise NotImplementedError("Online full covariance not implemented")
        # Compute inverse covariance with better numerical stability
        # try:
        #     # Use Cholesky decomposition for better numerical stability
        #     L = torch.linalg.cholesky(cov)
        #     inv_cov = torch.cholesky_inverse(L)
        # except RuntimeError:
        #     # Fallback to regular inverse if Cholesky fails
        #     try:
        #         inv_cov = torch.linalg.inv(cov)
        #     except RuntimeError as e:
        self.inv_cov = inv_cov
        return self.mean, self.inv_cov

    def _ledoit_wolf_shrinkage(self, sample_cov, shrinkage=None, regularization=1e-6):
        """Classic Ledoit-Wolf linear shrinkage (2004)"""
        d = sample_cov.shape[0]
        n = self.count
        
        # Target: diagonal with average variance
        target = torch.eye(d, device=sample_cov.device, dtype=sample_cov.dtype)
        target *= torch.trace(sample_cov) / d
        
        if shrinkage is not None:
            alpha = shrinkage
        else:
            # Estimate optimal shrinkage intensity
            # See Ledoit & Wolf (2004) Eq. (3.8) - simplified version
            diff = sample_cov - target
            phi = torch.sum(sample_cov ** 2) / n
            gamma = torch.sum(diff ** 2)
            alpha = max(0.0, min(1.0, phi / gamma)) if gamma > 0 else 0.0
        
        shrunk_cov = (1 - alpha) * sample_cov + alpha * target
        shrunk_cov += regularization * torch.eye(d, device=shrunk_cov.device, dtype=shrunk_cov.dtype)
        return shrunk_cov
    def compute_distance(self, features: torch.Tensor):
        """
        计算马氏距离
        :param features: [N, D] Tensor
        :return: [N] Tensor
        """
        diff = features - self.mean
        mahalanobis_squared = torch.einsum('bi,ij,bj->b', diff, self.inv_cov,diff)
        return torch.sqrt(torch.clamp(mahalanobis_squared, min=0)) 
        
class RescaleSegmentor():
    def __init__(self, device, target_size=256):
        self.device = device
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(self, patch_scores):
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            if _scores.ndim == 3:
                _scores = _scores.unsqueeze(1)  # (B, 1, H, W)
            elif _scores.ndim == 2:
                _scores = _scores.unsqueeze(0).unsqueeze(1)

            _scores = F.interpolate(
                _scores, size=self.target_size, mode="bilinear", align_corners=False
            )
            # GPU 2D Gaussian smoothing via separable conv2d
            sigma = float(self.smoothing)
            ksize = int(2 * round(2.5 * sigma) + 1)
            x = torch.arange(ksize, device=self.device, dtype=torch.float32) - ksize // 2
            k1d = torch.exp(-0.5 * (x / sigma) ** 2)
            k1d = (k1d / k1d.sum()).view(1, 1, 1, ksize)
            k2d = k1d.view(1, 1, ksize, 1)

            _scores = F.conv2d(_scores, k1d, padding=(0, ksize // 2))
            _scores = F.conv2d(_scores, k2d, padding=(ksize // 2, 0))
            _scores = _scores.squeeze()
            return _scores.cpu().numpy()