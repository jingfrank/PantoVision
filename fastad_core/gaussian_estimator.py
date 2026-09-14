import torch
import torch.nn as nn
from sklearn.covariance import LedoitWolf, ShrunkCovariance, EmpiricalCovariance

class GaussianEstimator(nn.Module):
    def __init__(self,covariance_method: str):
        super().__init__()
        self.covariance_method = covariance_method
        self.register_buffer("mean", torch.empty(0), persistent=True)
        self.register_buffer("inv_cov", torch.empty(0), persistent=True)
        # self.mean : torch.Tensor
        # self.inv_cov : torch.Tensor

    def fit(self,features:torch.Tensor):
        input_dtype = features.dtype  # 记录输入数据类型
        if features.dtype == torch.float16:
            features = features.float()
        self.mean = torch.mean(features, dim=0)

        x = features.cpu().numpy()
        if self.covariance_method == "LedoitWolf":
            estimator = LedoitWolf().fit(x)
            cov = torch.from_numpy(estimator.covariance_).to(device=features.device, dtype=features.dtype)
        elif self.covariance_method == "ShrunkCovariance":
            estimator = ShrunkCovariance(shrinkage=0.1).fit(x)
            cov = torch.from_numpy(estimator.covariance_).to(device=features.device, dtype=features.dtype)
        elif self.covariance_method == "EmpiricalCovariance":
            estimator = EmpiricalCovariance().fit(x)
            cov = torch.from_numpy(estimator.covariance_).to(device=features.device, dtype=features.dtype)
        elif self.covariance_method == "diagonal":
            var = torch.var(features, dim=0, unbiased=True) + 1e-6
            self.inv_cov = torch.diag(1.0 / var)
            return
        else:
            estimator = LedoitWolf().fit(x)
            cov = torch.from_numpy(estimator.covariance_).to(device=features.device, dtype=features.dtype)

        cov += torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype) * 1e-6  # 添加小正则项
        self.inv_cov = torch.linalg.inv(cov)
        
    def compute_distance(self, features: torch.Tensor):
        """
        计算马氏距离
        :param features: [N, D] Tensor
        :return: [N] Tensor
        """
        diff = features - self.mean
        mahalanobis_squared = torch.einsum('bi,ij,bj->b', diff, self.inv_cov,diff)
        return torch.sqrt(torch.clamp(mahalanobis_squared, min=0)) 

    
