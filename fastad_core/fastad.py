import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from patch_extractor import PatchExtractor
from feature_extractor import FeatureExtractor
from fastad_common import *
from gaussian_estimator import GaussianEstimator
from set_features import compute_cdf_and_binary_features
class FastAD(nn.Module):
    def __init__(self,
        backbone, 
        layer_to_extract, 
        device, 
        patchsize, 
        stride, 
        target_size, 
        n_quantiles,
        fm_quantiles,
        min_q,
        max_q,
        patch_covariance_method:str,
        global_covariance_method:str,
        regularization,
        pretrained_weights_path):
        super(FastAD, self).__init__()
        self.backbone = backbone
        self.layer_to_extract = layer_to_extract
        self.device = device
        self.patchsize = patchsize
        self.stride = stride
        self.target_size = target_size
        self.quantiles = n_quantiles
        self.fm_quantiles = fm_quantiles
        self.min_q = min_q
        self.max_q = max_q
        self.patch_covariance_method = patch_covariance_method
        self.global_covariance_method = global_covariance_method
        self.regularization = regularization
        self.pretrained_weights_path = pretrained_weights_path
        self.feature_extractor = FeatureExtractor(backbone, layer_to_extract, device,pretrained_weights_path)
        self.patch_extractor = PatchExtractor(patchsize, stride)
        self.patch_estimator = OnlineGaussianEstimator(self.patch_covariance_method)
        self.patch_raw_estimator = OnlineGaussianEstimator(self.patch_covariance_method)
        self.global_estimator = GaussianEstimator(self.global_covariance_method)
        self.global_mean_estimator= GaussianEstimator(self.global_covariance_method)
        # self.scaler_global = MinMaxScaler() # 全局马氏距离的归一化器
        # self.scaler_patch = MinMaxScaler()  # Patch KNN距离的归一化器
        self.register_buffer("global_score_mean", torch.tensor(0.0))
        self.register_buffer("global_score_std", torch.tensor(1.0))
        self.register_buffer("global_mean_score_mean", torch.tensor(0.0))
        self.register_buffer("global_mean_score_std", torch.tensor(1.0))

        self.register_buffer("patch_score_mean", torch.tensor(0.0))
        self.register_buffer("patch_score_std", torch.tensor(1.0))
        self.register_buffer("patch_raw_score_mean", torch.tensor(0.0))
        self.register_buffer("patch_raw_score_std", torch.tensor(1.0))
        self.register_buffer("calibrated_image_threshold", torch.tensor(15.0))
        self.register_buffer("calibrated_pixel_threshold", torch.tensor(4.0))
        self.anomaly_segmentor = RescaleSegmentor(device=self.device,target_size = target_size)

    def fit_minmax_scaler_torch(self,data: torch.Tensor):
        """
        输入: data shape [N] or [N, 1]
        输出: min 和 max（作为标量 tensor）
        """
        data_min = torch.min(data)
        data_max = torch.max(data)
        return data_min, data_max


    def transform_minmax_scaler_torch(self,data: torch.Tensor, data_min: torch.Tensor, data_max: torch.Tensor, eps=1e-8):
        """
        data: shape [N] or [N, 1]
        data_min, data_max: 标量 Tensor（在 GPU 上）
        """
        scale = 1.0 / (data_max - data_min + eps)
        return (data - data_min) * scale
    def fit_zscore_scaler_torch(self,data: torch.Tensor):
        """
        计算Z-Score标准化的参数
        输入: data shape [N] or [N, 1] 
        输出: mean 和 std（作为标量 tensor）
        """
        data_mean = torch.mean(data)
        data_std = torch.std(data, unbiased=True)
        return data_mean, data_std


    def transform_zscore_scaler_torch(self,data: torch.Tensor, data_mean: torch.Tensor, data_std: torch.Tensor, eps=1e-8):
        """
        执行Z-Score标准化
        data: shape [N] or [N, 1]
        data_mean, data_std: 标量 Tensor（在 GPU 上）
        eps: 防止除零的小常数
        """
        return (data - data_mean) / (data_std + eps)


    def embed(self,images,provide_patch_shapes = False):
        feature_maps = self.feature_extractor(images)
        patches,num_patches = self.patch_extractor(feature_maps,return_spatial_info = True)
        if provide_patch_shapes:
            return feature_maps, patches, num_patches
        return feature_maps,patches

    def fit(self,train_dataloader):
        with tqdm.tqdm(train_dataloader,desc = "Extracting features and Gauss",position = 1,leave=False) as data_iterator:

            feature_vectors = []
            feature_map_cdfs = []
            for batch in data_iterator:                
                images = batch['image'].to(self.device)
                with torch.no_grad():
                    feature_maps, patches = self.embed(images)
                patches_flat = patches.reshape(patches.shape[0], -1)  # [B*N, C*P*P]
                self.patch_raw_estimator.update(patches_flat)
                patch_cdf = compute_cdf_and_binary_features(patches,n_quantiles = self.quantiles,min_q=self.min_q, max_q=self.max_q)
                self.patch_estimator.update(patch_cdf)
            
                # Process global features
                feature_maps_reshaped = feature_maps.reshape(
                    feature_maps.shape[0], feature_maps.shape[1], -1
                )
                feature_map_cdf = compute_cdf_and_binary_features(
                    feature_maps_reshaped, n_quantiles=self.fm_quantiles
                )
                feature_map_cdfs.append(feature_map_cdf)

                feature_vector = torch.mean(feature_maps, dim=(2, 3))
                feature_vectors.append(feature_vector)
               
        self.patch_estimator.finalize(self.regularization)

        self.patch_raw_estimator.finalize(self.regularization)

        feature_map_cdfs = torch.cat(feature_map_cdfs, dim=0) 
        self.global_estimator.fit(feature_map_cdfs)

        feature_vectors = torch.cat(feature_vectors, dim=0)
        self.global_mean_estimator.fit(feature_vectors)
        return
        
    def validate(self,val_dataloader):
        global_scores_val= []
        global_mean_scores_val = [] 
        patch_scores_val = []
        patch_raw_scores_val = []
        with tqdm.tqdm(val_dataloader, desc="Validating...", leave=False) as data_iterator:
            for batch in data_iterator:
                images = batch["image"]
                images = images.to(self.device)
                batch_size = images.shape[0]
                with torch.no_grad():
                    feature_maps, patches = self.embed(images)

                feature_maps_reshaped = feature_maps.reshape(feature_maps.shape[0], feature_maps.shape[1], -1)
                feature_map_cdf =compute_cdf_and_binary_features(feature_maps_reshaped, n_quantiles=self.fm_quantiles)
                global_score = self.global_estimator.compute_distance(feature_map_cdf)
                    
                feature_vector = torch.mean(feature_maps, dim=(2, 3))
                global_mean_score = self.global_mean_estimator.compute_distance(feature_vector)

                global_scores_val.append(global_score)
                global_mean_scores_val.append(global_mean_score)

                batchsize = images.shape[0]
                patches_flat = patches.reshape(patches.shape[0], -1)  # [B*N, C*P*P]
                patch_raw_score = self.patch_raw_estimator.compute_distance(patches_flat)
                patch_raw_score = patch_raw_score.view(batch_size, -1)  #[B,N]
                patch_raw_scores_val.append(patch_raw_score)

                patch_cdf = compute_cdf_and_binary_features(patches,n_quantiles = self.quantiles,min_q=self.min_q, max_q=self.max_q)#[B*N,C*Q]
                patch_score =self.patch_estimator.compute_distance(patch_cdf) 
                patch_score = patch_score.view(batch_size, -1)  #[B,N]              
                patch_scores_val.append(patch_score)

        global_scores_val = torch.cat(global_scores_val,dim = 0) # [num_val_samples]
        global_mean_scores_val = torch.cat(global_mean_scores_val,dim = 0) # [num_val_samples]

        patch_scores_val = torch.cat(patch_scores_val,dim = 0) #
        patch_raw_scores_val = torch.cat(patch_raw_scores_val,dim = 0) # [num_val_samples,N]


        global_score_mean, global_score_std = self.fit_zscore_scaler_torch(global_scores_val)
        self.global_score_mean = global_score_mean
        self.global_score_std = global_score_std

        global_mean_score_mean, global_mean_score_std = self.fit_zscore_scaler_torch(global_mean_scores_val)
        self.global_mean_score_mean = global_mean_score_mean

        patch_scores_flattened = patch_scores_val.flatten()  #展平后统一标准化（全局标准化）
        patch_score_mean, patch_score_std = self.fit_zscore_scaler_torch(patch_scores_flattened )
        self.patch_score_mean = patch_score_mean
        patch_raw_scores_flattened = patch_raw_scores_val.flatten()  #展平后统一标准化（全局标准化）
        patch_raw_score_mean, patch_raw_score_std = self.fit_zscore_scaler_torch(patch_raw_scores_flattened )
        self.patch_raw_score_mean = patch_raw_score_mean
        self.patch_raw_score_std = patch_raw_score_std

        # Auto-calibrate image and pixel thresholds on held-out normal validation set
        norm_global = self.transform_zscore_scaler_torch(global_scores_val, self.global_score_mean, self.global_score_std)
        norm_patch_val = self.transform_zscore_scaler_torch(patch_scores_val, self.patch_score_mean, self.patch_score_std)
        norm_patch_max = torch.amax(norm_patch_val, dim=1)
        val_fused_scores = 0.5 * norm_global + 0.5 * norm_patch_max

        self.calibrated_image_threshold = torch.tensor(float(torch.quantile(val_fused_scores, 0.99).item()) * 1.15, device=self.device)
        self.calibrated_pixel_threshold = torch.tensor(float(torch.quantile(norm_patch_val.flatten(), 0.999).item()), device=self.device)
        print(f"[FastAD] Auto-calibrated normal baseline thresholds: image={self.calibrated_image_threshold.item():.2f}, pixel={self.calibrated_pixel_threshold.item():.2f}")
        return 
        

    def predict(self,images):
        images = images.to(torch.float).to(self.device)
        batch_size = images.shape[0] 
        with torch.no_grad():
            feature_maps, patches,num_patches = self.embed(images,provide_patch_shapes=True) 

            #计算全局马氏距离
        feature_maps_reshaped = feature_maps.reshape(feature_maps.shape[0], feature_maps.shape[1], -1)
        feature_map_cdf = compute_cdf_and_binary_features(feature_maps_reshaped, n_quantiles=self.fm_quantiles)            
        global_distance = self.global_estimator.compute_distance(feature_map_cdf)
        global_distance = self.transform_zscore_scaler_torch(global_distance, self.global_score_mean, self.global_score_std)
        
        feature_vector = torch.mean(feature_maps, dim=(2, 3))
        global_mean_distance = self.global_mean_estimator.compute_distance(feature_vector)
        global_mean_distance =  self.transform_zscore_scaler_torch(global_mean_distance, self.global_mean_score_mean, self.global_mean_score_std)
            

        #计算patch马氏距离        
        patch_cdf = compute_cdf_and_binary_features(patches,n_quantiles = self.quantiles,min_q=self.min_q, max_q=self.max_q)
        patch_distance = self.patch_estimator.compute_distance(patch_cdf)
        patch_distance = self.transform_zscore_scaler_torch(
                                                            patch_distance,
                                                            self.patch_score_mean,
                                                            self.patch_score_std
                                                            )
        patches_flat = patches.reshape(patches.shape[0], -1)  # [B*N, C*P*P]
        patch_raw_distance = self.patch_raw_estimator.compute_distance(patches_flat)
        patch_raw_distance = self.transform_zscore_scaler_torch(
                                                            patch_raw_distance,
                                                            self.patch_raw_score_mean,
                                                            self.patch_raw_score_std
                                                            )
        patch_raw_score = torch.amax(patch_raw_distance.view(batch_size, -1), dim=1)

        patch_distance = patch_distance.reshape(batch_size, num_patches[0], num_patches[1])
       
        segmentation = self.anomaly_segmentor.convert_to_segmentation(patch_distance)
        patch_score = torch.amax(patch_distance.view(batch_size, -1), dim=1)

        # 计算融合后的分数
        fused_score = 0.5 * global_distance + 0.5 * patch_score
        fused_mean_score = 0.5 * global_mean_distance + 0.5 * patch_score
        return patch_raw_score,patch_score,fused_mean_score,fused_score,patch_distance,segmentation

    def get_model_state(self) -> dict:
        return {
            "patch_estimator_mean": self.patch_estimator.mean.cpu() if self.patch_estimator.mean is not None else None,
            "patch_estimator_inv_cov": self.patch_estimator.inv_cov.cpu() if getattr(self.patch_estimator, "inv_cov", None) is not None else None,
            "patch_raw_estimator_mean": self.patch_raw_estimator.mean.cpu() if self.patch_raw_estimator.mean is not None else None,
            "patch_raw_estimator_inv_cov": self.patch_raw_estimator.inv_cov.cpu() if getattr(self.patch_raw_estimator, "inv_cov", None) is not None else None,
            "global_estimator_mean": self.global_estimator.mean.cpu() if getattr(self.global_estimator, "mean", None) is not None else None,
            "global_estimator_inv_cov": self.global_estimator.inv_cov.cpu() if getattr(self.global_estimator, "inv_cov", None) is not None else None,
            "global_mean_estimator_mean": self.global_mean_estimator.mean.cpu() if getattr(self.global_mean_estimator, "mean", None) is not None else None,
            "global_mean_estimator_inv_cov": self.global_mean_estimator.inv_cov.cpu() if getattr(self.global_mean_estimator, "inv_cov", None) is not None else None,
            "global_score_mean": self.global_score_mean.cpu(),
            "global_score_std": self.global_score_std.cpu(),
            "global_mean_score_mean": self.global_mean_score_mean.cpu(),
            "global_mean_score_std": self.global_mean_score_std.cpu(),
            "patch_score_mean": self.patch_score_mean.cpu(),
            "patch_score_std": self.patch_score_std.cpu(),
            "patch_raw_score_mean": self.patch_raw_score_mean.cpu(),
            "patch_raw_score_std": self.patch_raw_score_std.cpu(),
            "calibrated_image_threshold": getattr(self, "calibrated_image_threshold", torch.tensor(15.0)).cpu(),
            "calibrated_pixel_threshold": getattr(self, "calibrated_pixel_threshold", torch.tensor(4.0)).cpu(),
        }

    def load_model_state(self, state: dict):
        dev = self.device
        if state.get("patch_estimator_mean") is not None:
            self.patch_estimator.mean = state["patch_estimator_mean"].to(dev)
        if state.get("patch_estimator_inv_cov") is not None:
            self.patch_estimator.inv_cov = state["patch_estimator_inv_cov"].to(dev)
        if state.get("patch_raw_estimator_mean") is not None:
            self.patch_raw_estimator.mean = state["patch_raw_estimator_mean"].to(dev)
        if state.get("patch_raw_estimator_inv_cov") is not None:
            self.patch_raw_estimator.inv_cov = state["patch_raw_estimator_inv_cov"].to(dev)
        if state.get("global_estimator_mean") is not None:
            self.global_estimator.mean = state["global_estimator_mean"].to(dev)
        if state.get("global_estimator_inv_cov") is not None:
            self.global_estimator.inv_cov = state["global_estimator_inv_cov"].to(dev)
        if state.get("global_mean_estimator_mean") is not None:
            self.global_mean_estimator.mean = state["global_mean_estimator_mean"].to(dev)
        if state.get("global_mean_estimator_inv_cov") is not None:
            self.global_mean_estimator.inv_cov = state["global_mean_estimator_inv_cov"].to(dev)
        for k in ["global_score_mean", "global_score_std", "global_mean_score_mean", "global_mean_score_std",
                  "patch_score_mean", "patch_score_std", "patch_raw_score_mean", "patch_raw_score_std",
                  "calibrated_image_threshold", "calibrated_pixel_threshold"]:
            if state.get(k) is not None:
                setattr(self, k, state[k].to(dev))