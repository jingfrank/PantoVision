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
        fm_min_q,
        fm_max_q,
        patch_covariance_method:str,
        global_covariance_method:str,
        regularization,
        pretrained_weights_path,
        engine_path=None,
        prefer_tensorrt=True):
        super(FastAD, self).__init__()
        self.backbone = backbone
        self.layer_to_extract = layer_to_extract
        if not isinstance(self.layer_to_extract, list):
            self.layer_to_extract = [self.layer_to_extract]
        self.device = device
        self.patchsize = patchsize
        self.stride = stride
        self.target_size = target_size
        self.quantiles = n_quantiles
        self.fm_quantiles = fm_quantiles
        self.min_q = min_q
        self.max_q = max_q
        self.fm_min_q = fm_min_q
        self.fm_max_q = fm_max_q
        self.patch_covariance_method = patch_covariance_method
        self.global_covariance_method = global_covariance_method
        self.regularization = regularization
        self.pretrained_weights_path = pretrained_weights_path
        self.engine_path = engine_path
        self.prefer_tensorrt = prefer_tensorrt
        self.feature_extractor = FeatureExtractor(
            backbone,
            self.layer_to_extract,
            device,
            pretrained_weights_path,
            use_fp16=True,
            engine_path=self.engine_path,
            prefer_tensorrt=self.prefer_tensorrt,
        )
        self.patch_extractor = PatchExtractor(patchsize, stride)
        
        self.global_estimators = nn.ModuleList()
        self.global_mean_estimators = nn.ModuleList()
        
        for i in range(len(self.layer_to_extract)):
            self.global_estimators.append(GaussianEstimator(self.global_covariance_method))
            self.global_mean_estimators.append(GaussianEstimator(self.global_covariance_method))
            
            self.register_buffer(f"patch_mean_{i}", torch.empty(0), persistent=True)
            self.register_buffer(f"patch_inv_covariance_{i}", torch.empty(0), persistent=True)
            self.register_buffer(f"patch_raw_mean_{i}", torch.empty(0), persistent=True)
            self.register_buffer(f"patch_raw_inv_covariance_{i}", torch.empty(0), persistent=True)

            self.register_buffer(f"global_score_mean_{i}", torch.empty(0))
            self.register_buffer(f"global_score_std_{i}", torch.empty(0))
            self.register_buffer(f"global_mean_score_mean_{i}", torch.empty(0))
            self.register_buffer(f"global_mean_score_std_{i}", torch.empty(0))
            self.register_buffer(f"patch_score_mean_{i}", torch.empty(0))
            self.register_buffer(f"patch_score_std_{i}", torch.empty(0))
            self.register_buffer(f"patch_raw_score_mean_{i}", torch.empty(0))
            self.register_buffer(f"patch_raw_score_std_{i}", torch.empty(0))

        self.register_buffer("calibrated_image_threshold", torch.tensor(15.0))
        self.register_buffer("calibrated_pixel_threshold", torch.tensor(4.0))
        self.anomaly_segmentor = RescaleSegmentor(device=self.device,target_size = self.target_size)

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
        if isinstance(feature_maps, list):
            patches_list = []
            num_patches_list = []
            for fmap in feature_maps:
                p, np = self.patch_extractor(fmap, return_spatial_info=True)
                patches_list.append(p)
                num_patches_list.append(np)
            
            if provide_patch_shapes:
                return feature_maps, patches_list, num_patches_list
            return feature_maps, patches_list
        else:
            patches,num_patches = self.patch_extractor(feature_maps,return_spatial_info = True)
            if provide_patch_shapes:
                return feature_maps, patches, num_patches
            return feature_maps,patches

    def fit(self,train_dataloader):
        with tqdm.tqdm(train_dataloader,desc = "Extracting features and Gauss",position = 1,leave=False) as data_iterator:
            patch_raw_estimators = [OnlineGaussianEstimator(self.patch_covariance_method) for _ in self.layer_to_extract]
            patch_estimators = [OnlineGaussianEstimator(self.patch_covariance_method) for _ in self.layer_to_extract]
            feature_vectors_list = [[] for _ in self.layer_to_extract]
            feature_map_cdfs_list = [[] for _ in self.layer_to_extract]

            for batch in data_iterator:                
                images = batch['image'].to(self.device)
                
                with torch.no_grad():
                    feature_maps_list, patches_list = self.embed(images)
                
                if not isinstance(feature_maps_list, list):
                    feature_maps_list = [feature_maps_list]
                    patches_list = [patches_list]

                del images
                
                for i, (feature_maps, patches) in enumerate(zip(feature_maps_list, patches_list)):
                    patch_cdf = compute_cdf_and_binary_features(patches,n_quantiles = self.quantiles,min_q=self.min_q, max_q=self.max_q)
                    patch_estimators[i].update(patch_cdf)

                    patches_flat = patches.reshape(patches.shape[0], -1)  # [B*N, C*P*P]
                    patch_raw_estimators[i].update(patches_flat)
                    del patches, patch_cdf, patches_flat
                
                    # Process global features
                    feature_maps_reshaped = feature_maps.reshape(
                        feature_maps.shape[0], feature_maps.shape[1], -1
                    )
                    feature_map_cdf = compute_cdf_and_binary_features(
                        feature_maps_reshaped, n_quantiles=self.fm_quantiles,min_q=self.fm_min_q, max_q=self.fm_max_q
                    )
                    feature_map_cdfs_list[i].append(feature_map_cdf)

                    feature_vector = torch.mean(feature_maps, dim=(2, 3))
                    feature_vectors_list[i].append(feature_vector)
                    del feature_maps, feature_maps_reshaped, feature_map_cdf, feature_vector
                torch.cuda.empty_cache()
        
        for i in range(len(self.layer_to_extract)):
            mean, inv_cov = patch_estimators[i].finalize(self.regularization)
            setattr(self, f"patch_mean_{i}", mean)
            setattr(self, f"patch_inv_covariance_{i}", inv_cov)

            raw_mean, raw_inv_cov = patch_raw_estimators[i].finalize(self.regularization)
            setattr(self, f"patch_raw_mean_{i}", raw_mean)
            setattr(self, f"patch_raw_inv_covariance_{i}", raw_inv_cov)

            feature_map_cdfs = torch.cat(feature_map_cdfs_list[i], dim=0) 
            self.global_estimators[i].fit(feature_map_cdfs)

            feature_vectors = torch.cat(feature_vectors_list[i], dim=0)
            self.global_mean_estimators[i].fit(feature_vectors)
        return
        
    def validate(self,val_dataloader):
        global_scores_val_list = [[] for _ in self.layer_to_extract]
        global_mean_scores_val_list = [[] for _ in self.layer_to_extract]
        patch_scores_val_list = [[] for _ in self.layer_to_extract]
        patch_raw_scores_val_list = [[] for _ in self.layer_to_extract]

        with tqdm.tqdm(val_dataloader, desc="Validating...", leave=False) as data_iterator:
            for batch in data_iterator:
                images = batch["image"]
                images = images.to(self.device)
                batch_size = images.shape[0]
                with torch.no_grad():
                    feature_maps_list, patches_list = self.embed(images)
                
                if not isinstance(feature_maps_list, list):
                    feature_maps_list = [feature_maps_list]
                    patches_list = [patches_list]

                for i, (feature_maps, patches) in enumerate(zip(feature_maps_list, patches_list)):
                    feature_maps_reshaped = feature_maps.reshape(feature_maps.shape[0], feature_maps.shape[1], -1)
                    feature_map_cdf =compute_cdf_and_binary_features(feature_maps_reshaped, n_quantiles=self.fm_quantiles,min_q=self.fm_min_q, max_q=self.fm_max_q)
                    global_score = self.global_estimators[i].compute_distance(feature_map_cdf)
                        
                    feature_vector = torch.mean(feature_maps, dim=(2, 3))
                    global_mean_score = self.global_mean_estimators[i].compute_distance(feature_vector)

                    global_scores_val_list[i].append(global_score)
                    global_mean_scores_val_list[i].append(global_mean_score)

                    batchsize = images.shape[0]
                    patch_cdf = compute_cdf_and_binary_features(patches,n_quantiles = self.quantiles,min_q=self.min_q, max_q=self.max_q)#[B*N,C*Q]
                    
                    patch_mean = getattr(self, f"patch_mean_{i}")
                    patch_inv_cov = getattr(self, f"patch_inv_covariance_{i}")
                    patch_score = compute_mahalanobis_distance_pytorch(patch_cdf,patch_mean,patch_inv_cov,device=self.device)
                    patch_score = patch_score.view(batch_size, -1)  #[B,N]              
                    patch_scores_val_list[i].append(patch_score)

                    patches_flat = patches.reshape(patches.shape[0], -1)  # [B*N, C*P*P]
                    
                    patch_raw_mean = getattr(self, f"patch_raw_mean_{i}")
                    patch_raw_inv_cov = getattr(self, f"patch_raw_inv_covariance_{i}")
                    patch_raw_score = compute_mahalanobis_distance_pytorch(patches_flat,patch_raw_mean,patch_raw_inv_cov,device=self.device)
                    patch_raw_score = patch_raw_score.view(batch_size, -1)  #[B,N]
                    patch_raw_scores_val_list[i].append(patch_raw_score)

        for i in range(len(self.layer_to_extract)):
            global_scores_val = torch.cat(global_scores_val_list[i],dim = 0) 
            global_mean_scores_val = torch.cat(global_mean_scores_val_list[i],dim = 0) 
            patch_scores_val = torch.cat(patch_scores_val_list[i],dim = 0) 
            patch_raw_scores_val = torch.cat(patch_raw_scores_val_list[i],dim = 0) 

            global_score_mean, global_score_std = self.fit_zscore_scaler_torch(global_scores_val)
            setattr(self, f"global_score_mean_{i}", global_score_mean)
            setattr(self, f"global_score_std_{i}", global_score_std)

            global_mean_score_mean, global_mean_score_std = self.fit_zscore_scaler_torch(global_mean_scores_val)
            setattr(self, f"global_mean_score_mean_{i}", global_mean_score_mean)
            setattr(self, f"global_mean_score_std_{i}", global_mean_score_std)

            patch_scores_flattened = patch_scores_val.flatten() 
            patch_score_mean, patch_score_std = self.fit_zscore_scaler_torch(patch_scores_flattened )
            setattr(self, f"patch_score_mean_{i}", patch_score_mean)
            setattr(self, f"patch_score_std_{i}", patch_score_std)

            patch_raw_scores_flattened = patch_raw_scores_val.flatten() 
            patch_raw_score_mean, patch_raw_score_std = self.fit_zscore_scaler_torch(patch_raw_scores_flattened )
            setattr(self, f"patch_raw_score_mean_{i}", patch_raw_score_mean)
            setattr(self, f"patch_raw_score_std_{i}", patch_raw_score_std)

        # Auto-calibrate thresholds on validation dataset
        val_fused_scores = []
        val_pixel_scores = []
        with torch.no_grad():
            for batch in val_dataloader:
                images = batch["image"].to(self.device)
                _, patch_score, _, fused_score, avg_patch_map, _, _ = self.predict(images)
                val_fused_scores.append(fused_score)
                val_pixel_scores.append(avg_patch_map.flatten())

        val_fused_all = torch.cat(val_fused_scores, dim=0)
        val_pixel_all = torch.cat(val_pixel_scores, dim=0)

        if val_pixel_all.numel() > 1_000_000:
            step = max(1, val_pixel_all.numel() // 500_000)
            sampled_pixels = val_pixel_all[::step]
        else:
            sampled_pixels = val_pixel_all

        self.calibrated_image_threshold = torch.tensor(float(torch.quantile(val_fused_all, 0.99).item()) * 1.15, device=self.device)
        self.calibrated_pixel_threshold = torch.tensor(float(torch.quantile(sampled_pixels, 0.999).item()), device=self.device)
        print(f"[FastAD3] Auto-calibrated normal baseline thresholds: image={self.calibrated_image_threshold.item():.2f}, pixel={self.calibrated_pixel_threshold.item():.2f}")
        return 
        

    def predict(self,images, labels=None, plotter=None):
        images = images.to(torch.float).to(self.device)
        batch_size = images.shape[0] 
        with torch.no_grad():
            feature_maps_list, patches_list, num_patches_list = self.embed(images,provide_patch_shapes=True) 
            
            if not isinstance(feature_maps_list, list):
                feature_maps_list = [feature_maps_list]
                patches_list = [patches_list]
                num_patches_list = [num_patches_list]

        total_global_distance = 0
        total_global_mean_distance = 0
        total_patch_distance_map = 0
        total_patch_raw_score = 0
        
        for i, (feature_maps, patches, num_patches) in enumerate(zip(feature_maps_list, patches_list, num_patches_list)):
            # Global
            feature_maps_reshaped = feature_maps.reshape(feature_maps.shape[0], feature_maps.shape[1], -1)
            feature_map_cdf = compute_cdf_and_binary_features(feature_maps_reshaped, n_quantiles=self.fm_quantiles,min_q=self.fm_min_q, max_q=self.fm_max_q, labels=labels, name="Global", items_per_label=1, plotter=plotter)            
            global_distance = self.global_estimators[i].compute_distance(feature_map_cdf)
            global_distance = self.transform_zscore_scaler_torch(global_distance, getattr(self, f"global_score_mean_{i}"), getattr(self, f"global_score_std_{i}"))
            total_global_distance += global_distance
            
            feature_vector = torch.mean(feature_maps, dim=(2, 3))
            global_mean_distance = self.global_mean_estimators[i].compute_distance(feature_vector)
            global_mean_distance =  self.transform_zscore_scaler_torch(global_mean_distance, getattr(self, f"global_mean_score_mean_{i}"), getattr(self, f"global_mean_score_std_{i}"))
            total_global_mean_distance += global_mean_distance

            # Patch
            # Calculate items_per_label for patches
            # patches shape: [B*N, C, P, P]
            # labels length: B
            # items_per_label = N = patches.shape[0] // len(labels) if labels else 1
            items_per_label = patches.shape[0] // len(labels) if labels else 1
            
            patch_cdf = compute_cdf_and_binary_features(patches,n_quantiles = self.quantiles,min_q=self.min_q, max_q=self.max_q, labels=labels, name="Patch", items_per_label=items_per_label, plotter=plotter)
            patch_mean = getattr(self, f"patch_mean_{i}")
            patch_inv_cov = getattr(self, f"patch_inv_covariance_{i}")

            patch_distance = compute_mahalanobis_distance_pytorch(patch_cdf, patch_mean, patch_inv_cov, device=self.device)
            patch_distance = self.transform_zscore_scaler_torch(
                                                                patch_distance,
                                                                getattr(self, f"patch_score_mean_{i}"),
                                                                getattr(self, f"patch_score_std_{i}")
                                                                )
            
            patch_distance_map = patch_distance.reshape(batch_size, num_patches[0], num_patches[1])
            
            # Interpolate to target size
            patch_distance_map = patch_distance_map.unsqueeze(1)
            patch_distance_map = F.interpolate(
                patch_distance_map, size=self.target_size, mode="bilinear", align_corners=False
            )
            patch_distance_map = patch_distance_map.squeeze(1)
            
            if i == 0:
                total_patch_distance_map = patch_distance_map
            else:
                total_patch_distance_map += patch_distance_map

            # Patch Raw
            patches_flat = patches.reshape(patches.shape[0], -1)
            patch_raw_mean = getattr(self, f"patch_raw_mean_{i}")
            patch_raw_inv_cov = getattr(self, f"patch_raw_inv_covariance_{i}")
            patch_raw_distance = compute_mahalanobis_distance_pytorch(patches_flat, patch_raw_mean, patch_raw_inv_cov, device=self.device)
            patch_raw_distance = self.transform_zscore_scaler_torch(
                                                                patch_raw_distance,
                                                                getattr(self, f"patch_raw_score_mean_{i}"),
                                                                getattr(self, f"patch_raw_score_std_{i}")
                                                                )
            patch_raw_score_layer = torch.amax(patch_raw_distance.view(batch_size, -1), dim=1)
            total_patch_raw_score += patch_raw_score_layer

        avg_global_distance = total_global_distance / len(self.layer_to_extract)
        avg_global_mean_distance = total_global_mean_distance / len(self.layer_to_extract)
        avg_patch_distance_map = total_patch_distance_map / len(self.layer_to_extract)
        patch_raw_score = total_patch_raw_score / len(self.layer_to_extract)

        segmentation = self.anomaly_segmentor.convert_to_segmentation(avg_patch_distance_map)
        patch_score = torch.amax(avg_patch_distance_map.view(batch_size, -1), dim=1)

        fused_score = 0.5 * avg_global_distance + 0.5 * patch_score
        fused_mean_score = 0.5 * avg_global_mean_distance + 0.5 * patch_score
        return patch_raw_score, patch_score, fused_mean_score, fused_score, avg_patch_distance_map, segmentation, avg_global_distance
        
    def _predict(self,images):
        """Lightweight predict for latency benchmarking."""
        images = images.to(torch.float).to(self.device)
        batch_size = images.shape[0]
        with torch.no_grad():
            feature_maps_list, patches_list, num_patches_list = self.embed(images, provide_patch_shapes=True)

            if not isinstance(feature_maps_list, list):
                feature_maps_list = [feature_maps_list]
                patches_list = [patches_list]
                num_patches_list = [num_patches_list]

        total_global_distance = 0
        total_patch_score = 0

        for i, (feature_maps, patches, num_patches) in enumerate(zip(feature_maps_list, patches_list, num_patches_list)):
            # Global distance
            feature_maps_reshaped = feature_maps.reshape(feature_maps.shape[0], feature_maps.shape[1], -1)
            feature_map_cdf = compute_cdf_and_binary_features(feature_maps_reshaped, n_quantiles=self.fm_quantiles, min_q=self.fm_min_q, max_q=self.fm_max_q)
            global_distance = self.global_estimators[i].compute_distance(feature_map_cdf)
            global_distance = self.transform_zscore_scaler_torch(global_distance, getattr(self, f"global_score_mean_{i}"), getattr(self, f"global_score_std_{i}"))
            total_global_distance += global_distance

            # Patch distance
            patch_cdf = compute_cdf_and_binary_features(patches, n_quantiles=self.quantiles, min_q=self.min_q, max_q=self.max_q)
            patch_mean = getattr(self, f"patch_mean_{i}")
            patch_inv_cov = getattr(self, f"patch_inv_covariance_{i}")
            patch_distance = compute_mahalanobis_distance_pytorch(patch_cdf, patch_mean, patch_inv_cov, device=self.device)
            patch_distance = self.transform_zscore_scaler_torch(
                patch_distance,
                getattr(self, f"patch_score_mean_{i}"),
                getattr(self, f"patch_score_std_{i}")
            )
            patch_score = torch.amax(patch_distance.view(batch_size, -1), dim=1)
            total_patch_score += patch_score

        avg_global_distance = total_global_distance / len(self.layer_to_extract)
        avg_patch_score = total_patch_score / len(self.layer_to_extract)

        # Fused score
        fused_score = 0.5 * avg_global_distance + 0.5 * avg_patch_score
        return fused_score
        
    def save_model(self,save_path):

        # 保存模型状态和配置
        checkpoint = {}
        for i in range(len(self.layer_to_extract)):
            checkpoint[f'patch_mean_{i}'] = getattr(self, f'patch_mean_{i}')
            checkpoint[f'patch_inv_covariance_{i}'] = getattr(self, f'patch_inv_covariance_{i}')
            checkpoint[f'patch_raw_mean_{i}'] = getattr(self, f'patch_raw_mean_{i}')
            checkpoint[f'patch_raw_inv_covariance_{i}'] = getattr(self, f'patch_raw_inv_covariance_{i}')
            
            checkpoint[f'global_estimator_mean_{i}'] = self.global_estimators[i].mean
            checkpoint[f'global_estimator_inv_cov_{i}'] = self.global_estimators[i].inv_cov
            checkpoint[f'global_mean_estimator_mean_{i}'] = self.global_mean_estimators[i].mean
            checkpoint[f'global_mean_estimator_inv_cov_{i}'] = self.global_mean_estimators[i].inv_cov
            
            checkpoint[f'global_score_mean_{i}'] = getattr(self, f'global_score_mean_{i}')
            checkpoint[f'global_score_std_{i}'] = getattr(self, f'global_score_std_{i}')
            checkpoint[f'global_mean_score_mean_{i}'] = getattr(self, f'global_mean_score_mean_{i}')
            checkpoint[f'global_mean_score_std_{i}'] = getattr(self, f'global_mean_score_std_{i}')
            checkpoint[f'patch_score_mean_{i}'] = getattr(self, f'patch_score_mean_{i}')
            checkpoint[f'patch_score_std_{i}'] = getattr(self, f'patch_score_std_{i}')
            checkpoint[f'patch_raw_score_mean_{i}'] = getattr(self, f'patch_raw_score_mean_{i}')
            checkpoint[f'patch_raw_score_std_{i}'] = getattr(self, f'patch_raw_score_std_{i}')

        checkpoint['model_config'] = {
                'backbone': self.backbone,
                'layer_to_extract': self.layer_to_extract,
                'device': self.device,
                'patchsize': self.patchsize,
                'stride': self.stride,
                'target_size': self.target_size,
                'n_quantiles': self.quantiles,
                'fm_quantiles': self.fm_quantiles,
                'min_q': self.min_q,
                'max_q': self.max_q,
                'fm_min_q': self.fm_min_q,
                'fm_max_q': self.fm_max_q,
                'patch_covariance_method': self.patch_covariance_method,
                'global_covariance_method': self.global_covariance_method,
                'regularization': self.regularization,
                'pretrained_weights_path': self.pretrained_weights_path
        }
        torch.save(checkpoint, save_path)
        
        return 
    @classmethod
    def load_model(cls, load_path,device):
        """加载模型参数"""
        model_state = torch.load(load_path, map_location=device)
        
        config = model_state['model_config']
    
        # 使用保存的配置创建模型实例
        model = cls(
            backbone=config['backbone'],
            layer_to_extract=config['layer_to_extract'],
            device=device,
            patchsize=config['patchsize'],
            stride=config['stride'],
            target_size=config['target_size'],
            n_quantiles=config['n_quantiles'],
            fm_quantiles=config['fm_quantiles'],
            min_q=config['min_q'],
            max_q=config['max_q'],
            fm_min_q=config.get('fm_min_q', 0.01),
            fm_max_q=config.get('fm_max_q', 0.99),
            patch_covariance_method=config['patch_covariance_method'],
            global_covariance_method=config['global_covariance_method'],
            regularization=config['regularization'],
            pretrained_weights_path=config['pretrained_weights_path']
        )
        
        # 加载训练好的参数
        for i in range(len(model.layer_to_extract)):
            setattr(model, f'patch_mean_{i}', model_state[f'patch_mean_{i}'])
            setattr(model, f'patch_inv_covariance_{i}', model_state[f'patch_inv_covariance_{i}'])
            setattr(model, f'patch_raw_mean_{i}', model_state[f'patch_raw_mean_{i}'])
            setattr(model, f'patch_raw_inv_covariance_{i}', model_state[f'patch_raw_inv_covariance_{i}'])
            
            model.global_estimators[i].mean = model_state[f'global_estimator_mean_{i}']
            model.global_estimators[i].inv_cov = model_state[f'global_estimator_inv_cov_{i}']
            model.global_mean_estimators[i].mean = model_state[f'global_mean_estimator_mean_{i}']
            model.global_mean_estimators[i].inv_cov = model_state[f'global_mean_estimator_inv_cov_{i}']
            
            setattr(model, f'global_score_mean_{i}', model_state[f'global_score_mean_{i}'])
            setattr(model, f'global_score_std_{i}', model_state[f'global_score_std_{i}'])
            setattr(model, f'global_mean_score_mean_{i}', model_state[f'global_mean_score_mean_{i}'])
            setattr(model, f'global_mean_score_std_{i}', model_state[f'global_mean_score_std_{i}'])
            setattr(model, f'patch_score_mean_{i}', model_state[f'patch_score_mean_{i}'])
            setattr(model, f'patch_score_std_{i}', model_state[f'patch_score_std_{i}'])
            setattr(model, f'patch_raw_score_mean_{i}', model_state[f'patch_raw_score_mean_{i}'])
            setattr(model, f'patch_raw_score_std_{i}', model_state[f'patch_raw_score_std_{i}'])

        return model

    def get_model_state(self) -> dict:
        state = {
            "calibrated_image_threshold": getattr(self, "calibrated_image_threshold", torch.tensor(15.0)).cpu(),
            "calibrated_pixel_threshold": getattr(self, "calibrated_pixel_threshold", torch.tensor(4.0)).cpu(),
        }
        for i in range(len(self.layer_to_extract)):
            for attr in [
                f"patch_mean_{i}", f"patch_inv_covariance_{i}",
                f"patch_raw_mean_{i}", f"patch_raw_inv_covariance_{i}",
                f"global_score_mean_{i}", f"global_score_std_{i}",
                f"global_mean_score_mean_{i}", f"global_mean_score_std_{i}",
                f"patch_score_mean_{i}", f"patch_score_std_{i}",
                f"patch_raw_score_mean_{i}", f"patch_raw_score_std_{i}",
            ]:
                val = getattr(self, attr, None)
                if val is not None and isinstance(val, torch.Tensor):
                    state[attr] = val.cpu()
            if hasattr(self.global_estimators[i], "mean") and self.global_estimators[i].mean is not None:
                state[f"global_estimator_mean_{i}"] = self.global_estimators[i].mean.cpu()
            if hasattr(self.global_estimators[i], "inv_cov") and self.global_estimators[i].inv_cov is not None:
                state[f"global_estimator_inv_cov_{i}"] = self.global_estimators[i].inv_cov.cpu()
            if hasattr(self.global_mean_estimators[i], "mean") and self.global_mean_estimators[i].mean is not None:
                state[f"global_mean_estimator_mean_{i}"] = self.global_mean_estimators[i].mean.cpu()
            if hasattr(self.global_mean_estimators[i], "inv_cov") and self.global_mean_estimators[i].inv_cov is not None:
                state[f"global_mean_estimator_inv_cov_{i}"] = self.global_mean_estimators[i].inv_cov.cpu()
        return state

    def load_model_state(self, state: dict):
        dev = self.device
        if "calibrated_image_threshold" in state:
            self.calibrated_image_threshold = state["calibrated_image_threshold"].to(dev)
        if "calibrated_pixel_threshold" in state:
            self.calibrated_pixel_threshold = state["calibrated_pixel_threshold"].to(dev)
        for i in range(len(self.layer_to_extract)):
            for attr in [
                f"patch_mean_{i}", f"patch_inv_covariance_{i}",
                f"patch_raw_mean_{i}", f"patch_raw_inv_covariance_{i}",
                f"global_score_mean_{i}", f"global_score_std_{i}",
                f"global_mean_score_mean_{i}", f"global_mean_score_std_{i}",
                f"patch_score_mean_{i}", f"patch_score_std_{i}",
                f"patch_raw_score_mean_{i}", f"patch_raw_score_std_{i}",
            ]:
                if attr in state and state[attr] is not None:
                    setattr(self, attr, state[attr].to(dev))
            if f"global_estimator_mean_{i}" in state:
                self.global_estimators[i].mean = state[f"global_estimator_mean_{i}"].to(dev)
            if f"global_estimator_inv_cov_{i}" in state:
                self.global_estimators[i].inv_cov = state[f"global_estimator_inv_cov_{i}"].to(dev)
            if f"global_mean_estimator_mean_{i}" in state:
                self.global_mean_estimators[i].mean = state[f"global_mean_estimator_mean_{i}"].to(dev)
            if f"global_mean_estimator_inv_cov_{i}" in state:
                self.global_mean_estimators[i].inv_cov = state[f"global_mean_estimator_inv_cov_{i}"].to(dev)