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
        self.feature_extractor = FeatureExtractor(backbone, layer_to_extract, device,pretrained_weights_path,use_fp16 = True)
        self.patch_extractor = PatchExtractor(patchsize, stride)
        self.global_estimator = GaussianEstimator(self.global_covariance_method)
        self.global_mean_estimator= GaussianEstimator(self.global_covariance_method)
        # 改为注册空的 buffer，使用 persistent=True（默认值）
        self.register_buffer("patch_mean", torch.empty(0), persistent=True)
        self.register_buffer("patch_inv_covariance", torch.empty(0), persistent=True)
        self.register_buffer("patch_raw_mean", torch.empty(0), persistent=True)
        self.register_buffer("patch_raw_inv_covariance", torch.empty(0), persistent=True)

        self.register_buffer("global_score_mean", torch.empty(0))
        self.register_buffer("global_score_std", torch.empty(0))
        self.register_buffer("global_mean_score_mean", torch.empty(0))
        self.register_buffer("global_mean_score_std", torch.empty(0))
        self.register_buffer("patch_score_mean", torch.empty(0))
        self.register_buffer("patch_score_std", torch.empty(0))
        self.register_buffer("patch_raw_score_mean", torch.empty(0))
        self.register_buffer("patch_raw_score_std", torch.empty(0))
        # self.patch_mean: torch.Tensor = None
        # self.patch_inv_covariance :torch.Tensor = None
        # self.patch_raw_mean: torch.Tensor = None
        # self.patch_raw_inv_covariance :torch.Tensor = None

        # self.scaler_global = MinMaxScaler() # 全局马氏距离的归一化器
        # self.scaler_patch = MinMaxScaler()  # Patch KNN距离的归一化器
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
        patches,num_patches = self.patch_extractor(feature_maps,return_spatial_info = True)
        if provide_patch_shapes:
            return feature_maps, patches, num_patches
        return feature_maps,patches

    def fit(self,train_dataloader):
        with tqdm.tqdm(train_dataloader,desc = "Extracting features and Gauss",position = 1,leave=False) as data_iterator:
            patch_raw_estimator = OnlineGaussianEstimator(self.patch_covariance_method)
            patch_estimator = OnlineGaussianEstimator(self.patch_covariance_method)
            feature_vectors = []
            feature_map_cdfs = []
            for batch in data_iterator:                
                images = batch['image'].to(self.device)
                
                with torch.no_grad():
                    feature_maps, patches = self.embed(images)
                print(feature_maps.shape,patches.shape)
                del images
                # print(feature_maps.shape)
                patch_cdf = compute_cdf_and_binary_features(patches,n_quantiles = self.quantiles,min_q=self.min_q, max_q=self.max_q)
                patch_estimator.update(patch_cdf)

                

                patches_flat = patches.reshape(patches.shape[0], -1)  # [B*N, C*P*P]
                patch_raw_estimator.update(patches_flat)
                del patches, patch_cdf, patches_flat
            
                # Process global features
                feature_maps_reshaped = feature_maps.reshape(
                    feature_maps.shape[0], feature_maps.shape[1], -1
                )
                # feature_map_cdf = compute_cdf_and_binary_features(
                #     feature_maps_reshaped, n_quantiles=self.fm_quantiles,min_q=self.min_q, max_q=self.max_q,plot_cdf = True
                # )
                feature_map_cdf = compute_cdf_and_binary_features(
                    feature_maps_reshaped, n_quantiles=self.fm_quantiles
                )
                feature_map_cdfs.append(feature_map_cdf)

                feature_vector = torch.mean(feature_maps, dim=(2, 3))
                feature_vectors.append(feature_vector)
                del feature_maps, feature_maps_reshaped, feature_map_cdf, feature_vector
                torch.cuda.empty_cache()
        self.patch_mean, self.patch_inv_covariance = patch_estimator.finalize(self.regularization)

        self.patch_raw_mean, self.patch_raw_inv_covariance = patch_raw_estimator.finalize(self.regularization)

        feature_map_cdfs = torch.cat(feature_map_cdfs, dim=0) 
        # print(feature_map_cdfs.shape)
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
                patch_cdf = compute_cdf_and_binary_features(patches,n_quantiles = self.quantiles,min_q=self.min_q, max_q=self.max_q)#[B*N,C*Q]
                patch_score = compute_mahalanobis_distance_pytorch(patch_cdf,self.patch_mean,self.patch_inv_covariance,device=self.device)
                patch_score = patch_score.view(batch_size, -1)  #[B,N]              
                patch_scores_val.append(patch_score)

                patches_flat = patches.reshape(patches.shape[0], -1)  # [B*N, C*P*P]
                patch_raw_score = compute_mahalanobis_distance_pytorch(patches_flat,self.patch_raw_mean,self.patch_raw_inv_covariance,device=self.device)
                patch_raw_score = patch_raw_score.view(batch_size, -1)  #[B,N]
                patch_raw_scores_val.append(patch_raw_score)

        global_scores_val = torch.cat(global_scores_val,dim = 0) # [num_val_samples]
        global_mean_scores_val = torch.cat(global_mean_scores_val,dim = 0) # [num_val_samples]

        patch_scores_val = torch.cat(patch_scores_val,dim = 0) #
        patch_raw_scores_val = torch.cat(patch_raw_scores_val,dim = 0) # [num_val_samples,N]


        global_score_mean, global_score_std = self.fit_zscore_scaler_torch(global_scores_val)
        self.global_score_mean = global_score_mean
        self.global_score_std = global_score_std

        global_mean_score_mean, global_mean_score_std = self.fit_zscore_scaler_torch(global_mean_scores_val)
        self.global_mean_score_mean = global_mean_score_mean
        self.global_mean_score_std = global_mean_score_std

        patch_scores_flattened = patch_scores_val.flatten()  #展平后统一标准化（全局标准化）
        patch_score_mean, patch_score_std = self.fit_zscore_scaler_torch(patch_scores_flattened )
        self.patch_score_mean = patch_score_mean
        self.patch_score_std = patch_score_std

        patch_raw_scores_flattened = patch_raw_scores_val.flatten()  #展平后统一标准化（全局标准化）
        patch_raw_score_mean, patch_raw_score_std = self.fit_zscore_scaler_torch(patch_raw_scores_flattened )
        self.patch_raw_score_mean = patch_raw_score_mean
        self.patch_raw_score_std = patch_raw_score_std
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

        patch_distance = compute_mahalanobis_distance_pytorch(patch_cdf,self.patch_mean,self.patch_inv_covariance,device=self.device)
        patch_distance = self.transform_zscore_scaler_torch(
                                                            patch_distance,
                                                            self.patch_score_mean,
                                                            self.patch_score_std
                                                            )
        patches_flat = patches.reshape(patches.shape[0], -1)  # [B*N, C*P*P]
        patch_raw_distance = compute_mahalanobis_distance_pytorch(patches_flat,self.patch_raw_mean,self.patch_raw_inv_covariance,device=self.device)
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
        return patch_raw_score,patch_score,fused_mean_score,fused_score,patch_distance,segmentation,global_distance
        
    def _predict(self,images):
        batch_size = images.shape[0] 
        with torch.no_grad():
            feature_maps, patches,num_patches = self.embed(images,provide_patch_shapes=True) 

            #计算全局马氏距离
        feature_maps_reshaped = feature_maps.reshape(feature_maps.shape[0], feature_maps.shape[1], -1)
        feature_map_cdf = compute_cdf_and_binary_features(feature_maps_reshaped, n_quantiles=self.fm_quantiles)            
        global_distance = self.global_estimator.compute_distance(feature_map_cdf)
        global_distance = self.transform_zscore_scaler_torch(global_distance, self.global_score_mean, self.global_score_std)
        

        #计算patch马氏距离        
        patch_cdf = compute_cdf_and_binary_features(patches,n_quantiles = self.quantiles,min_q=self.min_q, max_q=self.max_q)

        patch_distance = compute_mahalanobis_distance_pytorch(patch_cdf,self.patch_mean,self.patch_inv_covariance,device=self.device)
        patch_distance = self.transform_zscore_scaler_torch(
                                                            patch_distance,
                                                            self.patch_score_mean,
                                                            self.patch_score_std
                                                            )
       
        patch_score = torch.amax(patch_distance.view(batch_size, -1), dim=1)

        # 计算融合后的分数
        fused_score = 0.5 * global_distance + 0.5 * patch_score
        return fused_score
        
    def save_model(self,save_path):

        # 保存模型状态和配置
        checkpoint = {
            'patch_mean': self.patch_mean,
            'patch_inv_covariance': self.patch_inv_covariance,
            'patch_raw_mean': self.patch_raw_mean,
            'patch_raw_inv_covariance': self.patch_raw_inv_covariance,
            'global_estimator_mean': self.global_estimator.mean,
            'global_estimator_inv_cov': self.global_estimator.inv_cov,
            'global_mean_estimator_mean': self.global_mean_estimator.mean,
            'global_mean_estimator_inv_cov': self.global_mean_estimator.inv_cov,
            'global_score_mean': self.global_score_mean,
            'global_score_std': self.global_score_std,
            'global_mean_score_mean': self.global_mean_score_mean,
            'global_mean_score_std': self.global_mean_score_std,
            'patch_score_mean': self.patch_score_mean,
            'patch_score_std': self.patch_score_std,
            'patch_raw_score_mean': self.patch_raw_score_mean,
            'patch_raw_score_std': self.patch_raw_score_std,

            'model_config': {
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
                'patch_covariance_method': self.patch_covariance_method,
                'global_covariance_method': self.global_covariance_method,
                'regularization': self.regularization,
                'pretrained_weights_path': self.pretrained_weights_path
            }
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
            patch_covariance_method=config['patch_covariance_method'],
            global_covariance_method=config['global_covariance_method'],
            regularization=config['regularization'],
            pretrained_weights_path=config['pretrained_weights_path']
        )
        
        # 加载训练好的参数
        model.patch_mean = model_state['patch_mean']
        model.patch_inv_covariance = model_state['patch_inv_covariance']
        model.patch_raw_mean = model_state['patch_raw_mean']
        model.patch_raw_inv_covariance = model_state['patch_raw_inv_covariance']
        model.global_estimator.mean = model_state['global_estimator_mean']
        model.global_estimator.inv_cov = model_state['global_estimator_inv_cov']
        model.global_mean_estimator.mean = model_state['global_mean_estimator_mean']
        model.global_mean_estimator.inv_cov = model_state['global_mean_estimator_inv_cov']
        model.global_score_mean = model_state['global_score_mean']
        model.global_score_std = model_state['global_score_std']
        model.global_mean_score_mean = model_state['global_mean_score_mean']
        model.global_mean_score_std = model_state['global_mean_score_std']
        model.patch_score_mean = model_state['patch_score_mean']
        model.patch_score_std = model_state['patch_score_std']
        model.patch_raw_score_mean = model_state['patch_raw_score_mean']
        model.patch_raw_score_std = model_state['patch_raw_score_std']

        return model