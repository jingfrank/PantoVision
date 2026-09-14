import logging
from datetime import datetime
import time
import os
from tqdm import tqdm
import argparse
import yaml
import random
import numpy as np
import torch

import json
import tifffile

from mvtec import MVTecDataset,DatasetSplit
from fastad3 import FastAD
from fastad_common import resize_patch_distance,RescaleSegmentor
from src.evaluation_utils import *
from src.util import compute_classification_auc_roc
from compute_metrics import compute_imagewise_retrieval_metrics, compute_pixelwise_retrieval_metrics,calculate_au_pro
from src.image import  DefectsConfig
from src.distribution_utils import DistributionPlotter
def set_seed(seed=42):
    random.seed(seed)                # Python 随机数生成器
    np.random.seed(seed)             # Numpy 随机数生成器
    torch.manual_seed(seed)          # CPU 上的随机种子
    torch.cuda.manual_seed(seed)     # 当前 GPU

    torch.backends.cudnn.deterministic = True  # 保证每次卷积算法确定
    torch.backends.cudnn.benchmark = False     # 关闭自动优化算法选择（牺牲一点性能）


# 配置日志
def setup_logging(log_dir, classname):
    """Configure logging """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f"{classname}_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file with OmegaConf support"""
    from config_utils import load_config as _load_config
    return _load_config(config_path)

def parse_list(value):
    return [int(x) for x in value.split(',')]

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="FastAD Anomaly Detection Training")
    
    # Model parameters
    parser.add_argument("--device", type=str, default=None,
                       help="Device to run on (overrides YAML when provided)")
    parser.add_argument("--layer-to-extract", type=parse_list, default=None,
                       )
    parser.add_argument("--model-name", type=str, default=None,
                       help="Model name (overrides YAML model.name)")
    parser.add_argument("--backbone", type=str, default=None,
                       help="Backbone model name (overrides YAML model.backbone)")
    parser.add_argument("--patchsize", type=int, default=None,
                       help="Patch size (overrides YAML when provided)")
    parser.add_argument("--stride", type=int, default=None,
                       help="Stride for patch extraction (overrides YAML when provided)")
    parser.add_argument("--target-size", type=int, default=None,
                       help="Model target size (overrides YAML model.target_size)")
    parser.add_argument("--n-quantiles", type=int, default=None,
                       help="Number of quantiles (overrides YAML model.n_quantiles)")
    parser.add_argument("--fm-quantiles", type=int, default=None,
                       help="FM quantiles (overrides YAML model.fm_quantiles)")
    parser.add_argument("--patch-covariance-method", type=str, default=None,
                       help="Patch covariance method (overrides YAML model.patch_covariance_method)")
    parser.add_argument("--global-covariance-method", type=str, default=None,
                       help="Global covariance method (overrides YAML model.global_covariance_method)")
    parser.add_argument("--pretrained-weights-path", type=str, default=None,
                       help="Path to pretrained weights (overrides YAML model.pretrained_weights_path)")
    
    # Dataset parameters
    parser.add_argument("--num-workers", type=int, default=None,
                       help="Number of data loader workers (overrides YAML when provided)")
    parser.add_argument("--resize", type=int, default=None,
                       help="Image resize dimension (overrides YAML when provided)")
    parser.add_argument("--classname", type=str, default=None,
                       help="MVTec AD class name to process (overrides YAML when provided)")
    parser.add_argument("--source", type=str, default=None,
                       help="Path to MVTec AD dataset (overrides YAML when provided)")
    parser.add_argument("--dataset-name", type=str, default=None,
                       help="Dataset name (overrides YAML dataset.name)")
    parser.add_argument("--train-val-split", type=float, default=None,
                       help="Train/val split (overrides YAML dataset.train_val_split)")
              
    # Training parameters
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config YAML file (overrides other args)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for results and logs (overrides YAML when provided)")
    parser.add_argument("--log-dir", type=str, default=None,
                       help="Directory for log files (overrides YAML when provided)")

    # Quantile overrides (only applied when provided)
    parser.add_argument("--min-q", type=float, default=None,
                       help="Minimum quantile (overrides YAML model.min_q when provided)")
    parser.add_argument("--max-q", type=float, default=None,
                       help="Maximum quantile (overrides YAML model.max_q when provided)")
    parser.add_argument("--fm-min-q", type=float, default=None,
                       help="Minimum quantile (overrides YAML model.fm_min_q when provided)")
    parser.add_argument("--fm-max-q", type=float, default=None,
                       help="Maximum quantile (overrides YAML model.fm_max_q when provided)")
    parser.add_argument("--regularization", type=float, default=None,
                       )
    parser.add_argument("--train-batch-size", type=int, default=None,
                       )
    parser.add_argument("--val-batch-size", type=int, default=None,
                       )
    parser.add_argument("--test-batch-size", type=int, default=None,
                       )
    parser.add_argument("--plot-distributions", action="store_true",
                       help="Plot feature distributions for normal vs anomaly")
    return parser.parse_args()

class FastADRunner:
    def __init__(self, config, logger, plot_distributions=False):
        self.config = config
        self.logger = logger
        self.plot_distributions = plot_distributions
        self.plotter = None
        if self.plot_distributions:
            plot_dir = os.path.join(self.config['logging']['output_dir'], 'distributions')
            self.plotter = DistributionPlotter(plot_dir)

    def create_dataloader(self):
        self.logger.info("Creating dataloaders...")
        train_dataset = MVTecDataset(
            source=self.config['dataset']['source'],
            classname=self.config['dataset']['classname'],
            resize=self.config['dataset']['resize'],
            split=DatasetSplit.TRAIN,
            train_val_split = self.config['dataset']['train_val_split']
        )
        val_dataset = MVTecDataset(
            source=self.config['dataset']['source'],
            classname=self.config['dataset']['classname'],
            resize=self.config['dataset']['resize'],
            split=DatasetSplit.VAL,
            train_val_split = self.config['dataset']['train_val_split']
        )
        test_dataset = MVTecDataset(
            source=self.config['dataset']['source'],
            classname=self.config['dataset']['classname'],
            resize=self.config['dataset']['resize'],
            split=DatasetSplit.TEST,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config['dataset']['train_batch_size'],
            shuffle=False,
            num_workers=self.config['dataset']['num_workers'],
            pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config['dataset']['val_batch_size'],
            shuffle=False,
            num_workers=self.config['dataset']['num_workers'],
            pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config['dataset']['test_batch_size'],
            shuffle=False,
            num_workers=self.config['dataset']['num_workers'],
            pin_memory=False
        )
        return train_loader, val_loader, test_loader

    def create_train_dataloader(self):
        self.logger.info("Creating dataloaders...")
        
        # 只创建train和val loader
        train_dataset = MVTecDataset(
            source=self.config['dataset']['source'],
            classname=self.config['dataset']['classname'],
            resize=self.config['dataset']['resize'],
            split=DatasetSplit.TRAIN,
            train_val_split = self.config['dataset']['train_val_split']
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config['dataset']['train_batch_size'],
            shuffle=False,
            num_workers=self.config['dataset']['num_workers'],
            pin_memory=True
        )

        return train_loader
    def create_val_dataloader(self):
        """单独创建val loader，在需要时才调用"""
        self.logger.info("Creating val dataloader...")
        
        val_dataset = MVTecDataset(
            source=self.config['dataset']['source'],
            classname=self.config['dataset']['classname'],
            resize=self.config['dataset']['resize'],
            split=DatasetSplit.VAL,
            train_val_split = self.config['dataset']['train_val_split']
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config['dataset']['val_batch_size'],
            shuffle=False,
            num_workers=self.config['dataset']['num_workers'],
            pin_memory=True
        )
        
        return val_loader
    def create_test_dataloader(self):
        """单独创建test loader，在需要时才调用"""
        self.logger.info("Creating test dataloader...")
        
        test_dataset = MVTecDataset(
            source=self.config['dataset']['source'],
            classname=self.config['dataset']['classname'],
            resize=self.config['dataset']['resize'],
            split=DatasetSplit.TEST,
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config['dataset']['test_batch_size'],
            shuffle=False,
            num_workers=self.config['dataset']['num_workers'],
            pin_memory=False,
        )
        
        return test_loader
    def initialize_model(self):
        self.logger.info("Initializing FastAD model...")
        fastad = FastAD(
            backbone=self.config['model']['backbone'],
            layer_to_extract=self.config['model']['layer_to_extract'],
            device=self.config['model']['device'],
            patchsize=self.config['model']['patchsize'],
            stride=self.config['model']['stride'],
            target_size = self.config['model']['target_size'],
            n_quantiles=self.config['model']['n_quantiles'],
            fm_quantiles=self.config['model']['fm_quantiles'],
            min_q=self.config['model']['min_q'],
            max_q=self.config['model']['max_q'],
            fm_min_q=self.config['model']['fm_min_q'],
            fm_max_q=self.config['model']['fm_max_q'],
            patch_covariance_method=self.config['model']['patch_covariance_method'],
            global_covariance_method=self.config['model']['global_covariance_method'],
            regularization=self.config['model']['regularization'],
            pretrained_weights_path=self.config['model'].get('pretrained_weights_path', None)
        )
        return fastad
    def predict_and_evaluate_LOCO(self, fastad, test_loader):
        self.logger.info("Starting prediction and evaluation...")
        
        patch_raw_scores = {"good": [], "logical_anomalies": [], "structural_anomalies": []}
        patch_scores = {"good": [], "logical_anomalies": [], "structural_anomalies": []}
        fused_mean_scores = {"good": [], "logical_anomalies": [], "structural_anomalies": []}
        fused_scores ={"good": [], "logical_anomalies": [], "structural_anomalies": []}
        global_scores = {"good": [], "logical_anomalies": [], "structural_anomalies": []}
        segmentations = []
        labels_gt = []
        masks_gt = []
        with tqdm(test_loader, desc="Inferring...", leave=False) as data_iterator:
            for batch in data_iterator:
                labels_gt.extend(batch["is_anomaly"].numpy())
                masks_gt.extend(batch["mask"].numpy())
                images = batch['image']
                anomaly = batch["anomaly"]
                ori_height = batch["ori_height"][0]
                ori_width = batch["ori_width"][0]
                image_ids = batch["image_id"]
                
                with torch.no_grad():
                    patch_raw_score,patch_score,fused_mean_score,fused_score,patch_distance,segmentation,global_distance = fastad.predict(images, labels=anomaly, plotter=self.plotter)
                # patch_distance = resize_patch_distance(patch_distance, ori_height, ori_width)
                segmentor = RescaleSegmentor(device="cuda", target_size=(ori_height, ori_width))
                patch_distance = segmentor.convert_to_segmentation(patch_distance)
                for anomaly_type,patch_raw_score,patch_score,fused_mean_score,fused_score,anomaly_map,global_score,image_id in zip(anomaly,patch_raw_score,patch_score,fused_mean_score,fused_score,patch_distance,global_distance,image_ids):
                    patch_raw_scores[anomaly_type].append(patch_raw_score.cpu().numpy())
                    patch_scores[anomaly_type].append(patch_score.cpu().numpy())
                    fused_mean_scores[anomaly_type].append(fused_mean_score.cpu().numpy())
                    fused_scores[anomaly_type].append(fused_score.cpu().numpy())
                    global_scores[anomaly_type].append(global_score.cpu().numpy())

                    # save anomaly map
                    if self.config["save_anomaly_maps"]:
                        anomaly_dir = os.path.join(self.config["anomaly_maps_dir"],f"{self.config['dataset']['classname']}/test/{anomaly_type}")
                        os.makedirs(anomaly_dir, exist_ok=True)
                        anomaly_map_path = os.path.join(anomaly_dir,f"{image_id}.tiff")
                        # anomaly_map_np = anomaly_map.cpu().numpy()
                        tifffile.imwrite(anomaly_map_path,anomaly_map.astype(np.float32))

                segmentations.extend(segmentation)
        # compute metrics
        agg_results = {'patch_raw_scores': {},
                          'patch_scores': {},
                          'fused_mean_scores': {},
                          'fused_scores': {},
                          'global_scores': {}

        }
        auc_roc_structural_prs = \
            compute_classification_auc_roc(patch_raw_scores['good'],patch_raw_scores['structural_anomalies'])
        auc_roc_logical_prs = \
            compute_classification_auc_roc(patch_raw_scores['good'],patch_raw_scores['logical_anomalies'])

        agg_results['patch_raw_scores']['structural_anomalies'] = auc_roc_structural_prs
        agg_results['patch_raw_scores']['logical_anomalies'] = auc_roc_logical_prs
        agg_results['patch_raw_scores']['mean'] = \
        0.5 * (auc_roc_logical_prs + auc_roc_structural_prs)

        auc_roc_structural_ps = compute_classification_auc_roc(
        patch_scores['good'], patch_scores['structural_anomalies']
    )
        auc_roc_logical_ps = compute_classification_auc_roc(
            patch_scores['good'], patch_scores['logical_anomalies']
        )
        
        agg_results['patch_scores']['structural_anomalies'] = auc_roc_structural_ps
        agg_results['patch_scores']['logical_anomalies'] = auc_roc_logical_ps
        agg_results['patch_scores']['mean'] = 0.5 * (auc_roc_logical_ps + auc_roc_structural_ps)
        
        # Image-level AUC-ROC for fused_mean_scores
        auc_roc_structural_fm = compute_classification_auc_roc(
            fused_mean_scores['good'], fused_mean_scores['structural_anomalies']
        )
        auc_roc_logical_fm = compute_classification_auc_roc(
            fused_mean_scores['good'], fused_mean_scores['logical_anomalies']
        )
        
        agg_results['fused_mean_scores']['structural_anomalies'] = auc_roc_structural_fm
        agg_results['fused_mean_scores']['logical_anomalies'] = auc_roc_logical_fm
        agg_results['fused_mean_scores']['mean'] = 0.5 * (auc_roc_logical_fm + auc_roc_structural_fm)
        
        # Image-level AUC-ROC for fused_scores
        auc_roc_structural_f = compute_classification_auc_roc(
            fused_scores['good'], fused_scores['structural_anomalies']
        )
        auc_roc_logical_f = compute_classification_auc_roc(
            fused_scores['good'], fused_scores['logical_anomalies']
        )
        
        agg_results['fused_scores']['structural_anomalies'] = auc_roc_structural_f
        agg_results['fused_scores']['logical_anomalies'] = auc_roc_logical_f
        agg_results['fused_scores']['mean'] = 0.5 * (auc_roc_logical_f + auc_roc_structural_f)
        

        auc_roc_structural_g = compute_classification_auc_roc(
            global_scores['good'], global_scores['structural_anomalies']
        )
        auc_roc_logical_g = compute_classification_auc_roc(
            global_scores['good'], global_scores['logical_anomalies']
        )
        agg_results['global_scores']['structural_anomalies'] = auc_roc_structural_g
        agg_results['global_scores']['logical_anomalies'] = auc_roc_logical_g
        agg_results['global_scores']['mean'] = 0.5 * (auc_roc_logical_g + auc_roc_structural_g)

        #Read the defects config file of the evaluated object.
        defects_config_path = os.path.join(self.config['dataset']['source'], self.config['dataset']['classname'], 'defects_config.json')
        with open(defects_config_path) as defects_config_file:
            defects_list = json.load(defects_config_file)
        defects_config = DefectsConfig.create_from_list(defects_list)

        gt_dir = os.path.join(self.config['dataset']['source'], self.config['dataset']['classname'],"ground_truth")

        anomaly_maps_test_dir = os.path.join(self.config["anomaly_maps_dir"], self.config['dataset']['classname'],"test")
        gt_maps,anomaly_maps = read_maps(
            gt_dir=gt_dir,
            anomaly_maps_test_dir=anomaly_maps_test_dir,
            defects_config=defects_config,
        )
        # Collect relevant metrics based on the ground truth and anomaly maps.
        metrics_aggregator = MetricsAggregator(
            gt_maps=gt_maps,
            anomaly_maps=anomaly_maps,
            parallel_workers=8,
            parallel_niceness=19)
        metrics = metrics_aggregator.run(
            curve_max_distance=0.1)

        localization_results = get_auc_spro_results(
            metrics=metrics,
            anomaly_maps_test_dir=anomaly_maps_test_dir)
        agg_results['auc_spro'] = localization_results['auc_spro']
        return agg_results

    def predict_and_evaluate(self, fastad, test_loader):
        self.logger.info("Starting prediction and evaluation...")
        patch_raw_scores = []
        patch_scores = []
        fused_mean_scores = []
        fused_scores = []
        patch_distances = []
        global_distances = []
        segmentations = []
        labels_gt = []
        masks_gt = []
        with tqdm(test_loader, desc="Inferring...", leave=False) as data_iterator:
            for batch in data_iterator:
                labels_gt.extend(batch["is_anomaly"].numpy())
                masks_gt.extend(batch["mask"].numpy())
                images = batch['image']
                anomaly = batch["anomaly"]

                with torch.no_grad():
                    patch_raw_score,patch_score,fused_mean_score,fused_score,patch_distance,segmentation,global_distance = fastad.predict(images, labels=anomaly, plotter=self.plotter)

                patch_raw_scores.extend(patch_raw_score.cpu().numpy())
                patch_scores.extend(patch_score.cpu().numpy())
                fused_mean_scores.extend(fused_mean_score.cpu().numpy())
                fused_scores.extend(fused_score.cpu().numpy())
                patch_distances.extend(patch_distance.cpu().numpy())
                segmentations.extend(segmentation)
                global_distances.extend(global_distance.cpu().numpy())

        # compute metrics
        self.logger.info("Computing metrics...")
        patch_raw_auroc = compute_imagewise_retrieval_metrics(patch_raw_scores,labels_gt )["auroc"]
        patch_auroc = compute_imagewise_retrieval_metrics(patch_scores,labels_gt )["auroc"]
        fused_mean_auroc = compute_imagewise_retrieval_metrics(fused_mean_scores,labels_gt )["auroc"]
        fused_auroc = compute_imagewise_retrieval_metrics(fused_scores,labels_gt )["auroc"]
        # patch_distance_auroc = compute_imagewise_retrieval_metrics(patch_distances,labels_gt )
        pixelwise_auroc = compute_pixelwise_retrieval_metrics(segmentations, masks_gt)["auroc"]
        global_auroc = compute_imagewise_retrieval_metrics(global_distances,labels_gt )["auroc"]
        aupro = calculate_au_pro(segmentations, masks_gt)
        
        results = {
            "patch_raw_auroc": patch_raw_auroc,
            "patch_auroc": patch_auroc,
            "global_auroc": global_auroc,
            "fused_mean_auroc": fused_mean_auroc,
            "fused_auroc": fused_auroc,
            # "patch_distance_auroc": patch_distance_auroc,
            "pixelwise_auroc": pixelwise_auroc,
            "aupro": aupro
        }
        return results
    def _log_nested_dict(self, parent_key, nested_dict, depth=0):
        """递归处理嵌套字典"""
        for key, value in nested_dict.items():
            current_key = f"{parent_key} - {key}" if depth > 0 else f"{parent_key} - {key}"
            
            if isinstance(value, dict):
                # 如果值还是字典，继续递归
                self._log_nested_dict(current_key, value, depth + 1)
            elif isinstance(value, (int, float)):
                # 如果是数值，直接记录
                self.logger.info(f"{current_key}: {value:.4f}")
            else:
                # 其他类型按原样记录
                self.logger.info(f"{current_key}: {value}")
    def write_nested_dict(self,f, data, indent_level=0):
        """递归写入嵌套字典"""
        indent = "  " * indent_level
        
        for key, value in data.items():
            if isinstance(value, dict):
                # 如果是字典，递归处理
                f.write(f"{indent}{key}:\n")
                self.write_nested_dict(f, value, indent_level + 1)
            elif isinstance(value, (int, float)):
                # 如果是数值，格式化输出
                f.write(f"{indent}{key}: {value:.4f}\n")
            else:
                # 其他类型原样输出
                f.write(f"{indent}{key}: {value}\n")
    def log_results(self, results):
        for metric, value in results.items():
            if isinstance(value, dict):
                self._log_nested_dict(metric, value)
            else:
                self.logger.info(f"{metric}: {value:.4f}")
        # Save results to file
        output_dir = self.config['logging']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(output_dir, f"results_{timestamp}_{self.config['dataset']['classname']}.txt")
        
        with open(results_file, 'w') as f:
            f.write("="*50 + "\n")
            f.write("Configuration Parameters\n")
            f.write("="*50 + "\n\n")
            
            # 递归写入配置参数
            def write_config(config_dict, indent=0):
                for key, value in config_dict.items():
                    if isinstance(value, dict):
                        f.write("  " * indent + f"{key}:\n")
                        write_config(value, indent + 1)
                    else:
                        f.write("  " * indent + f"{key}: {value}\n")
            
            write_config(self.config)
            
            # 保存结果
            f.write("\n" + "="*50 + "\n")
            f.write("Evaluation Results\n")
            f.write("="*50 + "\n\n")
            
            for metric, value in results.items():
                if isinstance(value, dict):
                    f.write(f"{metric}:\n")
                    self.write_nested_dict(f, value, indent_level=1)
                else:
                    f.write(f"{metric}: {value:.4f}\n")
        
        self.logger.info(f"Results saved to {results_file}")
    
    def run(self):
        

        fastad = self.initialize_model()
        train_loader = self.create_train_dataloader()
        # Training
        start_time = time.time()
        fastad.fit(train_loader)
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds.")
        del train_loader
        torch.cuda.empty_cache()

        val_loader = self.create_val_dataloader()
        # Validation
        self.logger.info("Starting validation...")
        fastad.validate(val_loader)
        
        # Testing
        del val_loader
        torch.cuda.empty_cache()
        
        test_loader = self.create_test_dataloader()
        self.logger.info("Starting testing...")
        if self.config['dataset']['name'] == 'MVTec-LOCO':
            results = self.predict_and_evaluate_LOCO(fastad, test_loader)
            # results = self.predict_and_evaluate(fastad, test_loader)
        else:
            results = self.predict_and_evaluate(fastad, test_loader)

        if self.plotter:
            self.plotter.plot_and_save()

        self.log_results(results)
        # Save model
        save_model = self.config.get('checkpoints', {}).get('save_model', False)
        if save_model:
            os.makedirs(self.config['checkpoints']['save_dir'], exist_ok=True)
            save_path = os.path.join(
                self.config['checkpoints']['save_dir'],
                "fastad.pth"
            )
            fastad.save_model(save_path)
        return 
def merge_config_with_args(config, args):
    """Merge command line arguments with config file"""
    if args.device is not None:
        config['model']['device'] = args.device
    if args.layer_to_extract is not None:
        config['model']['layer_to_extract'] = args.layer_to_extract
    if args.patchsize is not None:
        config['model']['patchsize'] = args.patchsize
    if args.stride is not None:
        config['model']['stride'] = args.stride
    if args.regularization is not None:
        config['model']['regularization'] = args.regularization
    if getattr(args, 'model_name', None) is not None:
        config['model']['name'] = args.model_name
    if args.backbone is not None:
        config['model']['backbone'] = args.backbone
    if getattr(args, 'target_size', None) is not None:
        config['model']['target_size'] = args.target_size
    if getattr(args, 'n_quantiles', None) is not None:
        config['model']['n_quantiles'] = args.n_quantiles
    if getattr(args, 'fm_quantiles', None) is not None:
        config['model']['fm_quantiles'] = args.fm_quantiles
    if getattr(args, 'patch_covariance_method', None) is not None:
        config['model']['patch_covariance_method'] = args.patch_covariance_method
    if getattr(args, 'global_covariance_method', None) is not None:
        config['model']['global_covariance_method'] = args.global_covariance_method
    if getattr(args, 'pretrained_weights_path', None) is not None:
        config['model']['pretrained_weights_path'] = args.pretrained_weights_path
    if args.num_workers is not None:
        config['dataset']['num_workers'] = args.num_workers
    if args.resize is not None:
        config['dataset']['resize'] = args.resize
        config['model']['image_size'] = args.resize
    if args.source is not None:
        config['dataset']['source'] = args.source
    if args.min_q is not None:
        config['model']['min_q'] = args.min_q
    if args.max_q is not None:
        config['model']['max_q'] = args.max_q
    if args.fm_min_q is not None:
        config['model']['fm_min_q'] = args.fm_min_q
    if args.fm_max_q is not None:
        config['model']['fm_max_q'] = args.fm_max_q
    if getattr(args, 'classname', None) is not None:
        config['dataset']['classname'] = args.classname
    if getattr(args, 'dataset_name', None) is not None:
        config['dataset']['name'] = args.dataset_name
    if getattr(args, 'train_val_split', None) is not None:
        config['dataset']['train_val_split'] = args.train_val_split
    if args.output_dir is not None:
        config['logging']['output_dir'] = args.output_dir
    if args.log_dir is not None:
        config['logging']['log_dir'] = args.log_dir
    if args.train_batch_size is not None:
        config['dataset']['train_batch_size'] = args.train_batch_size
    if args.val_batch_size is not None:
        config['dataset']['val_batch_size'] = args.val_batch_size
    if args.test_batch_size is not None:
        config['dataset']['test_batch_size'] = args.test_batch_size

    return config

def main():
    # set_seed(30)
    args = parse_arguments()
    # prefer config from CLI if provided, otherwise fall back to default
    config_file_path = args.config if args.config else "/home/11506/FastAD/configs/MVTec_AD/cable/base.yaml"

    config  = load_config(config_file_path)
    # Merge command-line overrides into config
    config = merge_config_with_args(config, args) 
    
    logger = setup_logging(config['logging']['log_dir'],config['dataset']['classname'])
    logger.info(f"Loading conf from {config_file_path}")
    # logger.info("Using set_features_IQR")
    runner = FastADRunner(config, logger, plot_distributions=args.plot_distributions) 
    runner.run()
    return 

if __name__ == "__main__":
    main()
