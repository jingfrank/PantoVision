import torch
import torch.nn.functional as F
from plot_cdf import save_cdf_plot_per_channel
def compute_cdf_and_binary_features(features, n_quantiles=100,min_q = 0.01,max_q = 0.99,plot_cdf = False, labels=None, name="", items_per_label=1, plotter=None):
    """
    计算累积分布函数(CDF)和二值化特征集合
    
    Args:
        features: [B*N, C, P*P] 的特征张量
        n_quantiles: 分位数的数量
        labels: Optional labels for the batch (for debugging/logging)
        name: Name of the feature set (e.g. "Global", "Patch")
        items_per_label: Number of items (N) per label (B)
        plotter: DistributionPlotter instance
    
    Returns:
        cdf: [B*N, C*n_quantiles] 累积分布函数值
        binary_set: [B*N, C*n_quantiles, P*P] 二值化特征集合
    """
    # print(f"Input features shape: {features.shape}, n_quantiles: {n_quantiles}")
    device = features.device
    B_N, C, P_P = features.shape
    eps = 1e-6
    # features = features / torch.sqrt(torch.sum(features**2, dim=-1, keepdim=True) + eps)
    input_dtype = features.dtype  # 记录输入数据类型
    if features.dtype == torch.float16:
        features = features.float()
    
    # Step 1: Vectorized quantile computation on GPU
    flattened = features.permute(0, 2, 1).reshape(-1, C)  # [B*N*P*P, C]
    
    if flattened.shape[0] > 80000:
        # Chunked vectorized quantile for massive training batches
        min_vals_list = []
        max_vals_list = []
        step = 256
        for c_start in range(0, C, step):
            c_end = min(c_start + step, C)
            sub = flattened[:, c_start:c_end]
            min_vals_list.append(torch.quantile(sub, min_q, dim=0))
            max_vals_list.append(torch.quantile(sub, max_q, dim=0))
        min_vals = torch.cat(min_vals_list)
        max_vals = torch.cat(max_vals_list)
    else:
        min_vals = torch.quantile(flattened, min_q, dim=0)
        max_vals = torch.quantile(flattened, max_q, dim=0)
    
    if labels is not None:
        # print(f"[{name}] Labels: {labels}")
        # print(f"[{name}] Max vals (0.99 quantile) mean: {max_vals.mean().item():.4f}, std: {max_vals.std().item():.4f}")
        # print(f"[{name}] Min vals (0.01 quantile) mean: {min_vals.mean().item():.4f}, std: {min_vals.std().item():.4f}")
        if plotter is not None:
            plotter.update(name, features, labels, items_per_label)

    # min_vals = torch.quantile(flattened.cpu(), min_q, dim=0)  # [C]
    # max_vals = torch.quantile(flattened.cpu(),max_q, dim=0)  # [C]
    # min_vals = min_vals.to(device)
    # max_vals = max_vals.to(device)
    # print(max_vals)
    # Step 2: 初始化输出张量
    cdf = torch.zeros((B_N, C, n_quantiles),device = device)              # [B*N, C, n_quantiles]
    
    temperature = 0.01  # 越小越接近硬二值化，越大越平滑
    # Step 3: 对每个分位数计算阈值和二值化
    for q in range(n_quantiles):
        # 计算第q个分位数的阈值
        threshold = min_vals + (max_vals - min_vals) * (q + 1) / (n_quantiles + 1)
        # threshold: [C]
        # threshold = torch.log(1 + torch.exp(threshold))  # 使用softplus确保阈值为正
        # 扩展阈值维度以匹配features的形状
        
        threshold_expanded = threshold.unsqueeze(0).unsqueeze(2)  # [1, C, 1]
        
        # 二值化：判断features中每个元素是否小于对应channel的阈值
        binary_mask = (features >= threshold_expanded).float()  # [B*N, C, P*P]
        # soft_mask = torch.sigmoid((threshold_expanded - features) / temperature)
 
        # 计算CDF：在P*P维度上取均值（即在空间维度上的比例）
        cdf[:, :, q] = binary_mask.mean(dim=2)  # [B*N, C]
    if plot_cdf:
        save_cdf_plot_per_channel(cdf, save_dir="/root/shared-nvme/FastAD/plot/cdf", prefix="cdf_plot", linewidth=1.5,)
    # 将cdf重塑为 [B*N, C*n_quantiles]
    cdf_final = cdf.permute(0, 2, 1).reshape(B_N, -1)  # [B*N, C*n_quantiles]
    # print(f"Output cdf shape: {cdf_final.shape}, binary_set shape: {binary_set_final.shape}")
    return cdf_final
