import torch
import torch.nn.functional as F
def compute_cdf_and_binary_features(features, n_quantiles=3,min_q = 0.01,max_q = 0.99, temperature=5.0):
    """
    计算累积分布函数(CDF)和二值化特征集合
    
    Args:
        features: [B*N, C, P*P] 的特征张量
        n_quantiles: 分位数的数量
    
    Returns:
        cdf: [B*N, C*n_quantiles] 累积分布函数值
        binary_set: [B*N, C*n_quantiles, P*P] 二值化特征集合
    """
    # print(f"Input features shape: {features.shape}, n_quantiles: {n_quantiles}")
    device = features.device
    dtype = features.dtype
    input_dtype = features.dtype  # 记录输入数据类型
    if features.dtype == torch.float16:
        features = features.float()
    B_N, C, L = features.shape
    with torch.no_grad():
        flattened = features.permute(0, 2, 1).reshape(-1, C) # [B*N*L, C]
        min_vals = torch.quantile(flattened, min_q, dim=0)   # [C]
        max_vals = torch.quantile(flattened, max_q, dim=0)   # [C]
    
    # Step 2: 生成分位数阈值
    # thresholds shape: [C, n_quantiles]
    quantiles = torch.linspace(1, n_quantiles, n_quantiles, device=device) / (n_quantiles + 1)
    # 利用广播机制生成阈值
    # min: [C, 1], max: [C, 1], q: [1, n_quantiles]
    min_v = min_vals.unsqueeze(1)
    max_v = max_vals.unsqueeze(1)
    q_vals = quantiles.unsqueeze(0)
    
    thresholds = min_v + (max_v - min_v) * q_vals # [C, n_quantiles]
    
    # Step 3: 计算 Soft CDF
    # 我们需要比较 features [B, C, L] 和 thresholds [C, Q]
    # 扩展维度以便广播:
    # feat_exp:  [B, C, L, 1]
    # thres_exp: [1, C, 1, Q]
    feat_exp = features.unsqueeze(-1)
    thres_exp = thresholds.unsqueeze(0).unsqueeze(2)
    
    # Soft Heaviside Step Function: 
    # Sigmoid(temp * (threshold - value))
    # 当 value < threshold 时，(thresh - value) > 0 -> sigmoid -> 1
    # 当 value > threshold 时，(thresh - value) < 0 -> sigmoid -> 0
    diff = (thres_exp - feat_exp) * temperature
    soft_binary = torch.sigmoid(diff) # [B, C, L, n_quantiles]
    
    # 计算 CDF：在空间维度 (L) 上取平均
    cdf = soft_binary.mean(dim=2) # [B, C, n_quantiles]
    
    # Step 4: Flatten output for Loss
    cdf_final = cdf.reshape(B_N, -1) # [B, C * n_quantiles]
    
    return cdf_final