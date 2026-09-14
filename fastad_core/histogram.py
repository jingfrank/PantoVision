
import torch
def compute_cdf_and_binary_features(features, n_quantiles=100, min_q=0.01, max_q=0.99):
    """
    Compute CDF-like histogram features for the given features.

    Args:
        features (torch.Tensor): Input features of shape (N, C, H*W).
        n_quantiles (int): Number of histogram bins.
        min_q (float): Lower quantile for clipping (to get min_val).
        max_q (float): Upper quantile for clipping (to get max_val).
    """
    n_bins = n_quantiles

    # 假设 features 形状是 (N, C, H*W)
    N, C, H_W = features.shape

    # 量化时要求 float32/float64
    if features.dtype == torch.float16:
        features = features.float()

    # 展平所有空间位置，用于计算每个通道的分位数
    # flattened: (N*H*W, C)
    flattened = features.permute(0, 2, 1).reshape(-1, C)

    # 按通道计算分位数，得到每个通道的 min_val / max_val
    min_val = torch.quantile(flattened, min_q, dim=0)  # (C,)
    max_val = torch.quantile(flattened, max_q, dim=0)  # (C,)

    range_val = max_val - min_val  # (C,)

    # 防止极端情况下 range 为 0
    eps = 1e-12
    range_val = torch.clamp(range_val, min=eps)

    bin_width = range_val / n_bins  # (C,)

    # 结果 CDF: (N, C, n_bins)
    cdf = torch.zeros((N, C, n_bins), device=features.device, dtype=torch.float32)

    for c in range(C):
        # 取出当前通道数据，形状 (N, H*W)
        activation = features[:, c, :]  # (N, H*W)

        # ---- 1. 先计算原始 bin 索引（可能越界/负数）----

        # idx = floor((value - min) / width)
        # 这里故意暂时不 clamp，为了后面模仿 histc 的丢弃行为
        bin_indices = ((activation - min_val[c]) / bin_width[c]).floor().long()  # (N, H*W)

        # ---- 2. 构建合法掩码，模仿 histc：只保留 [min_val, max_val) 内的值 ----
        #
        # histc 的行为大致是：
        #   - value < min  : 丢弃
        #   - value >= max : 丢弃
        #   - 其他落在 [min, max) 的才参与统计
        #
        # 对应到索引上：
        #   - 合法 bin 索引范围是 [0, n_bins-1]
        #   - >= n_bins 或 < 0 的都应该被丢弃
        # valid_mask = (activation > min_val[c]) & (activation < max_val[c])

        # # 合法的索引还需要在 [0, n_bins-1] 内
        # valid_mask = valid_mask & (bin_indices >= 0) & (bin_indices < n_bins)

        # # 为 scatter_add_ 做准备：把非法位置的索引设置成 0（不会真正被加，因为 mask 会为 0）
        # safe_bin_indices = bin_indices.clone()
        # safe_bin_indices[~valid_mask] = 0  # 任意合法范围内的数都可以，这里取 0

        # # ---- 3. 构造“加 1 源”并应用掩码：非法位置贡献为 0 ----
        # # ones: 所有位置原始“打账”都为 1
        # ones = torch.ones_like(activation, dtype=torch.float32)

        # # 只在 valid_mask 为 True 的位置保留 1，其余置为 0，相当于把 outliers 丢弃
        # ones = ones * valid_mask.float()

        # # ---- 4. 统计直方图 (N, n_bins) ----
        # hist = torch.zeros((N, n_bins), device=features.device, dtype=torch.float32)

        # # scatter_add_: 在 dim=1 (bin 维度) 上，按 safe_bin_indices 填入 ones
        # hist.scatter_add_(1, safe_bin_indices, ones)
        # 截断索引，防止 out of bounds (处理小于min_val或大于max_val的值)
        bin_indices = torch.clamp(bin_indices, 0, n_bins - 1)
        # 创建当前通道的直方图容器 (N, n_bins)
        
        hist = torch.zeros((N, n_bins), device=features.device)
        
        # 使用 scatter_add_ 进行计数
        # dim=1 表示在 n_bins 维度上进行加和
        # src 为全1矩阵，表示每个位置计数+1
        hist.scatter_add_(1, bin_indices, torch.ones_like(activation, dtype=torch.float))
        # ---- 5. 归一化 + CDF ----
        # 归一化为概率分布，避免除 0
        denom = hist.sum(dim=1, keepdim=True) + 1e-8  # (N, 1)
        prob = hist / denom  # (N, n_bins)

        # 只对前 n_bins-1 个 bin 做 CDF
        cumulative = torch.cumsum(prob[:, :n_bins-1], dim=1)  # (N, n_bins-1)

        # 注意 cdf 的第三维长度变成 n_bins-1
        cdf[:, c, :n_bins-1] = cumulative

    # 输出形状 (N, C * (n_bins-1))
    cdf = cdf[:, :, :n_bins-1].permute(0,2,1).reshape(N, C * (n_bins-1))
    return cdf
    
