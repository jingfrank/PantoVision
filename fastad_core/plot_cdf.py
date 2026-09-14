import torch
import matplotlib.pyplot as plt
import os
import numpy as np
def save_cdf_plot_per_channel(cdf, save_dir="cdf_plots", prefix="sample",
                              linewidth=1.5, figsize_cm=(1.11, 1.11), dpi=300):
    """
    为每个样本的每个通道分别绘制CDF折线图并保存
    
    Args:
        cdf: [B*N, C, n_quantiles] 的张量
        save_dir: 保存路径
        prefix: 文件名前缀
        linewidth: 线条宽度
        figsize_cm: 图像大小（厘米）
        dpi: 图像分辨率
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # cm 转英寸
    figsize_inch = (figsize_cm[0] / 2.54, figsize_cm[1] / 2.54)
    
    B_N, C, n_quantiles = cdf.shape
    x_vals = np.arange(1, n_quantiles + 1) / (n_quantiles + 1)


    for i in range(B_N):
        for c in range(C):
            plt.figure(figsize=figsize_inch, dpi=dpi)
            
            y_vals = cdf[i, c].cpu().numpy()
            
            # step线 + 灰色虚线圆点
            plt.step(x_vals, y_vals, linewidth=linewidth)
            # plt.plot(x_vals, y_vals, "o--", color='grey', 
            #          markersize=max(1, 0.5 * figsize_cm[0]))  # 自动缩放
            
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')  # 去掉坐标轴
            plt.tight_layout(pad=0)
            
            # 文件名：样本编号 + 通道编号
            save_path = os.path.join(save_dir, f"{prefix}_sample{i}_ch{c}.svg")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close()
            
            print(f"Saved: {save_path}")
