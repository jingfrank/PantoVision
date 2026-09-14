import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from pathlib import Path
import os


def visualize_segmentation_results(image_dir, segmentation_dir, ground_truth_dir, 
                                   output_dir='visualization_results'):
    """
    可视化分割结果：读取图像、叠加分割与真实掩码并保存结果

    参数:
        image_dir: 原始图像目录
        segmentation_dir: 模型预测的分割掩码目录
        ground_truth_dir: 真实掩码目录
        output_dir: 输出目录
    """
    os.makedirs(os.path.join(output_dir, 'image_gt_overlay'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'image_seg_overlay'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'overview_triplet'), exist_ok=True)
    # 获取所有图像文件
    image_files = sorted([f for f in os.listdir(image_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    if not image_files:
        print(f"⚠️ No images found in {image_dir}")
        return

    transform_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    resize_mask = transforms.Resize((224, 224))

    cmap = plt.cm.jet
    alpha = 0.7

    for image_name in image_files:
        base_name = os.path.splitext(image_name)[0]
        image_path = os.path.join(image_dir, image_name)

        # 读取原始图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform_img(image)
        image_np = image_tensor.permute(1, 2, 0).numpy()

        # ---------- 处理 Segmentation ----------
        seg_map = None
        seg_path = os.path.join(segmentation_dir, f"{base_name}.tiff")
        if not os.path.exists(seg_path):
            seg_path = os.path.join(segmentation_dir, image_name)

        # 如果 segmentation_dir/base_name/ 下有多个 mask 文件
        if not os.path.exists(seg_path):
            seg_subdir = os.path.join(segmentation_dir, base_name)
            if os.path.isdir(seg_subdir):
                mask_files = sorted([os.path.join(seg_subdir, f)
                                     for f in os.listdir(seg_subdir)
                                     if f.lower().endswith(('.png', '.jpg'))])
                if mask_files:
                    seg_maps = [np.array(resize_mask(Image.open(f).convert('L')), dtype=np.float32)
                                for f in mask_files]
                    seg_map = np.maximum.reduce(seg_maps)  # 合并多个 mask（取最大值）
        else:
            seg_map = np.array(resize_mask(Image.open(seg_path).convert('L')), dtype=np.float32)

        if seg_map.max() > seg_map.min():
            seg_map = (seg_map - seg_map.min()) / (seg_map.max() - seg_map.min())
        else:
            seg_map = np.zeros_like(seg_map)

        # ---------- 处理 Ground Truth ----------
        gt_map = None
        gt_path = os.path.join(ground_truth_dir, f"{base_name}.png")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(ground_truth_dir, image_name)

        # 如果 ground_truth_dir/base_name/ 下有多个 mask 文件
        if not os.path.exists(gt_path):
            gt_subdir = os.path.join(ground_truth_dir, base_name)
            if os.path.isdir(gt_subdir):
                mask_files = sorted([os.path.join(gt_subdir, f)
                                     for f in os.listdir(gt_subdir)
                                     if f.lower().endswith(('.png', '.jpg'))])
                if mask_files:
                    gt_maps = [np.array(resize_mask(Image.open(f).convert('L')), dtype=np.float32)
                               for f in mask_files]
                    gt_map = np.maximum.reduce(gt_maps)
        else:
            gt_map = np.array(resize_mask(Image.open(gt_path).convert('L')), dtype=np.float32)

        if gt_map.max() > gt_map.min():
            gt_map = (gt_map - gt_map.min()) / (gt_map.max() - gt_map.min())
        else:
            gt_map = gt_map = np.zeros_like(gt_map)

        # ---------- 创建叠加图 ----------
        gt_colored = cmap(gt_map)
        gt_overlay = image_np.copy()
        for c in range(3):
            gt_overlay[:, :, c] = image_np[:, :, c] * (1 - alpha) + gt_colored[:, :, c] * alpha * gt_map

        seg_colored = cmap(seg_map)
        seg_overlay = image_np.copy()
        for c in range(3):
            seg_overlay[:, :, c] = image_np[:, :, c] * (1 - alpha) + seg_colored[:, :, c] * alpha * seg_map

        # ---------- 保存 ----------
        plt.imsave(os.path.join(output_dir, 'image_gt_overlay', f'{base_name}.png'),
                   gt_overlay)
        plt.imsave(os.path.join(output_dir, 'image_seg_overlay', f'{base_name}.png'),
                   seg_overlay)


        # ====== 生成三联图（原图 + 叠加分割 + 原始GT） ======
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image_np)
        axes[0].set_title("Original", fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(seg_overlay)
        axes[1].set_title("Segmentation Overlay", fontsize=12)
        axes[1].axis('off')

        axes[2].imshow(gt_map, cmap='gray')
        axes[2].set_title("Ground Truth", fontsize=12)
        axes[2].axis('off')

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'overview_triplet', f'{base_name}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
    print(f"✅ 所有结果已保存到：{output_dir}")
    print(f"  ├── Image + Ground Truth overlays: {os.path.join(output_dir, 'image_gt_overlay')}")
    print(f"  └── Image + Segmentation overlays: {os.path.join(output_dir, 'image_seg_overlay')}")


def visualize_categories(image_base_dir, segmentation_base_dir, ground_truth_base_dir,
                         output_base_dir, categories, subfolders=("logical_anomalies", "structural_anomalies")):
    """
    可视化多个类别的分割结果，支持子文件夹（如 logical_anomalies、structural_anomalies）

    参数:
        image_base_dir: 原始图像基目录
        segmentation_base_dir: 分割掩码基目录
        ground_truth_base_dir: 真实掩码基目录
        output_base_dir: 输出基目录
        categories: 类别列表
        subfolders: 子目录列表（默认 logical_anomalies 和 structural_anomalies）
    """
    for category in categories:
        print(f"\n🟢 正在处理类别: {category}")
        for subfolder in subfolders:
            print(f"  🔹 子类别: {subfolder}")

            image_dir = os.path.join(image_base_dir, category, "test", subfolder)
            segmentation_dir = os.path.join(segmentation_base_dir, category, "test", subfolder)
            ground_truth_dir = os.path.join(ground_truth_base_dir, category, "ground_truth", subfolder)
            output_dir = os.path.join(output_base_dir, category, "test", subfolder)

            # 确保路径存在
            if not os.path.exists(image_dir):
                print(f"    ⚠️ 路径不存在: {image_dir}")
                continue

            os.makedirs(output_dir, exist_ok=True)

            visualize_segmentation_results(
                image_dir=image_dir,
                segmentation_dir=segmentation_dir,
                ground_truth_dir=ground_truth_dir,
                output_dir=output_dir
            )
# 使用示例
if __name__ == "__main__":
    image_base_dir = "/root/shared-nvme/mvtec_loco"
    segmentation_base_dir = "/root/shared-nvme/FastAD/anomaly_maps_dir"
    ground_truth_base_dir = "/root/shared-nvme/mvtec_loco"
    output_base_dir = "/root/visual_output/MVTec_LOCO"
    #"screw_bag",
    categories = ["pushpins"]  # 示例类别列表
    visualize_categories(
        image_base_dir=image_base_dir,
        segmentation_base_dir=segmentation_base_dir,
        ground_truth_base_dir=ground_truth_base_dir,
        output_base_dir=output_base_dir,
        categories=categories
    )
