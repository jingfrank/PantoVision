"""
Training Pipeline for HRNet Pantograph Keypoint Localization (4-Keypoint Scheme).
Implements heatmap MSE loss, PCK evaluation metrics, model checkpointing, and ONNX export.
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from .hrnet import HRNetKeypoint, decode_heatmaps
from .dataset import PantographKeypointDataset


def compute_pck(preds, targets, threshold=0.05):
    """
    Computes Percentage of Correct Keypoints (PCK).
    preds: (B, K, 2)
    targets: (B, K, 2)
    threshold: normalized distance threshold relative to bounding box diagonal
    """
    batch_size, num_kpts, _ = preds.shape
    correct = 0
    total = 0

    for b in range(batch_size):
        valid = targets[b, :, 0] > 0
        if not np.any(valid):
            continue
        min_xy = np.min(targets[b, valid, :2], axis=0)
        max_xy = np.max(targets[b, valid, :2], axis=0)
        norm_dist = np.linalg.norm(max_xy - min_xy) + 1e-6

        dists = np.linalg.norm(preds[b, :, :2] - targets[b, :, :2], axis=1)
        correct += np.sum((dists / norm_dist) <= threshold)
        total += num_kpts

    return correct / max(1, total)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for i, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        targets = batch["target_heatmaps"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(1, len(dataloader))


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_pck_05 = []
    all_pck_10 = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["target_heatmaps"].to(device)
            gt_kpts = batch["keypoints"].cpu().numpy()

            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            preds_hm, _ = decode_heatmaps(outputs)
            # Rescale heatmap coords (72x96) to target image coords (288x384), factor 4x
            preds_img = preds_hm * 4.0

            pck05 = compute_pck(preds_img, gt_kpts, threshold=0.05)
            pck10 = compute_pck(preds_img, gt_kpts, threshold=0.10)
            all_pck_05.append(pck05)
            all_pck_10.append(pck10)

    val_loss = running_loss / max(1, len(dataloader))
    mean_pck05 = float(np.mean(all_pck_05)) if all_pck_05 else 0.0
    mean_pck10 = float(np.mean(all_pck_10)) if all_pck_10 else 0.0

    return val_loss, mean_pck05, mean_pck10


def export_onnx(model, output_path="hrnet_pantograph.onnx", input_size=(1, 3, 288, 384)):
    try:
        import onnx
        model.eval()
        device = next(model.parameters()).device
        dummy_input = torch.randn(*input_size, device=device)
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=13,
            input_names=["input"],
            output_names=["heatmaps"],
            dynamic_axes={"input": {0: "batch_size"}, "heatmaps": {0: "batch_size"}},
        )
        print(f"[Export] Saved ONNX model to: {output_path}")
    except Exception as e:
        print(f"[Export] Note: ONNX export skipped ({e}). PyTorch model .pth checkpoint is safely saved.")


def main():
    parser = argparse.ArgumentParser(description="Train HRNet for Pantograph Keypoint Localization")
    parser.add_argument("--train_json", type=str, default="data/pantograph_train.json")
    parser.add_argument("--val_json", type=str, default="data/pantograph_val.json")
    parser.add_argument("--img_dir", type=str, default="/home/ibd/jyb/gongwang_hrnet_tilt/train")
    parser.add_argument("--num_joints", type=int, default=4)
    parser.add_argument("--width", type=int, default=32, choices=[18, 32, 48])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="weights")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== Initializing HRNet-W{args.width} for {args.num_joints} Pantograph Keypoints ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_dataset = PantographKeypointDataset(
        args.train_json, args.img_dir, target_size=(288, 384), heatmap_size=(72, 96), is_train=True
    )
    val_dataset = PantographKeypointDataset(
        args.val_json, args.img_dir, target_size=(288, 384), heatmap_size=(72, 96), is_train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Dataset Loaded: {len(train_dataset)} Train Samples, {len(val_dataset)} Validation Samples")

    model = HRNetKeypoint(num_joints=args.num_joints, width=args.width).to(device)

    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 50], gamma=0.1)

    best_pck = 0.0
    start_time = time.time()

    print(f"=== Starting Training for {args.epochs} Epochs ===")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, pck05, pck10 = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        is_best = pck05 > best_pck
        if is_best:
            best_pck = pck05
            best_path = os.path.join(args.save_dir, f"best_hrnet_w{args.width}_panto.pth")
            torch.save(model.state_dict(), best_path)

        latest_path = os.path.join(args.save_dir, f"latest_hrnet_w{args.width}_panto.pth")
        torch.save(model.state_dict(), latest_path)

        if epoch % 5 == 0 or epoch == 1 or is_best:
            print(
                f"Epoch [{epoch:02d}/{args.epochs}] ({elapsed:.1f}s) "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                f"PCK@0.05: {pck05 * 100:.1f}% | PCK@0.10: {pck10 * 100:.1f}% "
                f"{'[BEST]' if is_best else ''}"
            )

    total_time = (time.time() - start_time) / 60.0
    print(f"\nTraining Complete in {total_time:.2f} mins! Best PCK@0.05: {best_pck * 100:.2f}%")

    # Export ONNX
    best_weights = os.path.join(args.save_dir, f"best_hrnet_w{args.width}_panto.pth")
    if os.path.exists(best_weights):
        model.load_state_dict(torch.load(best_weights))
    onnx_path = os.path.join(args.save_dir, f"hrnet_w{args.width}_pantograph.onnx")
    export_onnx(model, output_path=onnx_path)


if __name__ == "__main__":
    main()
