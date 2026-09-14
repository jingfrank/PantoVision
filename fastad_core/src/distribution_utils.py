import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import random

class DistributionPlotter:
    def __init__(self, save_dir, max_samples=10000):
        self.save_dir = save_dir
        self.max_samples = max_samples
        # Structure: self.data[name][channel_idx][label_type] = list of values
        self.data = {} 
        
    def update(self, name, features, labels, items_per_label=1):
        """
        features: [Total_Items, C, Spatial_Dim]
        labels: list of length B
        items_per_label: int, Total_Items = B * items_per_label
        """
        if labels is None:
            return
            
        B = len(labels)
        Total_Items, C, Spatial_Dim = features.shape
        
        if Total_Items != B * items_per_label:
            print(f"Warning: Shape mismatch in plotter update. Features: {features.shape}, B: {B}, items: {items_per_label}")
            return

        # Reshape features to [B, items_per_label, C, Spatial_Dim]
        features_reshaped = features.view(B, items_per_label, C, Spatial_Dim)
        
        if name not in self.data:
            self.data[name] = {}
            
        # Iterate over channels
        for c in range(C):
            if c not in self.data[name]:
                self.data[name][c] = {'good': [], 'anomaly': []}
                
            for b in range(B):
                label = labels[b]
                # Determine category
                category = 'good' if label == 'good' else 'anomaly'
                
                # Extract values for this image, this channel
                # [items_per_label, Spatial_Dim]
                vals = features_reshaped[b, :, c, :].flatten()
                
                # Downsample if too many values in this single update
                if vals.numel() > 1000:
                     # Random sample 1000 values
                     indices = torch.randperm(vals.numel())[:1000]
                     vals = vals[indices]
                
                vals_np = vals.cpu().numpy()
                
                current_list = self.data[name][c][category]
                
                if len(current_list) < self.max_samples:
                    self.data[name][c][category].extend(vals_np)

    def plot_and_save(self):
        print(f"Plotting distributions to {self.save_dir}...")
        for name in self.data:
            name_dir = os.path.join(self.save_dir, name)
            os.makedirs(name_dir, exist_ok=True)
            
            # Sort channels to process in order
            channels = sorted(self.data[name].keys())
            
            for c in channels:
                good_data = self.data[name][c]['good']
                anomaly_data = self.data[name][c]['anomaly']
                
                if not good_data and not anomaly_data:
                    continue
                    
                plt.figure(figsize=(10, 6))
                
                # Plot histograms
                if good_data:
                    plt.hist(good_data, bins=50, alpha=0.5, color='green', label='Normal', density=True)
                if anomaly_data:
                    plt.hist(anomaly_data, bins=50, alpha=0.5, color='red', label='Anomaly', density=True)
                
                plt.title(f'{name} Feature Distribution - Channel {c}')
                plt.xlabel('Feature Value')
                plt.ylabel('Density')
                plt.legend()
                
                save_path = os.path.join(name_dir, f'channel_{c:04d}.png')
                plt.savefig(save_path)
                plt.close()
        print("Plotting done.")
