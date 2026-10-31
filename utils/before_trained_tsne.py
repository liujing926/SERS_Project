#!/usr/bin/env python
# coding: utf-8

# analyze_random_init.py

import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, random_split

from data.dataset import EFDataset, ContrastiveDataset
from data.transforms import get_contrastive_transform
from models.backbone import ResNetBackbone
from models import ContrastiveLearn 
from utils.logger import setup_logger
from utils.helpers import set_seed


def main():
    # 🔥 Fixed: Added missing device definition
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load configuration
    config_path = './config/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Create necessary directories
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    os.makedirs('./experiments/tsne', exist_ok=True)

    # Set random seed and logger
    set_seed(42)
    logger = setup_logger(config['paths']['log_dir'])
    logger.info("Starting analysis of random initialization")

    # Create dataset
    path_dataset = EFDataset(
        image_dir=config['paths']['data_dir'],
        is_labeled=True,
        transform=None
    )
    
    train_size = int(0.8 * len(path_dataset))
    val_size = len(path_dataset) - train_size
    train_dataset, val_dataset = random_split(path_dataset, [train_size, val_size]) 
    print(f"Train set: {train_size}, Val set: {val_size}")

    train_labels = [path_dataset.labels[i] for i in train_dataset.indices]
    
    contrastive_dataset_train = ContrastiveDataset(
        image_paths=[path_dataset.image_paths[i] for i in train_dataset.indices],
        labels=train_labels, 
        transform=get_contrastive_transform()
    )
    
    dataloader_train = DataLoader(
        contrastive_dataset_train,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )

    # Load randomly initialized model
    backbone = ResNetBackbone.get_backbone('resnet18')
    projection = ContrastiveLearn.ContrastiveHead(in_features=512, out_features=128)
    model = ContrastiveLearn.SupConModel(backbone, projection)
    model = model.to(device)
    model.eval()

    # ========== Manually extract features ==========
    all_features = []
    all_labels = []

    with torch.no_grad():
        for img1, img2, labels in dataloader_train:
            img1 = img1.to(device)
            # Extract backbone features (512-dim)
            features = model.backbone(img1).cpu().numpy()
            all_features.append(features)
            all_labels.extend(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels)

    print(f"Feature shape: {features.shape}, Label shape: {labels.shape}")
    # =================================================

    # t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=10)
    features_2d = tsne.fit_transform(features)

    # Plot
    colors = {1: 'blue', 2: 'green', 3: 'red'}
    plt.figure(figsize=(8, 6))
    for label in [1, 2, 3]:
        mask = labels == label
        if mask.any():
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=colors[label], label=f'Class {label}', alpha=0.7, s=20)
    plt.legend()
    plt.title('Feature Space Before Training (Random Initialization)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig('./experiments/tsne/tsne_epoch_0_random.png', dpi=150)
    plt.show()

    # Calculate Silhouette Score
    if len(set(labels)) > 1:
        score = silhouette_score(features_2d, labels)
        print(f"Silhouette Score before training: {score:.4f}")

    # ========== Export t-SNE coordinates to CSV ==========
    df_tsne = pd.DataFrame({
        'x': features_2d[:, 0],
        'y': features_2d[:, 1],
        'label': labels,
    })
    
    csv_path = './experiments/tsne/tsne_epoch_0_random.csv'
    df_tsne.to_csv(csv_path, index=False)
    print(f"✅ t-SNE coordinates saved to: {csv_path}")


if __name__ == '__main__':
    main()