#!/usr/bin/env python
# coding: utf-8

## Extract features for a specific epoch and generate t-SNE plot and CSV data.

import os
import glob
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from models import ContrastiveLearn
from models.backbone import ResNetBackbone
from models.ContrastiveLearn import ContrastiveHead, SupConModel

# ========== Load configuration ==========
config_path = './config/config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# ========== Define Model Loading Function ==========
def load_model(checkpoint_path, device='cpu'):
    """Load trained model"""
    backbone = ResNetBackbone.get_backbone('resnet18')
    projection_head = ContrastiveHead(in_features=512, out_features=128)
    model = SupConModel(backbone, projection_head)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded: {checkpoint_path}")
    return model

# ========== Define Feature Extraction Function ==========
def extract_features(model, image_path, device='cpu'):
    """Extract feature vector for a single image"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model.backbone(image_tensor)
        features = features.cpu().numpy().flatten()
    
    return features

if __name__ == '__main__':
    # ========== Configuration ==========
    # 🔥 Updated to relative paths
    checkpoint_path = './experiments/checkpoints/contrastive_epoch2.pth'
    data_dir = config['paths']['data_dir']
    excel_path = config['paths']['excel_path']
    device = 'cpu'

    # ========== 1. Load Model ==========
    model = load_model(checkpoint_path, device)

    # ========== 2. Get all image paths ==========
    image_paths = glob.glob(os.path.join(data_dir, "*.jpg"))
    image_paths += glob.glob(os.path.join(data_dir, "*.png"))
    print(f"Found {len(image_paths)} images")

    # ========== 3. Get labels ==========
    df = pd.read_excel(excel_path)
    # 🔥 Updated column names to English
    label_dict = dict(zip(df['Filename'], df['Roughness (1-3)']))

    # ========== 4. Extract features ==========
    features = []
    labels = []
    valid_names = []

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        name = filename.replace('.jpg', '').replace('.png', '')
        
        if name in label_dict:
            feat = extract_features(model, img_path, device)
            features.append(feat)
            labels.append(label_dict[name])
            valid_names.append(filename)
            print(f"  ✅ Extracted: {filename}")

    features = np.array(features)
    labels = np.array(labels)

    print(f"\nSuccessfully extracted features for {len(features)} images, dimension: {features.shape}")

    # ========== 5. t-SNE Dimensionality Reduction ==========
    print("\nRunning t-SNE...")
    # Note: if the dataset is very small, perplexity might need to be adjusted (e.g., < number of samples)
    tsne = TSNE(n_components=2, random_state=42, perplexity=10)
    features_2d = tsne.fit_transform(features)
    print("t-SNE completed!")

    # ========== 6. Save CSV ==========
    df_tsne = pd.DataFrame({
        'x': features_2d[:, 0],
        'y': features_2d[:, 1],
        'label': labels,
        'filename': valid_names
    })
    df_tsne.to_csv('tsne_coordinates.csv', index=False)
    print("✅ t-SNE coordinates saved to: tsne_coordinates.csv")

    # ========== 7. Plot for Verification ==========
    colors = {1: 'blue', 2: 'green', 3: 'red'}
    plt.figure(figsize=(8, 6))
    for label in [1, 2, 3]:
        mask = labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                    c=colors[label], label=f'Roughness {label}', alpha=0.7, s=20)
    plt.legend()
    plt.title('t-SNE Visualization (Epoch 2)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig('tsne_epoch2.png', dpi=300)
    plt.show()
    print("✅ t-SNE plot saved: tsne_epoch2.png")