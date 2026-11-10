#!/usr/bin/env python
# coding: utf-8

## Calculate ranking scores for train and val sets to plot boxplots.

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


Image.MAX_IMAGE_PIXELS = None

from models.backbone import ResNetBackbone
from models import ContrastiveLearn
from EF_ranking_train import RankingModel


def load_pretrained_backbone(checkpoint_path, device='cpu'):
    """Load pre-trained backbone"""
    backbone = ResNetBackbone.get_backbone('resnet18')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    backbone_state = {}
    for k, v in checkpoint.items():
        if k.startswith('backbone.'):
            backbone_state[k.replace('backbone.', '')] = v
    
    backbone.load_state_dict(backbone_state)
    backbone = backbone.to(device)
    backbone.eval()
    print(f" Loaded pre-trained backbone: {checkpoint_path}")
    return backbone


def load_ranking_model(model_path, device='cpu'):
    """Load trained ranking model"""
    model = RankingModel(input_dim=512, hidden_dim=256)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f" Loaded ranking model: {model_path}")
    return model


def compute_scores_for_paths(model, backbone, image_paths, device='cpu'):
    """Calculate ranking scores for a list of images"""
    scores = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feat = backbone(img_tensor).cpu().numpy().flatten()
            feat_tensor = torch.tensor(feat, dtype=torch.float32).to(device)
            score = model(feat_tensor).item()
        scores.append(score)
    return scores


def plot_score_boxplot(train_scores, train_roughness, val_scores, val_roughness, save_path='score_boxplot.png'):
    """Plot score boxplots for training and validation sets"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training set
    train_data = [[], [], []]
    for score, rough in zip(train_scores, train_roughness):
        if rough is not None and 1 <= rough <= 3:
            train_data[int(rough)-1].append(score)
    
    axes[0].boxplot(train_data, positions=[1, 2, 3], widths=0.6)

    axes[0].set_xticklabels(['Smooth (Class 1)', 'Medium (Class 2)', 'Rough (Class 3)'])
    axes[0].set_xlabel('Roughness Level')
    axes[0].set_ylabel('Ranking Score')
    axes[0].set_title(f'Training Set (n={len(train_scores)})')
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    

    val_data = [[], [], []]
    for score, rough in zip(val_scores, val_roughness):
        if rough is not None and 1 <= rough <= 3:
            val_data[int(rough)-1].append(score)
    
    axes[1].boxplot(val_data, positions=[1, 2, 3], widths=0.6)
    axes[1].set_xticklabels(['Smooth (Class 1)', 'Medium (Class 2)', 'Rough (Class 3)'])
    axes[1].set_xlabel('Roughness Level')
    axes[1].set_ylabel('Ranking Score')
    axes[1].set_title(f'Validation Set (n={len(val_scores)})')
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f" Boxplot saved: {save_path}")
    
    # Print statistics
    print("\nTraining Set Score Statistics:")
    # Fixed the order of classes to match the array index (1, 2, 3)
    for i, name in enumerate(['Smooth (Class 1)', 'Medium (Class 2)', 'Rough (Class 3)']):
        if len(train_data[i]) > 0:
            print(f"  {name}: mean={np.mean(train_data[i]):.3f}, std={np.std(train_data[i]):.3f}, n={len(train_data[i])}")
    
    print("\nValidation Set Score Statistics:")
    for i, name in enumerate(['Smooth (Class 1)', 'Medium (Class 2)', 'Rough (Class 3)']):
        if len(val_data[i]) > 0:
            print(f"  {name}: mean={np.mean(val_data[i]):.3f}, std={np.std(val_data[i]):.3f}, n={len(val_data[i])}")


def main():

    checkpoint_path = './experiments/checkpoints/contrastive_epoch50.pth'
    ranking_model_path = './experiments/checkpoints/best_ranking_model.pth'
    pkl_path = './experiments/contrastive_train_val_paths.pkl'
    
    device = 'cpu'
    
    print("\n1. Loading models...")
    backbone = load_pretrained_backbone(checkpoint_path, device)
    ranking_model = load_ranking_model(ranking_model_path, device)
    
    print("\n2. Loading train and val image paths...")
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        train_images = data['train_paths']
        val_images = data['val_paths']
        train_roughness = data['train_labels']
        val_roughness = data['val_labels']
        print(f"Training images: {len(train_images)}")
        print(f"Validation images: {len(val_images)}")
    else:
        print(f"\nCould not find {pkl_path}!")
        print("Please run the contrastive learning training script first to generate this file.")
        return
    
    print("\n3. Using train and val paths from pkl...")

    print("Plotting score boxplots for train and val sets")


    print("\nComputing ranking scores for training set...")
    train_scores = compute_scores_for_paths(ranking_model, backbone, train_images, device)

    print("Computing ranking scores for validation set...")
    val_scores = compute_scores_for_paths(ranking_model, backbone, val_images, device)


    plot_score_boxplot(train_scores, train_roughness, val_scores, val_roughness)


    df_train = pd.DataFrame({
        'filename': [os.path.basename(p) for p in train_images],
        'score': train_scores,
        'roughness': train_roughness
    })
    df_train.to_csv('train_scores.csv', index=False)
    print(f" Training scores saved to: train_scores.csv")

    df_val = pd.DataFrame({
        'filename': [os.path.basename(p) for p in val_images],
        'score': val_scores,
        'roughness': val_roughness
    })
    df_val.to_csv('val_scores.csv', index=False)
    print(f" Validation scores saved to: val_scores.csv")


if __name__ == '__main__':
    main()