#!/usr/bin/env python
# coding: utf-8

# scripts/similarity_calculation.py

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


def compute_similarity_stats(model, dataloader, device):
    """
    Compute positive and negative sample similarity statistics.
    Returns arrays of positive and negative similarities.
    """
    pos_sims = []
    neg_sims = []
    
    model.eval()
    with torch.no_grad():
        for img1, img2 in dataloader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            
            # Get features (correctly call forward with two arguments)
            z1, z2 = model(img1, img2) 
            
            # Normalize features so cosine similarity is not affected by vector magnitude
            z1_norm = F.normalize(z1, dim=1)
            z2_norm = F.normalize(z2, dim=1)
            
            # 1. Positive similarities: z1 and z2 at corresponding positions
            pos = torch.cosine_similarity(z1_norm, z2_norm)  # [B]
            pos_sims.extend(pos.cpu().numpy())
            
            # 2. Negative similarities: among different images within z1
            batch_size = z1_norm.shape[0]
            
            # Compute similarity matrix within z1
            sim_matrix = torch.mm(z1_norm, z1_norm.T)  # [B, B]
            
            # Remove diagonal (self-similarity)
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
            neg = sim_matrix[mask]  # [B*(B-1)]
            neg_sims.extend(neg.cpu().numpy())
    
    return np.array(pos_sims), np.array(neg_sims)
    

def evaluate_model(model, val_loader, device, epoch, save_dir='./experiments'):
    """Evaluate model: calculate intra-class / inter-class similarity."""
    os.makedirs(os.path.join(save_dir, 'tsne'), exist_ok=True)
    
    model.eval()
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Handle different return formats
            if len(batch) == 2:
                images, labels = batch
            else:
                # Assume returns (img1, img2, labels), take the first view and labels
                images, _, labels = batch
     
            images = images.to(device)
            features = model.get_features(images).cpu().numpy()
            all_features.append(features)
            all_labels.extend(labels.numpy())
    
    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels)
    
    # Compute intra-class and inter-class similarities
    # Randomly sample pairs to avoid excessive computation
    n_samples = min(500, len(features))
    indices = np.random.choice(len(features), n_samples, replace=False)
    sampled_features = features[indices]
    sampled_labels = labels[indices]
    
    sim_matrix = cosine_similarity(sampled_features)
    
    pos_sims = []
    neg_sims = []
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            if sampled_labels[i] == sampled_labels[j]:
                pos_sims.append(sim_matrix[i, j])
            else:
                neg_sims.append(sim_matrix[i, j])
    
    pos_mean = np.mean(pos_sims) if pos_sims else 0
    neg_mean = np.mean(neg_sims) if neg_sims else 0
    
    return pos_mean, neg_mean


def visualize_feature_space(model, dataloader, epoch, device, save_dir='./experiments/tsne'):
    """Visualize feature space using t-SNE."""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Handle different return formats
            if len(batch) == 2:
                images, label = batch
            else:
                # Assume returns (img1, img2, labels), take the first view and labels
                images, _, label = batch

            images = images.to(device)
            # Use backbone to extract features (bypass projection head)
            feat = model.backbone(images).cpu().numpy()
            features.append(feat)
            labels.extend(label.numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    
    # t-SNE dimensionality reduction
    if len(features) < 2:
        print("Validation set has less than 2 samples, skipping t-SNE visualization.")
        return 0.0

    
    perplexity_val = min(10, len(features) - 1) # 动态调整 perplexity
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val)
    features_2d = tsne.fit_transform(features)

    tsne = TSNE(n_components=2, random_state=42, perplexity=10)
    features_2d = tsne.fit_transform(features)
    
    # Plotting
    colors = {1: 'blue', 2: 'green', 3: 'red'}
    plt.figure(figsize=(8, 6))
    for label in [1, 2, 3]:
        mask = labels == label
        if mask.any():
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=colors[label], label=f'Class {label}', alpha=0.7)
    plt.legend()
    plt.title(f'Epoch {epoch} - Feature Space')
    plt.savefig(os.path.join(save_dir, f'tsne_epoch_{epoch}.png'), dpi=150)
    plt.close()
    
    # Calculate Silhouette Score
    if len(set(labels)) > 1:
        score = silhouette_score(features_2d, labels)
        print(f"Epoch {epoch} - Silhouette Score: {score:.4f}")
        return score
    else:
        return -1