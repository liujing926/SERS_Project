#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def evaluate_classification(model, dataloader, device):
    """Evaluate roughness classification accuracy."""
    model.eval()
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features = model.backbone(images)
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    if len(all_features) == 0:
        print("Error: No features extracted.")
        return 0.0
    
    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels)
    
    print(f"Test set feature shape: {features.shape}, label shape: {labels.shape}")
    
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(features, labels)
    pred = knn.predict(features)
    acc = accuracy_score(labels, pred)
    
    return acc