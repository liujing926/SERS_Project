#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def evaluate_with_classifier(model, val_loader, device, epoch, logger=None):
    """
    Train a KNN classifier using validation set features to evaluate roughness prediction accuracy.
    """
    model.eval()
    
    # Collect all features and labels from the validation set
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for img1, img2, labels in val_loader:  # Only take the first view
            img1 = img1.to(device)
            labels = labels.to(device)
 
            features = model.backbone(img1)  # [batch, 512]
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy()) 
    
    features = np.concatenate(all_features, axis=0)  # [num_samples, 512]
    labels = np.array(all_labels)     # e.g., 37 samples -> 37 labels (each is 1/2/3)
    
    print(f"Validation feature shape: {features.shape}, label shape: {labels.shape}")
    print(f"Label distribution: 1:{np.sum(labels==1)}, 2:{np.sum(labels==2)}, 3:{np.sum(labels==3)}")
    
    # Use KNN classifier (simple and suitable for small samples)
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')  
    knn.fit(features, labels)
    
    #  Calculate training accuracy
    train_pred = knn.predict(features)
    train_acc = accuracy_score(labels, train_pred)
    print(f"Training accuracy: {train_acc:.4f}")
  

   
    pred_labels = knn.predict(features)
    accuracy = accuracy_score(labels, pred_labels)
    
    print(f"\nEpoch {epoch+1} Validation accuracy: {accuracy:.4f}")
    
    # Print accuracy for each class
    for cls in [1, 2, 3]:
        mask = labels == cls
        if mask.any():
            cls_acc = accuracy_score(labels[mask], pred_labels[mask])
            print(f"  Roughness {cls}: {cls_acc:.4f} ({np.sum(mask)} images)")
    
    return accuracy