#!/usr/bin/env python
# coding: utf-8

"""
Supervised Contrastive Loss
Paper: "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLossWithTwoViews(nn.Module):
    """
    Supervised contrastive loss specifically designed for SimCLR-style dual views.
    """
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2, labels):
        """
        Args:
            z1: Features from the first augmented view [batch_size, dim]
            z2: Features from the second augmented view [batch_size, dim]
            labels: Roughness labels [batch_size]
        """
        batch_size = z1.size(0) 
        device = z1.device
        
        # Normalization
        z1 = F.normalize(z1, dim=1) # [B, dim]
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate two views: [2*batch_size, dim]
        features = torch.cat([z1, z2], dim=0) # [2B, dim]
        labels = torch.cat([labels, labels], dim=0)  # [2B]
        
        # Calculate similarity matrix
        sim_matrix = torch.mm(features, features.T) / self.temperature  # [2B, 2B] (Controls the sharpness of the similarity distribution)
        
        # Create positive mask
        # 1. Samples with the same label are positive pairs
        label_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float() # Broadcasting to get a [2B, 2B] matrix
        
        # 2. Two views of the same image are also positive pairs (diagonal shifted by batch_size)
        batch_indices = torch.arange(batch_size, device=device) # Creates an index sequence from 0 to batch_size-1
        view_mask = torch.zeros_like(label_mask)  # Creates a zero matrix with the same shape as label_mask
        view_mask[batch_indices, batch_indices + batch_size] = 1
        view_mask[batch_indices + batch_size, batch_indices] = 1  # view_mask marks the two augmented views of the same image as positive pairs [2B, 2B]
        
        # Combine masks
        mask = label_mask * (1 - torch.eye(2*batch_size, device=device)) + view_mask # If a pair satisfies both conditions, they add up to 2

        # ✅ Correct: Convert to boolean mask
        is_positive = (mask > 0).float()  # [2B, 2B], values are 0 or 1
        
        # Negative mask
        neg_mask = 1 - is_positive         # [2B, 2B], values are 0 or 1
        
        # Calculate exp similarity
        exp_sim = torch.exp(sim_matrix)
        
        # Denominator: sum of exp over all negative samples
        neg_sum = (exp_sim * neg_mask).sum(dim=1)
        
        # Numerator: sum of exp over positive samples
        pos_sum = (exp_sim * mask).sum(dim=1)
        
        # Calculate loss
        all_sum = pos_sum + neg_sum
        prob = pos_sum / (all_sum + 1e-8)
        loss = -torch.log(prob + 1e-8)
        
        # Only calculate loss for samples with at least 1 positive sample
        valid_mask = mask.sum(dim=1) > 0
        loss = loss[valid_mask].mean()
        
        return loss
        

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss for single view"""
    
    def __init__(self, temperature=0.1, base_temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(self, features, labels, mask=None):
        """
        Args:
            features: [batch_size, feature_dim] - Projected features
            labels: [batch_size] - Roughness labels (1, 2, 3)
            mask: Optional, custom positive mask
        Returns:
            loss: Scalar loss
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Calculate similarity matrix
        sim_matrix = torch.mm(features, features.T) / self.temperature  # [B, B]
        
        # Create positive mask
        if mask is None:
            # Samples with the same label are positive pairs
            labels = labels.contiguous().view(-1, 1)  # Reshape labels to a column vector [B, 1]
            mask = torch.eq(labels, labels.T).float().to(device)  # labels and labels.T broadcast to [B, B] for comparison, resulting in a matrix with True on the diagonal [B, B]
        
        # Remove self (set diagonal to 0)
        mask = mask * (1 - torch.eye(batch_size, device=device))
        
        # Negative mask
        neg_mask = 1 - mask
        
        # Calculate exp similarity
        exp_sim = torch.exp(sim_matrix)  # Exponentiating converts similarity to non-negative weights and amplifies differences [B, B]
        
        # Denominator: sum of exp over all negative samples
        neg_sum = (exp_sim * neg_mask).sum(dim=1) # Sum along each row, calculating the sum of similarities with all negative samples for each sample [B]
        
        # Numerator: sum of exp over positive samples
        pos_sum = (exp_sim * mask).sum(dim=1) # [B]
        
        # Calculate loss
        all_sum = (exp_sim * (mask + neg_mask)).sum(dim=1)  # Or simply exp_sim.sum(dim=1)

        # Positive sample probability = pos_sum / all_sum
        prob = pos_sum / (all_sum + 1e-8)  # [B]

        # Loss = -log(probability)
        loss = -torch.log(prob + 1e-8)   # [B]
        
        # Only calculate loss for samples with at least 1 positive sample
        valid_mask = mask.sum(dim=1) > 0
        loss = loss[valid_mask].mean()   # Scalar, log(1) = 0
        
        return loss