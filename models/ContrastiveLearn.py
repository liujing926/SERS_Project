#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn

class ContrastiveHead(nn.Module):
    """Projection head for contrastive learning"""
    def __init__(self, in_features=512, hidden_features=256, out_features=128):
        super().__init__()  
        self.projector = nn.Sequential(
            nn.Linear(in_features, hidden_features), 
            nn.ReLU(),                              
            nn.Linear(hidden_features, out_features)  
        )
        
    def forward(self, x):
        return self.projector(x)
        

class SupConModel(nn.Module):
    """Supervised Contrastive Learning Model"""
    
    def __init__(self, backbone, projection_head):
        super().__init__()
        self.backbone = backbone
        self.projection_head = projection_head

    def forward(self, x1, x2=None):
        """Returns projected features"""
        if x2 is not None:
            # Dual views
            z1 = self.projection_head(self.backbone(x1))
            z2 = self.projection_head(self.backbone(x2))
            return z1, z2
        else:
            # Single view
            return self.projection_head(self.backbone(x1))
    
    def get_features(self, x):
        """Returns original features (for downstream tasks)"""
        return self.backbone(x)
        

class SimCLR(nn.Module):
    """SimCLR Contrastive Learning Framework"""

    def __init__(self, backbone, projection_head, temperature=0.5):
        super().__init__()
        self.backbone = backbone  
        self.projection_head = projection_head
        self.temperature = temperature  

    def forward(self, x1, x2):
        """
        x1, x2: Two different augmented versions of the same image
        """
        # Extract features
        h1 = self.backbone(x1) 
        h2 = self.backbone(x2)

        # Project to contrastive space
        z1 = self.projection_head(h1) 
        z2 = self.projection_head(h2)

        return z1, z2

    def contrastive_loss(self, z1, z2):
        """NT-Xent loss (Normalized Temperature-scaled Cross Entropy)"""
        # Normalize
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)

        # Calculate similarity matrix
        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0) # [2B, D]
        similarity = torch.mm(z, z.T) / self.temperature # [2B, 2B]

        # Create labels (positive samples are on the diagonal)
        labels = torch.arange(batch_size).to(z.device) 
        labels = torch.cat([labels + batch_size, labels]) 


        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        similarity = similarity.masked_fill(mask, float('-inf'))

        # Calculate loss
        loss = nn.functional.cross_entropy(similarity, labels)

        return loss   