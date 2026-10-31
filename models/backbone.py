#!/usr/bin/env python
# coding: utf-8

import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights 


class ResNetBackbone(nn.Module):
    """ResNet series backbone network"""
    
    def __init__(self, model_name='resnet18', pretrained=True):
        super().__init__()
        if model_name == 'resnet18':
            # Use weights parameter instead of deprecated pretrained=True
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
        elif model_name == 'resnet50':
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Remove the final fully connected layer 
        # (FC layer is for classification, we need a feature extractor)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512 if model_name == 'resnet18' else 2048

    def forward(self, x):
        return self.features(x).flatten(1)

   
    @staticmethod
    def get_backbone(backbone_name='resnet18', pretrained=True):
        """Factory function, returns an instance of the backbone network"""
        return ResNetBackbone(backbone_name, pretrained)