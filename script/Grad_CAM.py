#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Import your classes
from models import ContrastiveLearn
from models.backbone import ResNetBackbone

#  Ranking Model 
class RankingModel(nn.Module):
    """Ranking model: takes features as input, outputs a ranking score"""
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  
            nn.ReLU(),                        
            nn.Dropout(0.3),                 
            nn.Linear(hidden_dim, 1)            
        )
    
    def forward(self, x):
        return self.net(x).squeeze()     # Output: [batch_size, 1]

# Combined Model 
class BackbonePlusRanking(nn.Module):
    """Combines the contrastive backbone and ranking head for Grad-CAM"""
    def __init__(self, backbone, ranking_model):
        super().__init__()
        self.backbone = backbone
        self.ranking_model = ranking_model

    def forward(self, x):
        feat = self.backbone(x)            # [B, 512]
        score = self.ranking_model(feat)   # [B]
        return score

# Grad-CAM 
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, img_tensor):
        """
        img_tensor: [1, C, H, W]
        Returns: heatmap [H, W] numpy array, normalized to 0-1
        """
        self.model.eval()
        img_tensor = img_tensor.requires_grad_(True)

        score = self.model(img_tensor)   # [1]
        self.model.zero_grad()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.squeeze().cpu().numpy()
        return cam

# Overlay Heatmap 
def overlay_heatmap(img_path, heatmap, save_path, alpha=0.3):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Cannot read image: {img_path}")
        return

    h, w = img.shape[:2]                                  
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    cv2.imwrite(save_path, overlay)
    print(f" Saved heatmap: {save_path}")

# Main Process 
def main():
 
    contrastive_ckpt = './experiments/checkpoints/contrastive_epoch50.pth'
    ranking_ckpt = './experiments/checkpoints/best_ranking_model.pth'

    image_paths = [
        './sample_data/ranking_data/images/1_1.jpg',
        './sample_data/ranking_data/images/1_2.jpg',
        './sample_data/ranking_data/images/1_3.jpg',
        './sample_data/ranking_data/images/1_4.jpg'
    ]

    save_dir = './experiments/gradcam'
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cpu')

    # Rebuild Contrastive Model 
    backbone = ResNetBackbone.get_backbone('resnet18')
    projection = ContrastiveLearn.ContrastiveHead(in_features=512, out_features=128)
    supcon_model = ContrastiveLearn.SupConModel(backbone, projection)

    ckpt = torch.load(contrastive_ckpt, map_location=device)
    supcon_model.load_state_dict(ckpt)
    supcon_model = supcon_model.to(device)
    print(f"Loaded contrastive model: {contrastive_ckpt}")

    backbone = supcon_model.backbone
    backbone.eval()

    # Rebuild Ranking Model 
    ranking_model = RankingModel(input_dim=512)
    ranking_model.load_state_dict(torch.load(ranking_ckpt, map_location=device))
    ranking_model = ranking_model.to(device)
    ranking_model.eval()
    print(f"Loaded ranking model: {ranking_ckpt}")

    # Combine 
    combined = BackbonePlusRanking(backbone, ranking_model).to(device)
    target_layer = combined.backbone.features[5][-1]
    cam = GradCAM(combined, target_layer)

    #Preprocessing 
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Generate Heatmaps Loop 
    for i, img_path in enumerate(image_paths):
        if not os.path.exists(img_path):
            print(f" Image not found: {img_path}")
            continue

        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        heatmap = cam.generate(img_tensor)

        base = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(save_dir, f'{base}_cam.png')
        overlay_heatmap(img_path, heatmap, save_path)

    print("\nAll heatmaps generated successfully")

if __name__ == '__main__':
    main()