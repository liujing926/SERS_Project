#!/usr/bin/env python
# coding: utf-8

# script/run_occlusion.py
# Usage: python script/run_occlusion.py

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from models import ContrastiveLearn
from models.backbone import ResNetBackbone

# RankingModel 
class RankingModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()

#  Combined Model 
class BackbonePlusRanking(nn.Module):
    def __init__(self, backbone, ranking_model):
        super().__init__()
        self.backbone = backbone
        self.ranking_model = ranking_model

    def forward(self, x):
        feat = self.backbone(x)
        score = self.ranking_model(feat)
        return score

#  Occlusion Core Function 
def occlusion_sensitivity(model, img_tensor, patch_size=32, stride=16,
                          baseline_value=0.0, device='cpu'):
    """
    model: BackbonePlusRanking, set to eval mode
    img_tensor: [1, C, H, W]
    patch_size: side length of the occluding patch
    stride: sliding step
    baseline_value: fill value for occlusion (0 corresponds to gray after normalization)
    Returns: sensitivity [H, W], larger value means more important
    """
    model.eval()
    H, W = img_tensor.shape[2], img_tensor.shape[3]

    with torch.no_grad():
        base_score = model(img_tensor).item()

    sensitivity = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    ys = list(range(0, H - patch_size + 1, stride))
    xs = list(range(0, W - patch_size + 1, stride))

    total = len(ys) * len(xs)
    done = 0

    for y in ys:
        for x in xs:
            occluded = img_tensor.clone()
            mean_val = img_tensor.mean().item()
            occluded[:, :, y:y+patch_size, x:x+patch_size] = mean_val

            with torch.no_grad():
                score = model(occluded).item()

            drop = base_score - score
            sensitivity[y:y+patch_size, x:x+patch_size] += drop
            count[y:y+patch_size, x:x+patch_size] += 1

            done += 1
            if done % 50 == 0:
                print(f"  Progress: {done}/{total}")

    sensitivity = sensitivity / np.maximum(count, 1)

    # Normalize to 0-1
    smin, smax = sensitivity.min(), sensitivity.max()
    if smax - smin > 1e-8:
        sensitivity = (sensitivity - smin) / (smax - smin)
    else:
        sensitivity = np.zeros_like(sensitivity)

    return sensitivity, base_score

#  Overlay and Save
def overlay_and_save(img_path, sensitivity, save_path, alpha=0.5):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    sens_resized = cv2.resize(sensitivity, (w, h), interpolation=cv2.INTER_CUBIC)
    sens_uint8 = np.uint8(255 * sens_resized)
    sens_color = cv2.applyColorMap(sens_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 1 - alpha, sens_color, alpha, 0)
    cv2.imwrite(save_path, overlay)
    print(f" Saved: {save_path}")

# Main Process 
def main():
  
    contrastive_ckpt = './experiments/checkpoints/contrastive_epoch50.pth'
    ranking_ckpt = './experiments/checkpoints/best_ranking_model.pth'


    img_path = './sample_data/ranking_data/images/1_1.jpg'

    save_dir = './experiments/occlusion'
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cpu')

    #  Load Models 
    backbone = ResNetBackbone.get_backbone('resnet18')
    projection = ContrastiveLearn.ContrastiveHead(in_features=512, out_features=128)
    supcon_model = ContrastiveLearn.SupConModel(backbone, projection)
    supcon_model.load_state_dict(torch.load(contrastive_ckpt, map_location=device))
    supcon_model = supcon_model.to(device).eval()
    backbone = supcon_model.backbone.eval()

    ranking_model = RankingModel(input_dim=512)
    ranking_model.load_state_dict(torch.load(ranking_ckpt, map_location=device))
    ranking_model = ranking_model.to(device).eval()

    combined = BackbonePlusRanking(backbone, ranking_model).to(device).eval()

    #  Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(img_path):
        print(f" Error: Image not found at {img_path}")
        return

    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Run Occlusion 
    print("Starting Occlusion analysis...")
    sensitivity, base_score = occlusion_sensitivity(
        model=combined,
        img_tensor=img_tensor,
        patch_size=8,
        stride=4,
        baseline_value=0.0,
        device=device
    )
    print(f"Base ranking score: {base_score:.4f}")

    #  Save Overlay 
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(save_dir, f'{base_name}_occlusion.png')
    overlay_and_save(img_path, sensitivity, save_path)

    #  Save Standalone Heatmap 
    plt.figure(figsize=(6, 5))
    plt.imshow(sensitivity, cmap='jet')
    plt.colorbar(label='Score drop')
    plt.title(f'Occlusion sensitivity (base score={base_score:.3f})')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{base_name}_occlusion_heat.png'), dpi=300)
    plt.close()

    print("\n Occlusion completed")

if __name__ == '__main__':
    main()