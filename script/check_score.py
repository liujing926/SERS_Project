#!/usr/bin/env python
# coding: utf-8

# script/check_score.py
# Usage: python script/check_score.py

import sys
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from models import ContrastiveLearn
from models.backbone import ResNetBackbone


# ==================== RankingModel Definition (Consistent with training) ====================
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


# ==================== Combined Model ====================
class BackbonePlusRanking(nn.Module):
    def __init__(self, backbone, ranking_model):
        super().__init__()
        self.backbone = backbone
        self.ranking_model = ranking_model

    def forward(self, x):
        feat = self.backbone(x)
        score = self.ranking_model(feat)
        return score


def main():
    # 🔥 Update these paths to point to your actual checkpoints and test image
    contrastive_ckpt = './experiments/checkpoints/contrastive_epoch50.pth'
    ranking_ckpt = './experiments/checkpoints/best_ranking_model.pth'

    # 🔥 Change this to the image you want to test
    img_path = './sample_data/ranking_data/images/1_1.jpg'

    device = torch.device('cpu')

    # ---------- 1. Load Contrastive Learning Model ----------
    backbone = ResNetBackbone.get_backbone('resnet18')
    projection = ContrastiveLearn.ContrastiveHead(in_features=512, out_features=128)
    supcon_model = ContrastiveLearn.SupConModel(backbone, projection)

    ckpt = torch.load(contrastive_ckpt, map_location=device)
    supcon_model.load_state_dict(ckpt)
    supcon_model = supcon_model.to(device)
    supcon_model.eval()

    backbone = supcon_model.backbone
    backbone.eval()

    # ---------- 2. Load Ranking Model ----------
    ranking_model = RankingModel(input_dim=512)
    ranking_model.load_state_dict(torch.load(ranking_ckpt, map_location=device))
    ranking_model = ranking_model.to(device)
    ranking_model.eval()

    # ---------- 3. Combine Models ----------
    combined = BackbonePlusRanking(backbone, ranking_model).to(device)
    combined.eval()

    # ---------- 4. Preprocessing ----------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ---------- 5. Forward Pass and Print Score ----------
    if not os.path.exists(img_path):
        print(f"❌ Error: Test image not found at {img_path}")
        return

    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        score = combined(img_tensor).item()

    print(f"Image: {img_path}")
    print(f"Ranking Score: {score:.4f}")


if __name__ == '__main__':
    main()