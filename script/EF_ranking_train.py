#!/usr/bin/env python
# coding: utf-8

"""
EF Value Ranking Learning Training Script
Trains a ranking model based on a pre-trained contrastive learning backbone.
"""
import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image          
Image.MAX_IMAGE_PIXELS = None  
from torchvision import transforms
import matplotlib.pyplot as plt

from models import ContrastiveLearn
from models.backbone import ResNetBackbone

# Load pre-trained backbone 
def load_pretrained_backbone(checkpoint_path, device='cpu'):
    """Load contrastive pre-trained backbone"""
    backbone = ResNetBackbone.get_backbone('resnet18')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    backbone_state = {}
    for k, v in checkpoint.items():
        if k.startswith('backbone.'):
            backbone_state[k.replace('backbone.', '')] = v
    
    backbone.load_state_dict(backbone_state)
    backbone = backbone.to(device)
    backbone.eval()
    
    print(f"Loaded pre-trained backbone: {checkpoint_path}")
    return backbone

# Extract features 
def extract_features(backbone, image_paths, device='cpu', max_size=1024):
    """Batch extract image features"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    features = []
    valid_paths = []
    
    print(f"Extracting features for {len(image_paths)} images...")
    
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert('RGB')
            width, height = image.size
            max_dim = max(width, height)
    
            if max_dim > max_size:
                ratio = max_size / max_dim
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                image = image.resize((new_width, new_height), Image.LANCZOS)
            
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                feat = backbone(img_tensor)
                feat = feat.cpu().numpy().flatten()
            
            features.append(feat)
            valid_paths.append(img_path)
            
        except Exception as e:
            print(f"  Failed: {os.path.basename(img_path)} - {e}")
    
    print(f"Successfully extracted {len(features)} feature vectors")
    return np.array(features), valid_paths

# Build ranking pairs from Excel 
def build_ranking_pairs(excel_path, image_dir):
    """
    Build ranking comparison pairs from Excel
    Excel columns: Filename, Group ID, In-group Rank (1=highest)
    """
    df = pd.read_excel(excel_path)
    
    # Check required columns
    required_cols = ['Filename', 'Group ID', 'In-group Rank']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Excel missing column: {col}")
    
    groups = df.groupby('Group ID')
    
    pairs = []  
    ef_values = {}  
    
    for group_name, group in groups:
        # Higher number = better, sort descending
        group_sorted = group.sort_values('In-group Rank', ascending=False)
        imgs = []  
        
        for _, row in group_sorted.iterrows():
            name = row['Filename']
            # Try multiple extensions
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.tif']:
                test_path = os.path.join(image_dir, f"{name}{ext}")
                if os.path.exists(test_path):
                    img_path = test_path
                    break
                    
            if img_path:
                imgs.append(img_path)
            else:
                print(f"Image not found for {name}")
        
        # Build comparison pairs
        for i in range(len(imgs)):
            for j in range(i+1, len(imgs)):
                pairs.append((imgs[i], imgs[j], 1))  # A is better than B
        
        # Record EF values (using the English column name)
        if 'Original EF Value' in df.columns:
            for _, row in group.iterrows():
                if pd.notna(row['Original EF Value']):
                    ef_values[row['Filename']] = row['Original EF Value']
    
    print(f"Constructed {len(pairs)} comparison pairs")
    print(f"Images with EF values: {len(ef_values)}")
    return pairs, ef_values

# Ranking Dataset
class RankingDataset(Dataset):
    """Ranking learning dataset"""
    def __init__(self, pairs, feature_dict):
        self.data = []
        for img_a, img_b, label in pairs:
            if img_a in feature_dict and img_b in feature_dict:
                self.data.append({
                    'feat_a': feature_dict[img_a],
                    'feat_b': feature_dict[img_b],
                    'label': label
                })
        print(f"Dataset contains {len(self.data)} valid pairs")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return (torch.tensor(item['feat_a'], dtype=torch.float32),
                torch.tensor(item['feat_b'], dtype=torch.float32),
                torch.tensor(item['label'], dtype=torch.float32))

# Ranking Model 
class RankingModel(nn.Module):
    """Ranking model: inputs features, outputs ranking score"""
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

# Ranking Loss 
class RankingLoss(nn.Module):
    """Pairwise Ranking Loss"""
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, score_a, score_b, label):
        diff = score_a - score_b
        
        loss = torch.where(
            label == 1,
            torch.relu(-diff + self.margin), 
            torch.relu(diff + self.margin)    
        )
        
        return loss.mean()

# Train Ranking Model 
def train_ranking_model(feature_dict, pairs, epochs=100, lr=0.001):
    """Train ranking model"""
    valid_pairs = []
    for img_a, img_b, label in pairs:
        if img_a in feature_dict and img_b in feature_dict:
            valid_pairs.append((img_a, img_b, label))
    
    print(f"Valid comparison pairs: {len(valid_pairs)} / {len(pairs)}")
    
    if len(valid_pairs) == 0:
        print("No valid pairs available")
        return None, []

    random.seed(42)
    random.shuffle(valid_pairs)
    
    n_pairs = len(valid_pairs)
    n_train = int(0.8 * n_pairs)
    train_pairs = valid_pairs[:n_train]
    val_pairs = valid_pairs[n_train:]

    print(f"Total pairs: {len(valid_pairs)}")
    print(f"Train set: {len(train_pairs)} pairs")
    print(f"Val set: {len(val_pairs)} pairs")
    
    train_dataset = RankingDataset(train_pairs, feature_dict)
    val_dataset = RankingDataset(val_pairs, feature_dict)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    model = RankingModel(input_dim=512)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = RankingLoss(margin=0.5)
    
  
    print("Starting ranking model training")

    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    

    os.makedirs('./experiments/checkpoints', exist_ok=True)
    best_model_path = './experiments/checkpoints/best_ranking_model.pth'
    
    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        for feat_a, feat_b, label in train_loader:
            score_a = model(feat_a)
            score_b = model(feat_b)
            loss = criterion(score_a, score_b, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validate
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for feat_a, feat_b, label in val_loader:
                score_a = model(feat_a)
                score_b = model(feat_b)
                loss = criterion(score_a, score_b, label)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    print("Training complete")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Plot loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Ranking Model Training Loss')
    plt.savefig('./experiments/ranking_loss_curve.png', dpi=150)
    plt.show()
    
    return model, val_pairs

# Evaluate Ranking Accuracy 
def evaluate_ranking_accuracy(model, feature_dict, pairs):
    """Evaluate ranking accuracy"""
    model.eval()
    correct = 0
    total = 0
    wrong_cases = []
    
    with torch.no_grad():
        for img_a, img_b, label in pairs:
            if img_a in feature_dict and img_b in feature_dict:
                feat_a = torch.tensor(feature_dict[img_a], dtype=torch.float32)
                feat_b = torch.tensor(feature_dict[img_b], dtype=torch.float32)
                
                score_a = model(feat_a).item()
                score_b = model(feat_b).item()
                
                pred = 1 if score_a > score_b else 0
                if pred == label:
                    correct += 1
                else:
                    wrong_cases.append({
                        'img_a_name': os.path.basename(img_a),
                        'img_b_name': os.path.basename(img_b),
                        'true_label': label,
                        'pred_label': pred,
                        'score_a': score_a,
                        'score_b': score_b,
                    })
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n Ranking Accuracy: {correct}/{total} = {accuracy:.4f}")

    if wrong_cases:
        print(f"\n Wrong cases (Total {len(wrong_cases)}):")
        for i, case in enumerate(wrong_cases[:5], 1): # Only print first 5 to avoid spamming
            print(f"Case {i}: A={case['img_a_name']}, B={case['img_b_name']} | True: {'A' if case['true_label']==1 else 'B'} | Pred: {'A' if case['pred_label']==1 else 'B'}")
    else:
        print("\n All predictions correct!")
    
    return accuracy, wrong_cases

if __name__ == '__main__':

    checkpoint_path = './experiments/checkpoints/contrastive_final.pth'
    image_dir = './sample_data/ranking_data/images'
    excel_path = './sample_data/labels/data_EF.xlsx'
    
    device = 'cpu'
    

    backbone = load_pretrained_backbone(checkpoint_path, device)
    

    pairs, ef_values = build_ranking_pairs(excel_path, image_dir)
    

    all_image_paths = set()    
    for img_a, img_b, _ in pairs:
        all_image_paths.add(img_a)
        all_image_paths.add(img_b)
    all_image_paths = list(all_image_paths)
    
    print(f"\nImages requiring feature extraction: {len(all_image_paths)}")
    

    features, valid_paths = extract_features(backbone, all_image_paths, device)
    

    feature_dict = {path: feat for path, feat in zip(valid_paths, features)}
    

    valid_pairs = []
    for img_a, img_b, label in pairs:
        if img_a in feature_dict and img_b in feature_dict:
            valid_pairs.append((img_a, img_b, label))
    
    print(f"Valid comparison pairs: {len(valid_pairs)} / {len(pairs)}")
    

    model, val_pairs = train_ranking_model(feature_dict, pairs, epochs=100, lr=0.001)
    

    import pickle
    with open('./experiments/train_val_pairs.pkl', 'wb') as f:
        pickle.dump({'val_pairs': val_pairs}, f)
    

    evaluate_ranking_accuracy(model, feature_dict, val_pairs)
    
    print("\nBest model saved in experiments/checkpoints/best_ranking_model.pth")