#!/usr/bin/env python
# coding: utf-8

"""
EF Value Calibration Script
Maps the scores from the ranking model to EF values.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.metrics import r2_score


Image.MAX_IMAGE_PIXELS = None


from models.backbone import ResNetBackbone
from models import ContrastiveLearn
from EF_ranking_train import RankingModel, extract_features


def load_pretrained_backbone(checkpoint_path, device='cpu'):
    """Load pre-trained backbone"""
    backbone = ResNetBackbone.get_backbone('resnet18')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    backbone_state = {}
    for k, v in checkpoint.items():
        if k.startswith('backbone.'):
            backbone_state[k.replace('backbone.', '')] = v
    
    backbone.load_state_dict(backbone_state)
    backbone = backbone.to(device)
    backbone.eval()
    return backbone


def load_ranking_model(model_path, device='cpu'):
    """Load the trained ranking model"""
    model = RankingModel(input_dim=512, hidden_dim=256)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def get_all_image_paths(excel_path, image_dir):
    """Get all image paths from Excel"""
    df = pd.read_excel(excel_path)
    all_paths = []
    for filename in df['Filename']:
        # Try adding extensions
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
            img_path = os.path.join(image_dir, f"{filename}{ext}")
            if os.path.exists(img_path):
                all_paths.append(img_path)
                break
        else:
            print(f"Image not found: {filename}")
    return all_paths


def compute_scores(model, backbone, image_paths, device='cpu'):
    """Calculate ranking scores for all images"""
    scores = {}
    features = []
    valid_paths = []
    

    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                feat = backbone(img_tensor)
                feat = feat.cpu().numpy().flatten()
            
            features.append(feat)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"  Feature extraction failed: {os.path.basename(img_path)} - {e}")
    
 
    for img_path, feat in zip(valid_paths, features):
        feat_tensor = torch.tensor(feat, dtype=torch.float32).to(device)
        with torch.no_grad():
            score = model(feat_tensor).item()
        scores[img_path] = score
    
    return scores


def calibrate_with_ef_anchors(scores, excel_path, image_dir):
    """Linear calibration using EF value anchors"""
    df = pd.read_excel(excel_path)
    
    scores_list = []
    log_ef_list = []
    valid_files = []
    
    for _, row in df.iterrows():
        if pd.notna(row['Original EF Value']):
            filename = row['Filename']
            
            # Find corresponding image path
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
                test_path = os.path.join(image_dir, f"{filename}{ext}")
                if os.path.exists(test_path):
                    img_path = test_path
                    break
            
            if img_path and img_path in scores:
                score = scores[img_path]
                log_ef = np.log10(row['Original EF Value'])
                scores_list.append(score)
                log_ef_list.append(log_ef)
                valid_files.append(filename)
                print(f"  Anchor: {filename} -> score={score:.4f}, log10(EF)={log_ef:.2f}")
    
    if len(scores_list) < 3:
        print(f"Insufficient anchors. Only {len(scores_list)} EF values, at least 3 required.")
        return None, None, None, None
    
    # Linear fitting
    scores_array = np.array(scores_list)
    log_ef_array = np.array(log_ef_list)
    a, b = np.polyfit(scores_array, log_ef_array, 1)   
    
    # Calculate R²
    log_ef_pred = a * scores_array + b
    r2 = r2_score(log_ef_array, log_ef_pred) 
    
    print(f"\nCalibration Results:")
    print(f"  log10(EF) = {a:.4f} × score + {b:.4f}")
    print(f"  R² = {r2:.4f}")
    
    return a, b, r2, (scores_array, log_ef_array, valid_files)


def plot_calibration(scores_array, log_ef_array, a, b, r2, save_path='calibration_plot.png'):
    """Plot calibration curve"""
    plt.figure(figsize=(8, 6))
    
    # Scatter plot
    plt.scatter(scores_array, log_ef_array, color='blue', s=80, label='Anchor Data')
    

    score_range = np.array([min(scores_array), max(scores_array)])
    log_ef_fit = a * score_range + b
    plt.plot(score_range, log_ef_fit, color='red', linewidth=2, label=f'Fitted Line: y={a:.3f}x+{b:.3f}')
    
    plt.xlabel('score', fontsize=12)
    plt.ylabel('log10(EF)', fontsize=12)
    plt.title('EF Value Calibration Curve', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add R²
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()



def predict_ef_for_images(scores, a, b, output_csv='ef_predictions.csv'):
    """Predict EF values for all images and save"""
    results = []
    for img_path, score in scores.items():
        log_ef = a * score + b
        ef = 10 ** log_ef
        results.append({
            'filename': os.path.basename(img_path),
            'score': score,
            'log10_EF': log_ef,
            'EF': ef
        })
    

    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Prediction results saved: {output_csv}")
    

    print("\nPredictions (sorted from highest to lowest score):")

    for i, r in enumerate(results[:10]):
        print(f"{i+1}. {r['filename']}")
        print(f"   Score: {r['score']:.4f}")
        print(f"   Predicted EF: {r['EF']:.2e}")
        print(f"   log10(EF): {r['log10_EF']:.2f}")
        print()
    
    return results


def main():
    # ========== Configuration ==========
    checkpoint_path = './experiments/checkpoints/contrastive_epoch50.pth'
    ranking_model_path = './experiments/checkpoints/best_ranking_model.pth'
    image_dir = './sample_data/ranking_data/images'
    excel_path = './sample_data/labels/data_EF.xlsx'
    
    device = 'cpu'
    
    print("EF Value Calibration Script")
    
    # 1. Load models
    print("\n1. Loading models...")
    backbone = load_pretrained_backbone(checkpoint_path, device)
    ranking_model = load_ranking_model(ranking_model_path, device)
    
    # 2. Get all image paths
    print("\n2. Getting image paths...")
    image_paths = get_all_image_paths(excel_path, image_dir)
    print(f"Found {len(image_paths)} images")
    
    # 3. Calculate ranking scores
    print("\n3. Calculating ranking scores...")
    scores = compute_scores(ranking_model, backbone, image_paths, device)
    
    # 4. Calibrate with EF anchors
    print("\n4. Calibrating with EF value anchors...")
    result = calibrate_with_ef_anchors(scores, excel_path, image_dir)
    
    if result[0] is None:
        print("Calibration failed, please check EF value data")
        return
    
    a, b, r2, (scores_array, log_ef_array, valid_files) = result
    
    # 5. Plot calibration curve
    print("\n5. Plotting calibration curve...")
    plot_calibration(scores_array, log_ef_array, a, b, r2)
    
    # 6. Predict EF values for all images
    print("\n6. Predicting EF values for all images...")
    predict_ef_for_images(scores, a, b)
    
    print(f"Formula used: log10(EF) = {a:.4f} × score + {b:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()