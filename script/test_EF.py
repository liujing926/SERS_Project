#!/usr/bin/env python
# coding: utf-8

"""
Final Model Testing Script
Test set: Images not involved in training/calibration.
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Import models
from models.backbone import ResNetBackbone
from EF_ranking_train import RankingModel, extract_features


def load_models(checkpoint_path, ranking_model_path, device='cpu'):
    """Load pre-trained backbone and ranking model"""
    
    # Load backbone
    backbone = ResNetBackbone.get_backbone('resnet18')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    backbone_state = {}
    for k, v in checkpoint.items():
        if k.startswith('backbone.'):
            backbone_state[k.replace('backbone.', '')] = v
    
    backbone.load_state_dict(backbone_state)
    backbone = backbone.to(device)
    backbone.eval()
    print(f"Loaded backbone: {checkpoint_path}")
    
    # Load ranking model
    ranking_model = RankingModel(input_dim=512, hidden_dim=256)
    ranking_model.load_state_dict(torch.load(ranking_model_path, map_location=device))
    ranking_model = ranking_model.to(device)
    ranking_model.eval()
    print(f"Loaded ranking model: {ranking_model_path}")
    
    return backbone, ranking_model


def load_calibration_params(calibration_file='calibration_params.csv'):
    """Load calibration parameters (a, b)"""
    df = pd.read_csv(calibration_file)
    a = df['a'].values[0]
    b = df['b'].values[0]
    print(f"Loaded calibration parameters: log10(EF) = {a:.4f} × score + {b:.4f}")
    return a, b


def predict_batch_image(backbone, ranking_model, image_paths, a, b, device='cpu'):
    """Predict EF values for a batch of images"""
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    results = []
    
    for img_path in image_paths:

        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
    

        with torch.no_grad():
            feat = backbone(img_tensor)
            feat = feat.cpu().numpy().flatten()
    

        feat_tensor = torch.tensor(feat, dtype=torch.float32).to(device)
        with torch.no_grad():
            score = ranking_model(feat_tensor).item()
    

        log10_ef = a * score + b
        ef = 10 ** log10_ef
        
        results.append({
            'filename': os.path.basename(img_path),
            'score': score,
            'log10_ef': log10_ef,
            'ef': ef
        })
    
    return results
    

def test_ranking_pairs(test_pairs_excel, image_dir, backbone, ranking_model, a, b, device='cpu'):
    """Test pairwise ranking capability"""
    df = pd.read_excel(test_pairs_excel)
    
    all_images = set()
    for _, row in df.iterrows():
        all_images.add(f"{row['Image A']}.jpg")
        all_images.add(f"{row['Image B']}.jpg")
    
    image_paths = [os.path.join(image_dir, img) for img in all_images]
    
    batch_results = predict_batch_image(backbone, ranking_model, image_paths, a, b, device)
    score_dict = {r['filename']: r['score'] for r in batch_results}
    
    correct = 0
    total = 0

    print("Pairwise Ranking Test")

    
    for _, row in df.iterrows():
        img_a = f"{row['Image A']}.jpg"
        img_b = f"{row['Image B']}.jpg"
        true_label = row['True Result']  # 1: A better than B
        
        score_a = score_dict.get(img_a, -999)
        score_b = score_dict.get(img_b, -999)
        
        pred_label = 1 if score_a > score_b else 0
        is_correct = (pred_label == true_label)
        
        if is_correct:
            correct += 1
        total += 1
        
        print(f"\n{img_a} vs {img_b}")
        print(f"  True: {'A better' if true_label == 1 else 'B better'}")
        print(f"  Pred: {'A better' if pred_label == 1 else 'B better'}")
        print(f"  Result: {'Correct' if is_correct else 'Wrong'}")
        print(f"  score_A: {score_a:.4f}, score_B: {score_b:.4f}")
    
    accuracy = correct / total if total > 0 else 0

    print(f"Ranking Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    return accuracy


def test_model_batch(test_excel, image_dir, backbone, ranking_model, a, b, device='cpu'):
    """Test model performance on test set (batch version)"""
    df = pd.read_excel(test_excel)
    
    valid_rows = []
    image_paths = []
    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, f"{row['Filename']}.jpg")
        if os.path.exists(img_path):
            image_paths.append(img_path)
            valid_rows.append(row)
        else:
            print(f"Image not found: {img_path}")
    
    if not image_paths:
        print("No test images found")
        return []
    
    batch_results = predict_batch_image(backbone, ranking_model, image_paths, a, b, device)
    
    results = []

    print("Test Set Prediction Results")

    
    all_errors = []
    
    for row, pred in zip(valid_rows, batch_results):
        filename = row['Filename']
        score = pred['score']
        log10_ef = pred['log10_ef']
        ef = pred['ef']
        
        result = {
            'filename': filename,
            'score': score,
            'pred_log10_EF': log10_ef,
            'pred_EF': ef
        }
        
        # Calculate error if true EF is available
        if 'Original EF Value' in df.columns and pd.notna(row['Original EF Value']):
            true_ef = row['Original EF Value']
            true_log10 = np.log10(true_ef)
            error_log10 = abs(true_log10 - log10_ef)
            error_percent = abs(ef - true_ef) / true_ef * 100
            
            result['true_EF'] = true_ef
            result['true_log10_EF'] = true_log10
            result['error_log10'] = error_log10
            result['error_percent'] = error_percent
            all_errors.append(error_log10)
            
            print(f"\n {filename}")
            print(f"   Score: {score:.4f}")
            print(f"   Pred EF: {ef:.2e} (log10: {log10_ef:.2f})")
            print(f"   True EF: {true_ef:.2e} (log10: {true_log10:.2f})")
            print(f"   Error: {error_log10:.2f} log10 units ({error_percent:.1f}%)")
        else:
            print(f"\n{filename}")
            print(f"   Score: {score:.4f}")
            print(f"   Pred EF: {ef:.2e} (log10: {log10_ef:.2f})")
        
        results.append(result)
    
    if all_errors:

        print("Test Set Statistics")

        print(f"Mean Absolute Error: {np.mean(all_errors):.3f} log10 units")
        print(f"Max Error: {np.max(all_errors):.3f} log10 units")
        print(f"Min Error: {np.min(all_errors):.3f} log10 units")
    
    return results


def visualize_test_results(results, save_path='test_results.png'):
    """Visualize test results"""
    
    plt.figure(figsize=(10, 6))
    
    if 'true_log10_EF' in results[0]:
        true_vals = [r['true_log10_EF'] for r in results]
        pred_vals = [r['pred_log10_EF'] for r in results]
        
        plt.scatter(true_vals, pred_vals, color='blue', s=80)
        
        min_val = min(min(true_vals), min(pred_vals))
        max_val = max(max(true_vals), max(pred_vals))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Line')
        
        plt.xlabel('True log10(EF)', fontsize=12)
        plt.ylabel('Predicted log10(EF)', fontsize=12)
        plt.title('Test Set EF Prediction Results', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        for i, r in enumerate(results):
            plt.annotate(r['filename'].replace('.jpg', '')[:10], 
                        (r['true_log10_EF'], r['pred_log10_EF']),
                        fontsize=8)
    
    else:
        filenames = [r['filename'].replace('.jpg', '')[:15] for r in results]
        scores = [r['score'] for r in results]
        log10_efs = [r['pred_log10_EF'] for r in results]
        
        x = np.arange(len(filenames))
        width = 0.35
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        ax1.bar(x - width/2, scores, width, label='Ranking Score', color='blue')
        ax1.set_xlabel('Test Images', fontsize=12)
        ax1.set_ylabel('Ranking Score', fontsize=12, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        ax2 = ax1.twinx()
        ax2.bar(x + width/2, log10_efs, width, label='log10(EF)', color='orange')
        ax2.set_ylabel('log10(EF)', fontsize=12, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(filenames, rotation=45, ha='right')
        plt.title('Test Set Prediction Results', fontsize=14)
        fig.tight_layout()
    
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Test results plot saved: {save_path}")


def save_calibration_params(a, b, save_path='calibration_params.csv'):
    """Save calibration parameters"""
    df = pd.DataFrame({'a': [a], 'b': [b]})
    df.to_csv(save_path, index=False)
    print(f" Calibration parameters saved: {save_path}")


def main():
    # ========== Configuration ==========
    checkpoint_path = './experiments/checkpoints/contrastive_epoch50.pth'
    ranking_model_path = './experiments/checkpoints/best_ranking_model.pth'
    
    # Test set configuration
    test_excel = './sample_data/labels/data_EF.xlsx'
    test_image_dir = './sample_data/ranking_data/images'
    
    # Pairwise ranking test configuration (Optional)
    test_pairs_excel = './sample_data/labels/test_pairs.xlsx'
    
    device = 'cpu'
    
    # Calibration parameters (from previous calibration results)
    a = 1.3560
    b = 4.9766
    

    print("Final Model Testing")

    backbone, ranking_model = load_models(checkpoint_path, ranking_model_path, device)
    
    save_calibration_params(a, b)
    

    if os.path.exists(test_excel):
        results = test_model_batch(test_excel, test_image_dir, backbone, ranking_model, a, b, device)
        visualize_test_results(results)
        
   
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv('test_results.csv', index=False)
            print(f"Test results saved: test_results.csv")
    else:
        print(f"Test set Excel not found: {test_excel}")
    

    if os.path.exists(test_pairs_excel):
        accuracy = test_ranking_pairs(test_pairs_excel, test_image_dir, backbone, ranking_model, a, b, device)
        print(f"\nFinal ranking accuracy: {accuracy:.2%}")
    else:
        print(f"Ranking test Excel not found: {test_pairs_excel}")


if __name__ == '__main__':
    main()