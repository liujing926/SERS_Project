#!/usr/bin/env python
# coding: utf-8

import os
import glob
import yaml
import pandas as pd
import torch
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import your models
from models import ContrastiveLearn
from models.backbone import ResNetBackbone
from data.dataset import EFDataset, ContrastiveDataset
from data.transforms import get_test_transform
from utils.evaluate_with_classifier import evaluate_with_classifier

import warnings
warnings.filterwarnings('ignore')

# Load config
config_path = './config/config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)


def batch_test(test_dir, model, ref_features, ref_labels, ref_names, device='cpu'):
    """Batch test all images in a folder"""
    test_paths = glob.glob(os.path.join(test_dir, "*.jpg"))
    test_paths += glob.glob(os.path.join(test_dir, "*.png"))
    
    print(f"\nFound {len(test_paths)} test images\n")
    print("="*60)
    
    results = []
    
    for i, img_path in enumerate(test_paths):
        filename = os.path.basename(img_path)
        test_features = extract_features(model, img_path, device)
        
        pred_label, top_k_labels, top_k_sims, top_k_idx = predict_roughness(
            test_features, ref_features, ref_labels, k=5
        )
        
        top_k_names = [ref_names[idx] for idx in top_k_idx]
        
        results.append({
            'top_k_names': top_k_names,  
            'filename': filename,
            'img_path': img_path,
            'pred_label': pred_label,
            'top_k_labels': top_k_labels,
            'top_k_sims': top_k_sims,
            'top_k_idx': top_k_idx
        })
        
        roughness_map = {1: 'Smooth', 2: 'Medium', 3: 'Rough'}
        print(f"{i+1}. {filename}")
        print(f"   Predicted Roughness: {pred_label} ({roughness_map[pred_label]})")
        print(f"   Top 5 most similar reference images:")
        for j, (idx, label, sim) in enumerate(zip(top_k_idx, top_k_labels, top_k_sims)):
            print(f"      {j+1}. {ref_names[idx]} (Roughness: {label} - {roughness_map[label]}, Similarity: {sim:.3f})")
        print()
    
    print("="*80)
    print(f"✅ Batch testing completed! Tested {len(results)} images.")
    
    return results


def visualize_batch_results(results, ref_dir, save_dir='./test_results'):
    """Visualize batch test results (saving each image individually)"""
    os.makedirs(save_dir, exist_ok=True)
    roughness_map = {1: 'Smooth', 2: 'Medium', 3: 'Rough'}
    
    for result in results:
        img = Image.open(result['img_path'])
        plt.figure(figsize=(12, 4))
        
        # Test image
        plt.subplot(1, 4, 1)
        plt.imshow(img)
        plt.title(f"Test Image\nPred: {roughness_map[result['pred_label']]}")
        plt.axis('off')
        
        # Top 3 reference images
        for i, (label, sim, name) in enumerate(zip(
            result['top_k_labels'][:3], 
            result['top_k_sims'][:3],
            result['top_k_names'][:3]   
        )):
            ref_img_path = os.path.join(ref_dir, result['top_k_names'][i])
            ref_img = Image.open(ref_img_path)
            
            plt.subplot(1, 4, i+2)
            plt.imshow(ref_img)
            plt.title(f"Ref {i+1}\n{roughness_map[label]}\nSim: {sim:.3f}")
            plt.axis('off')
        
        plt.tight_layout()
        save_name = result['filename'].replace('.jpg', '_result.png').replace('.png', '_result.png')
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    print(f"✅ Visualization results saved to: {save_dir}")


def evaluate_classification(model, dataloader, device):
    """Evaluate roughness classification accuracy"""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features = model.backbone(images)
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    if len(all_features) == 0:
        print("Error: No features extracted")
        return 0.0
    
    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels)
    print(f"Test set feature shape: {features.shape}, label shape: {labels.shape}")
    
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(features, labels)
    pred = knn.predict(features)
    acc = accuracy_score(labels, pred)
    
    return acc


def load_model(checkpoint_path, device='cpu'):
    """Load trained model"""
    backbone = ResNetBackbone.get_backbone(config['model']['backbone'])
    projection = ContrastiveLearn.ContrastiveHead(
        in_features=512 if config['model']['backbone'] == 'resnet18' else 2048,
        out_features=config['model']['embedding_dim']
    )
    model = ContrastiveLearn.SupConModel(backbone, projection)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded: {checkpoint_path}")
    return model


def extract_features(model, image_path, device='cpu', max_size=1024):
    """Extract feature vector for a single image"""
    try:
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        max_dim = max(width, height)
    
        if max_dim > max_size:
            ratio = max_size / max_dim
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height), Image.LANCZOS)
    except Exception as e:
        print(f"❌ Failed to open/resize image {image_path}: {e}")
        return np.zeros(512)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model.backbone(image_tensor)
        features = features.cpu().numpy().flatten()
    
    return features


def load_reference_features(data_dir, excel_path, model, device='cpu'):
    """Load features and labels of all images in the training set as references"""
    df = pd.read_excel(excel_path)
    # 🔥 Updated column name to English
    label_dict = dict(zip(df['Filename'], df['Roughness (1-3)']))
    
    image_paths = glob.glob(os.path.join(data_dir, "*.jpg"))
    image_paths += glob.glob(os.path.join(data_dir, "*.png"))
    
    features_list = []
    labels_list = []
    names_list = []
    
    print(f"Loading reference set images...")
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        name = filename.replace('norm_', '').replace('.jpg', '').replace('.png', '')
        if name in label_dict:
            features = extract_features(model, img_path, device)
            features_list.append(features)
            labels_list.append(label_dict[name])
            names_list.append(filename)
    
    print(f"✅ Loaded {len(features_list)} reference images.")
    return np.array(features_list), np.array(labels_list), names_list


def predict_roughness(test_features, ref_features, ref_labels, k=5):
    """Predict roughness using KNN"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarities = cosine_similarity(test_features.reshape(1, -1), ref_features)[0]
    top_k_idx = np.argsort(similarities)[-k:][::-1]
    top_k_labels = ref_labels[top_k_idx]
    top_k_sims = similarities[top_k_idx]
    
    unique_labels = np.unique(top_k_labels)
    vote_weights = {}
    for label in unique_labels:
        mask = top_k_labels == label
        vote_weights[label] = top_k_sims[mask].sum()
    
    pred_label = max(vote_weights, key=vote_weights.get)
    
    return pred_label, top_k_labels, top_k_sims, top_k_idx


if __name__ == '__main__':
    # ========== Configuration ==========
    # 🔥 Updated to relative paths
    checkpoint_path = './experiments/checkpoints/contrastive_epoch50.pth'
    
    # Training data directory (Reference set)
    data_dir = './sample_data/contrastive_data/images'
    excel_path = './sample_data/labels/unlabeled_data.xlsx'
    
    # Test images directory
    test_dir = './sample_data/ranking_data/images'
    
    device = 'cpu' 
    
    # 1. Load model
    model = load_model(checkpoint_path, device)
    
    # 2. Load reference set
    ref_features, ref_labels, ref_names = load_reference_features(data_dir, excel_path, model, device)
    
    # 3. Batch test
    results = batch_test(test_dir, model, ref_features, ref_labels, ref_names, device)

    # 4. Roughness accuracy test
    test_dataset = EFDataset(
        image_dir=test_dir,
        excel_path='./sample_data/labels/data_EF.xlsx',  # 🔥 Updated to data_EF.xlsx
        is_labeled=True,  
        transform=get_test_transform()  
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )

    test_accuracy = evaluate_classification(model, test_loader, device)
    print(f"Test set roughness accuracy: {test_accuracy:.4f}")
    
    # 5. Output results
    roughness_map = {1: 'Smooth', 2: 'Medium', 3: 'Rough'}
    print("\n" + "="*50)
    print(f"📊 Prediction Results")
    print("="*50)
    
    # 6. Visualize
    visualize_batch_results(results, data_dir, save_dir='./test_results')