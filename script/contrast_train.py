#!/usr/bin/env python
# coding: utf-8

import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml

# Add project root to path
project_root = os.path.dirname(os.getcwd())
if project_root not in sys.path:
    sys.path.append(project_root)

from data.dataset import EFDataset, ContrastiveDataset
from data.transforms import get_contrastive_transform
from models.backbone import ResNetBackbone
from models import ContrastiveLearn 
from models.Supervised_contrastive_loss import SupConLossWithTwoViews
from utils.logger import setup_logger
from utils.helpers import set_seed
from utils.similarity_caculation import compute_similarity_stats, evaluate_model, visualize_feature_space
from utils.evaluate_with_classifier import evaluate_with_classifier


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Create generator
g = torch.Generator()
g.manual_seed(42)


def main():
    # Use relative path for config
    config_path = './config/config.yaml'
    print(f"Using configuration file: {config_path}")

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Create necessary directories
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    os.makedirs('./experiments/tsne', exist_ok=True)  # Directory for visualization

    # Set random seed
    set_seed(42)

    # Setup logger
    logger = setup_logger(config['paths']['log_dir'])
    logger.info("Starting contrastive learning training")

    # Create dataset
    # 1. Use EFDataset to load image paths and roughness labels
    path_dataset = EFDataset(
        image_dir=config['paths']['data_dir'],
        is_labeled=True, 
        transform=None
    )
    train_size = int(0.8 * len(path_dataset))
    val_size = len(path_dataset) - train_size

    train_dataset, val_dataset = random_split(path_dataset, [train_size, val_size]) 

    print(f"Train set: {train_size}, Val set: {val_size}")
    
    # Get labels
    train_labels = [path_dataset.labels[i] for i in train_dataset.indices]
    val_labels = [path_dataset.labels[i] for i in val_dataset.indices]

    # 2. Use ContrastiveDataset to generate dual-view augmentations
    contrastive_dataset_train = ContrastiveDataset(
        image_paths=[path_dataset.image_paths[i] for i in train_dataset.indices], 
        labels=train_labels, 
        transform=get_contrastive_transform()
    )
    contrastive_dataset_val = ContrastiveDataset(
        image_paths=[path_dataset.image_paths[i] for i in val_dataset.indices],
        labels=val_labels, 
        transform=get_contrastive_transform()
    )
    contrastive_dataset_val_no_labels = ContrastiveDataset(
        image_paths=[path_dataset.image_paths[i] for i in val_dataset.indices],
        labels=None, 
        transform=get_contrastive_transform()
    ) 
    
    dataloader_train = DataLoader(
        contrastive_dataset_train,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        worker_init_fn=seed_worker, 
        generator=g  
    )
    dataloader_val = DataLoader(
        contrastive_dataset_val,
        batch_size=config['data']['batch_size'],
        shuffle=False, # Changed to False for validation
        num_workers=config['data']['num_workers'],
        worker_init_fn=seed_worker,
        generator=g 
    )
    dataloader_val_no_labels = DataLoader(
        contrastive_dataset_val_no_labels,
        batch_size=config['data']['batch_size'],
        shuffle=False, # Changed to False for validation
        num_workers=config['data']['num_workers'],
        worker_init_fn=seed_worker,
        generator=g  
    )

    # Create model
    backbone = ResNetBackbone.get_backbone(config['model']['backbone'])
    projection = ContrastiveLearn.ContrastiveHead(
        in_features=512 if config['model']['backbone'] == 'resnet18' else 2048,
        out_features=config['model']['embedding_dim']
    )  
    model = ContrastiveLearn.SupConModel(backbone, projection)
    
    device = torch.device('cpu')
    # device = torch.device('cpu') # Keep this if you strictly want to run on CPU
    model = model.to(device)

    # Optimizer and loss function
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['contrastive']['learning_rate'],
        weight_decay=config['contrastive']['weight_decay']
    )
    criterion = SupConLossWithTwoViews(temperature=config['contrastive']['temperature'])
    
    # Contrastive learning training loop
    epochs = config['contrastive']['epochs'] 
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for img1, img2, labels in dataloader_train: 
            img1 = img1.to(device)  
            img2 = img2.to(device)  
            labels = labels.to(device)
            
            z1, z2 = model(img1, img2)  
            loss = criterion(z1, z2, labels) 
            optimizer.zero_grad()  
            loss.backward() 
            optimizer.step() 

            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader_train)  
        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")

        # Save model
        if (epoch + 1) % 1 == 0:
            checkpoint_path = os.path.join(config['paths']['checkpoint_dir'], f'contrastive_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)

        # ========== Validation Phase ==========
        if (epoch + 1) % 1 == 0 or epoch == epochs - 1: 
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for img1, img2, labels in dataloader_val:
                    img1 = img1.to(device)
                    img2 = img2.to(device)
                    labels = labels.to(device)
                    z1, z2 = model(img1, img2)
                    loss = criterion(z1, z2, labels)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(dataloader_val)
            
            pos_sims, neg_sims = compute_similarity_stats(model, dataloader_val_no_labels, device)
        
            print(f"\n📊 Epoch {epoch+1}/{epochs}")
            print(f"   Train Loss: {avg_loss:.4f}")
            print(f"   Val Loss: {avg_val_loss:.4f}")  
            print(f"   Positive Similarity: {pos_sims.mean():.4f} ± {pos_sims.std():.4f}")
            print(f"   Negative Similarity: {neg_sims.mean():.4f} ± {neg_sims.std():.4f}")
            print(f"   Gap: {pos_sims.mean() - neg_sims.mean():.4f}")
            
            pos_mean, neg_mean = evaluate_model(model, dataloader_val, device, epoch+1)
            logger.info(f"  - Intra-class Similarity: {pos_mean:.4f}")
            logger.info(f"  - Inter-class Similarity: {neg_mean:.4f}")
            logger.info(f"  - Gap: {pos_mean - neg_mean:.4f}")
            
            # Visualize feature space
            score = visualize_feature_space(model, dataloader_val, epoch+1, device)
            logger.info(f"Epoch {epoch+1}: Silhouette Score={score:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            accuracy = evaluate_with_classifier(model, dataloader_val, device, epoch, logger)
            
    # Save final model after training
    final_path = os.path.join(config['paths']['checkpoint_dir'], 'contrastive_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"✅ Final model saved: {final_path}")
    logger.info("Training completed")


if __name__ == '__main__':
    main()