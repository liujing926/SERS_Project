#!/usr/bin/env python
# coding: utf-8

# Data processing
import pandas as pd
import numpy as np
import glob

# Image processing
from PIL import Image
import os

# PyTorch related packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights 


class EFDataset(Dataset):
    """EF image dataset, supports both labeled and unlabeled modes"""

    def __init__(self, image_dir, excel_path=None, label_dir=None, transform=None, is_train=True, is_labeled=True):
        """
        Args:
            image_dir: Image directory
            label_dir: Label directory (if any)
            transform: Data augmentation
            is_train: Training mode flag
            is_labeled: Whether labels are available (True: labeled, False: unlabeled - for contrastive learning)
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.is_train = is_train
        self.is_labeled = is_labeled

        # Read Excel file
        if excel_path is None:
 
            if 'contrastive' in image_dir:
                excel_path = './sample_data/contrastive_data/unlabeled_data.xlsx'
            elif 'ranking' in image_dir:
                excel_path = './sample_data/ranking_data/data_EF.xlsx'
            else:
                # Fallback for general cases
                excel_path = './sample_data/labels/data_EF.xlsx'
                
        self.df = pd.read_excel(excel_path)
        print(f"Total rows in Excel: {len(self.df)}")
        print(f"Columns: {self.df.columns.tolist()}")

        # Create mapping dictionary from filename to label

        if 'Filename' in self.df.columns:
            name_col = 'Filename'
        else:
            raise ValueError("Column 'Filename' not found in the Excel file.")

        if 'Roughness (1-3)' in self.df.columns:
            label_col = 'Roughness (1-3)'
        elif 'In-group Rank' in self.df.columns:
            label_col = 'In-group Rank'
        else:
            raise ValueError("Target label column not found in dataset.")

        self.label_dict = dict(zip(self.df[name_col], self.df[label_col]))

        # Get all image paths
        self.image_paths = []
        self.labels = []

        # Support multiple image formats
        self.image_paths = sorted(
            glob.glob(os.path.join(image_dir, "*.jpg")) + 
            glob.glob(os.path.join(image_dir, "*.jpeg")) +
            glob.glob(os.path.join(image_dir, "*.png")) +
            glob.glob(os.path.join(image_dir, "*.tif"))
        )
        print(f"Found {len(self.image_paths)} images in folder.")

        # Load labels based on flag
        if self.is_labeled:
            self._load_labels_from_excel()
        else:
            # Unlabeled mode (for contrastive learning)
            self.labels = [-1] * len(self.image_paths)
            print("Unlabeled mode: Loading images only for contrastive learning.")

    def _load_labels_from_excel(self):
        """Load labels from excel (keep only images that exist in excel)"""
        self.image_paths_with_labels = []
        self.labels = []
        missing_in_excel = 0

        for img_path in self.image_paths:
            # Get filename (without path and extension)
            filename = os.path.basename(img_path)
 
            name_without_ext = os.path.splitext(filename)[0]

            # Check if it exists in excel
            if name_without_ext in self.label_dict:
                self.image_paths_with_labels.append(img_path)
                self.labels.append(self.label_dict[name_without_ext])
            else:
                missing_in_excel += 1

        # Replace with labeled image paths
        self.image_paths = self.image_paths_with_labels

        print(f"Successfully loaded {len(self.image_paths)} labeled images.")
        print(f"Label distribution: {pd.Series(self.labels).value_counts().sort_index().to_dict()}")

        if missing_in_excel > 0:
            print(f"Warning: {missing_in_excel} images not found in Excel, filtered out.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
    
        if self.transform:
            image = self.transform(image)
    
        if self.is_labeled:
            label = self.labels[idx]
            return image, label
        else:
            return image


def get_test_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning: returns two augmented versions of the same image"""
    
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.labels = labels
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform twice to get two augmented views
        img1 = self.transform(image)
        img2 = self.transform(image)
        
        if self.labels is not None:
            label = self.labels[idx]
            return img1, img2, label
        else:
            return img1, img2