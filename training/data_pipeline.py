import os
import cv2
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
import torch

class DermDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms: A.Compose = None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def apply_clahe(self, image):
        # CLAHE Contrast Enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(image.shape) == 3 and image.shape[2] == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            image = clahe.apply(image)
        return image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_path = row['image_path']
        label = row['label']

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply CLAHE
        image = self.apply_clahe(image)

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']

        return image, label

def get_transforms(image_size=224):
    """
    Image preprocessing:
    - Resize 224x224
    - Normalize
    - Albumentations augmentations
    """
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return train_transform, val_transform

def create_dataloaders(df: pd.DataFrame, batch_size=32, num_workers=4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Implements Stratified train/val/test split and Class weighting + Imbalance Correction via Sampler
    Assumes `df` contains columns: 'image_path', 'label', 'skin_tone' (for Fitzpatrick scale audit)
    """
    # Combine datasets (HAM10000, ISIC, PAD-UFES-20 logic) is assumed built in `df`
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 80/20 train/test split. Val split could just be inside the fold.
    # For demonstration, doing one split simply:
    train_idx, test_idx = next(skf.split(df, df['label']))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_test_df = df.iloc[test_idx].reset_index(drop=True)

    # Split test into val/test
    val_idx, final_test_idx = next(StratifiedKFold(n_splits=2, shuffle=True, random_state=42).split(temp_test_df, temp_test_df['label']))
    val_df = temp_test_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_test_df.iloc[final_test_idx].reset_index(drop=True)
    
    # Audit distributions per Fitzpatrick scale
    print("Skin Tone Distribution (Train):", train_df['skin_tone'].value_counts())
    
    # Class weighting sampler for imbalance correction
    class_counts = train_df['label'].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in train_df['label']]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_tfms, val_tfms = get_transforms()

    train_ds = DermDataset(train_df, transforms=train_tfms)
    val_ds = DermDataset(val_df, transforms=val_tfms)
    test_ds = DermDataset(test_df, transforms=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
