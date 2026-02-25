import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
import torchvision.transforms as transforms
import timm
import torch.nn.functional as F

class HAM10000Dataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label = row['label']
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for missing/corrupted an image
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha 
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def main():
    metadata_path = "/Users/lingeshwaran/Desktop/HAM10000/HAM10000_metadata.tab"
    images_dir = "/Users/lingeshwaran/Desktop/HAM10000/images"

    if not os.path.exists(metadata_path):
        print(f"Error: Could not find {metadata_path}")
        return

    print("Loading metadata...")
    # Load using pandas (tab-separated)
    df = pd.read_csv(metadata_path, sep='\t')

    # Map image_id to corresponding image files (.jpg extension is standard for HAM10000)
    df['image_path'] = df['image_id'].apply(lambda x: os.path.join(images_dir, f"{x}.jpg"))

    # Encode dx column into integer labels
    labels = df['dx'].unique()
    label_mapping = {label: idx for idx, label in enumerate(labels)}
    df['label'] = df['dx'].map(label_mapping)
    num_classes = len(labels)
    
    mel_index = label_mapping.get('mel', -1)
    
    print(f"Classes found: {label_mapping}")
    if mel_index != -1:
        print(f"Melanoma class index mapping: {mel_index}")
    else:
        print("Warning: 'mel' class not found in dx column.")

    print("Splitting dataset...")
    # Stratified 80/10/10 split
    train_val_df, test_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
    # To get 10% of total validation size from the 90% train_val pool, 0.1/0.9 ≈ 0.1111
    train_df, val_df = train_test_split(train_val_df, test_size=0.1111, stratify=train_val_df['label'], random_state=42)

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    # Image size 224x224 and data augmentation (random flip + rotation)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = HAM10000Dataset(train_df, images_dir=images_dir, transform=train_transforms)
    val_dataset = HAM10000Dataset(val_df, images_dir=images_dir, transform=val_test_transforms)
    test_dataset = HAM10000Dataset(test_df, images_dir=images_dir, transform=val_test_transforms)

    num_workers = 4 # Might need to reduce to 0 if macOS multiprocess crashes M1/M2 chips 
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers)

    # Use Apple MPS device if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Creating EfficientNet-B0 model via timm...")
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
    model = model.to(device)

    # Use Focal Loss (gamma=2) to handle class imbalance
    criterion = FocalLoss(gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    num_epochs = 15
    best_val_auc = 0.0
    save_path = "ham10000_model.pth"

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, labels_batch in train_loader:
            images, labels_batch = images.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels_batch in val_loader:
                images, labels_batch = images.to(device), labels_batch.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item() * images.size(0)
                
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        
        # Metrics Tracking
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        val_recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # Tracking specific 'mel' (melanoma) recall
        if mel_index != -1:
            recalls = recall_score(all_labels, all_preds, average=None, zero_division=0)
            mel_recall = recalls[mel_index] if len(recalls) > mel_index else 0.0
        else:
            mel_recall = 0.0
            
        try:
            val_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        except ValueError:
            val_auc = 0.0
            
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Accuracy: {val_accuracy:.4f} | Prec: {val_precision_macro:.4f} | Recall: {val_recall_macro:.4f}")
        print(f"  Melanoma Recall: {mel_recall:.4f} | AUC Score: {val_auc:.4f}")
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print(f"  => Saved new best model to {save_path}!")
            
    print("-" * 50)
    print("Training completed. Evaluating best model on purely unseen Test set...")
    model.load_state_dict(torch.load(save_path))
    model.eval()

    test_preds = []
    test_labels = []

    with torch.no_grad():
        for images, labels_batch in test_loader:
            images, labels_batch = images.to(device), labels_batch.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels_batch.cpu().numpy())

    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)

    print("\nClassification Report:")
    target_names = [str(k) for k, v in sorted(label_mapping.items(), key=lambda item: item[1])]
    print(classification_report(test_labels, test_preds, target_names=target_names, zero_division=0))

if __name__ == '__main__':
    main()
