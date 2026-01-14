"""
Bean Leaf Classification - Main Training Script
Hỗ trợ nhiều model: VGG, EfficientNet, DeiT, MobileNet
"""

import argparse
import os

# ===================== IMPORT MODELS =====================
# PyTorch Models
from model_nguyenhoangloc1 import (
    create_vgg_model, 
    train_one_epoch as train_vgg, 
    validate as validate_vgg,
    get_optimizer_scheduler as get_vgg_optim,
    EarlyStopping,
    device
)
from model_nguyenhoangloc2 import (
    create_efficientnet_model,
    train_one_epoch as train_effnet,
    validate as validate_effnet,
    get_optimizer_scheduler as get_effnet_optim
)
from model_nguyenvantanphat2 import (
    create_deit_model,
    train_one_epoch as train_deit,
    validate as validate_deit
)

# Shared modules
from preprocessing import create_df
from evaluation import evaluate_model, plot_roc_auc
import eda

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ===================== CONFIGURATION =====================
CONFIG = {
    'vgg': {'IMG_SIZE': 400, 'EPOCHS': 80, 'BATCH_SIZE': 16},
    'efficientnet': {'IMG_SIZE': 300, 'EPOCHS': 30, 'BATCH_SIZE': 16},
    'deit': {'IMG_SIZE': 384, 'EPOCHS': 50, 'BATCH_SIZE': 16},
    # 'mobilenet' sử dụng TensorFlow, không tương thích với main.py (PyTorch)
}
NUM_CLASSES = 3
OUTPUT_DIR = r'D:\DataMining\DoAnFinal\results'


# ===================== DATA TRANSFORMS =====================
def get_transforms(img_size):
    """Get train and val transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def get_dataloaders(train_dir, val_dir, img_size, batch_size):
    """Create train and val dataloaders"""
    train_transform, val_transform = get_transforms(img_size)
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader


# ===================== TRAINING FUNCTIONS =====================
def train_model(model_name, train_loader, val_loader, model, output_dir):
    """Train a specific model"""
    # Tạo folder riêng cho từng model
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    model_path = os.path.join(model_output_dir, f'best_{model_name}_model.pth')
    
    # Check if model exists -> skip
    if os.path.exists(model_path):
        print(f"[SKIP] Model '{model_name}' already exists at {model_path}")
        model.load_state_dict(torch.load(model_path))
        return model
    
    print(f"\n[TRAIN] Training {model_name}...")
    print("=" * 60)
    
    config = CONFIG[model_name]
    epochs = config['EPOCHS']
    
    # Get optimizer based on model type
    if model_name == 'vgg':
        criterion, optimizer, scheduler = get_vgg_optim(model, train_loader, epochs)
        train_fn, val_fn = train_vgg, validate_vgg
        use_scheduler_step = True  # OneCycleLR
    elif model_name == 'efficientnet':
        criterion, optimizer, scheduler = get_effnet_optim(model)
        train_fn, val_fn = train_effnet, validate_effnet
        use_scheduler_step = False  # ReduceLROnPlateau
    elif model_name == 'deit':
        import torch.nn as nn
        import torch.optim as optim
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
        scheduler = None
        from timm.utils import ModelEmaV2
        model_ema = ModelEmaV2(model, decay=0.9999, device=device)
        train_fn, val_fn = train_deit, validate_deit
        use_scheduler_step = False
    else:
        print(f"Unknown model: {model_name}")
        return None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=10, verbose=True, path=model_path)
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 40)
        
        # Train
        if model_name == 'deit':
            train_loss, train_acc = train_fn(model, model_ema, train_loader, criterion, optimizer, scheduler, device)
            val_loss, val_acc, _, _ = val_fn(model_ema.module, val_loader, criterion, device)
        elif model_name == 'vgg':
            train_loss, train_acc = train_fn(model, train_loader, criterion, optimizer, scheduler, device)
            val_loss, val_acc, _, _ = val_fn(model, val_loader, criterion, device)
        else:
            train_loss, train_acc = train_fn(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = val_fn(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        # Scheduler step
        if scheduler and not use_scheduler_step:
            scheduler.step(val_loss)
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("\nEarly stopping triggered!")
            break
    
    print(f"\n[SAVED] Model saved to {model_path}")
    return model


# ===================== MAIN =====================
def main():
    parser = argparse.ArgumentParser(description='Bean Leaf Classification')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--model', type=str, choices=['vgg', 'efficientnet', 'deit', 'all'], 
                        default='all', help='Model to train')
    parser.add_argument('--eda', action='store_true', help='Run EDA before training')
    args = parser.parse_args()
    
    # Setup directories
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # EDA (optional)
    if args.eda:
        print("\n=== Running EDA ===")
        train_df = create_df(train_dir)
        eda.plot_class_distribution(train_df, title='Training Set Distribution')
    
    # Models to train
    models_to_train = ['vgg', 'efficientnet', 'deit'] if args.model == 'all' else [args.model]
    
    for model_name in models_to_train:
        config = CONFIG[model_name]
        
        # Get dataloaders
        train_loader, val_loader = get_dataloaders(
            train_dir, val_dir, 
            config['IMG_SIZE'], 
            config['BATCH_SIZE']
        )
        
        # Create model
        if model_name == 'vgg':
            model = create_vgg_model(NUM_CLASSES).to(device)
        elif model_name == 'efficientnet':
            model = create_efficientnet_model(NUM_CLASSES).to(device)
        elif model_name == 'deit':
            model = create_deit_model(NUM_CLASSES).to(device)
        
        # Train
        model = train_model(model_name, train_loader, val_loader, model, OUTPUT_DIR)
        
        print(f"\n{model_name.upper()} completed!")
    
    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETED!")
    print("=" * 60)


if __name__ == '__main__':
    main()
