"""
DeiT (Data-efficient Image Transformer) Model for Bean Leaf Classification
Using timm library
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import timm
from timm.utils import ModelEmaV2

# ===================== CONFIGURATION =====================
NUM_CLASSES = 3
IMG_SIZE = 384
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 5
PATIENCE = 10
LABEL_SMOOTHING = 0.1
EMA_DECAY = 0.9999
GRAD_CLIP = 1.0

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===================== TRAINING FUNCTIONS =====================
def train_one_epoch(model, model_ema, loader, criterion, optimizer, scheduler, device):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        optimizer.step()
        scheduler.step()
        model_ema.update(model)
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model on validation set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for inputs, labels in tqdm(loader, desc='Validating'):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels


# ===================== MODEL CREATION =====================
def create_deit_model(num_classes=NUM_CLASSES, pretrained=True):
    """Create DeiT model using timm"""
    model = timm.create_model(
        'deit3_small_patch16_384.fb_in1k',
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model
