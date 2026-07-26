"""
DeiT (Data-efficient Image Transformer) Model for Bean Leaf Classification
Using timm library
"""

import math

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import timm
from timm.utils import ModelEmaV2

from bean_leaf.config import DEFAULT_CONFIG

# ===================== CONFIGURATION =====================
# Đọc từ config.py trung tâm (Single Source of Truth) - đổi DEFAULT_CONFIG áp dụng ngay ở đây.
# IMG_SIZE=384 khớp sẵn với timm 'deit3_small_patch16_384'.
NUM_CLASSES = DEFAULT_CONFIG.num_classes
IMG_SIZE = DEFAULT_CONFIG.img_size
BATCH_SIZE = DEFAULT_CONFIG.batch_size
NUM_EPOCHS = DEFAULT_CONFIG.num_epochs
LEARNING_RATE = DEFAULT_CONFIG.learning_rate
WEIGHT_DECAY = DEFAULT_CONFIG.weight_decay
PATIENCE = DEFAULT_CONFIG.patience
LABEL_SMOOTHING = DEFAULT_CONFIG.label_smoothing
# Riêng của DeiT (không thuộc config chung): warmup + EMA
WARMUP_EPOCHS = 2
EMA_DECAY = 0.9998
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


def get_optimizer_scheduler(model, train_loader, num_epochs=NUM_EPOCHS, warmup_epochs=WARMUP_EPOCHS):
    """Create optimizer + warmup/cosine LR scheduler + EMA wrapper"""
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_steps = len(train_loader) * num_epochs
    warmup_steps = len(train_loader) * warmup_epochs

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    model_ema = ModelEmaV2(model, decay=EMA_DECAY, device=device)

    return criterion, optimizer, scheduler, model_ema
