"""
ResNet50 Model for Bean Leaf Classification
PyTorch Implementation with Transfer Learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from bean_leaf.config import DEFAULT_CONFIG
from bean_leaf.training.amp import autocast_context

# ===================== CONFIGURATION =====================
NUM_CLASSES = DEFAULT_CONFIG.num_classes
IMG_SIZE = DEFAULT_CONFIG.img_size
BATCH_SIZE = DEFAULT_CONFIG.batch_size
NUM_EPOCHS = DEFAULT_CONFIG.num_epochs
LEARNING_RATE = DEFAULT_CONFIG.learning_rate
WEIGHT_DECAY = DEFAULT_CONFIG.weight_decay
PATIENCE = DEFAULT_CONFIG.patience
LABEL_SMOOTHING = DEFAULT_CONFIG.label_smoothing

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===================== TRAINING FUNCTIONS =====================
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train model for one epoch (AMP nếu có scaler)"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    autocast_ctx = autocast_context(device)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast_ctx:
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Validate model on validation set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(), autocast_context(device):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


# ===================== MODEL CREATION =====================
def create_resnet50_model(num_classes=NUM_CLASSES, pretrained=True):
    """Create ResNet50 model with custom classifier"""
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    
    return model


def get_optimizer_scheduler(model, num_epochs=NUM_EPOCHS, lr=None):
    """Create optimizer and scheduler"""
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=lr or LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    return criterion, optimizer, scheduler
