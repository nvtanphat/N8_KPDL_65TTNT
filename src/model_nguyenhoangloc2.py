"""
EfficientNetB3 Model for Bean Leaf Classification
PyTorch Implementation with Transfer Learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# ===================== CONFIGURATION =====================
NUM_CLASSES = 3
IMG_SIZE = 300
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
PATIENCE = 5

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===================== EARLY STOPPING =====================
class EarlyStopping:
    """Early stops training khi validation loss không cải thiện."""
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# ===================== TRAINING FUNCTIONS =====================
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
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
    
    with torch.no_grad():
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
def create_efficientnet_model(num_classes=NUM_CLASSES, pretrained=True):
    """Create EfficientNetB3 model with custom classifier"""
    model = models.efficientnet_b3(pretrained=pretrained)
    
    # Freeze base layers (optional)
    # for param in model.features.parameters():
    #     param.requires_grad = False
    
    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    return model


def get_optimizer_scheduler(model, learning_rate=LEARNING_RATE):
    """Create optimizer and scheduler"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    return criterion, optimizer, scheduler
