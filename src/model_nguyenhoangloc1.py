"""
Custom VGG Model from Scratch for Bean Leaf Classification
PyTorch Implementation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import accuracy_score

# ===================== CONFIGURATION =====================
NUM_CLASSES = 3
IMG_SIZE = 400
BATCH_SIZE = 16
NUM_EPOCHS = 80
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.02
PATIENCE = 15
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 1.0

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===================== MODEL ARCHITECTURE =====================
class VGGBlock(nn.Module):
    """
    Block cơ bản của VGG: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool
    """
    def __init__(self, in_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.pool(x)
        return x


class BeanLeafVGG(nn.Module):
    """
    Custom VGG model cho Bean Leaf Classification.
    Kiến trúc dạng phễu: Tăng channels, giảm spatial size.
    Input: 3 x 400 x 400
    """
    def __init__(self, num_classes=3):
        super(BeanLeafVGG, self).__init__()
        
        # Block 1: 32 filters (Output: 200x200)
        self.block1 = VGGBlock(3, 32)
        
        # Block 2: 64 filters (Output: 100x100)
        self.block2 = VGGBlock(32, 64)
        
        # Block 3: 128 filters (Output: 50x50)
        self.block3 = VGGBlock(64, 128)
        
        # Block 4: 256 filters (Output: 25x25)
        self.block4 = VGGBlock(128, 256)
        
        # Block 5: 512 filters (Output: 12x12)
        self.block5 = VGGBlock(256, 512)
        
        # Classifier - Global Average Pooling thay vì FC lớn
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ===================== EARLY STOPPING =====================
class EarlyStopping:
    """Early stops training khi validation loss không cải thiện."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
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
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    """Validate model on validation set"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(all_labels)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels


# ===================== MODEL CREATION =====================
def create_vgg_model(num_classes=NUM_CLASSES):
    """Create BeanLeafVGG model"""
    model = BeanLeafVGG(num_classes=num_classes)
    return model


def get_optimizer_scheduler(model, train_loader, num_epochs=NUM_EPOCHS):
    """Create optimizer and scheduler"""
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=2e-3,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3  # Warm-up 30%
    )
    return criterion, optimizer, scheduler


def print_model_summary(model):
    """Print model parameter count"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tổng số tham số model: {total_params:,}")
    print(f"Số tham số trainable: {trainable_params:,}")
