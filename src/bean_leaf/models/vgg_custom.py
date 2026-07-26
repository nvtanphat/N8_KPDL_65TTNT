"""
Custom VGG Model from Scratch for Bean Leaf Classification
PyTorch Implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import accuracy_score

from bean_leaf.config import DEFAULT_CONFIG
from bean_leaf.training.amp import autocast_context, get_scaler

# ===================== CONFIGURATION =====================
# Đọc từ config.py trung tâm (Single Source of Truth) - đổi DEFAULT_CONFIG áp dụng ngay ở đây.
NUM_CLASSES = DEFAULT_CONFIG.num_classes
IMG_SIZE = DEFAULT_CONFIG.img_size
BATCH_SIZE = DEFAULT_CONFIG.batch_size
NUM_EPOCHS = DEFAULT_CONFIG.num_epochs
LEARNING_RATE = DEFAULT_CONFIG.learning_rate
WEIGHT_DECAY = DEFAULT_CONFIG.weight_decay
PATIENCE = DEFAULT_CONFIG.patience
LABEL_SMOOTHING = DEFAULT_CONFIG.label_smoothing
GRAD_CLIP = 1.0  # Riêng của VGG - không thuộc config chung

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


# ===================== TRAINING FUNCTIONS =====================
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, scaler=None):
    """Train model for one epoch (AMP nếu có scaler - xem bean_leaf.training.amp)"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    autocast_ctx = autocast_context(device)

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast_ctx:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # cần unscale trước khi clip theo norm thật
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
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

    with torch.no_grad(), autocast_context(device):
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
