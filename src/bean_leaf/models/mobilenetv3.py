"""
MobileNetV3Large Model for Bean Leaf Classification
PyTorch Implementation - End-to-End Fine-Tuning chuẩn, y hệt EfficientNet-B3
(AdamW + CosineAnnealingLR, full fine-tune từ epoch 1, không đóng băng backbone).

Trước đây dùng transfer learning 2-phase (freeze rồi unfreeze dần) - bỏ đi vì đó là
1 recipe train riêng biệt cho MobileNetV3 mà 3 model kia không có, khiến so sánh
giữa các kiến trúc trong Controlled Benchmark Protocol không thực sự công bằng.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from bean_leaf.config import DEFAULT_CONFIG
from bean_leaf.training.amp import autocast_context

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

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===================== MODEL CREATION =====================
def create_mobilenetv3_model(num_classes=NUM_CLASSES, pretrained=True):
    """
    MobileNetV3-Large pretrained ImageNet + head phân loại DÙNG CHUNG với EfficientNet-B0
    và ResNet50: Dropout(0.3) -> Linear(in_features -> num_classes).

    Trước đây model này có head riêng (Linear(960->256) -> BatchNorm1d -> SiLU -> Dropout(0.3)
    -> Linear(256->3)), tức thêm hẳn một tầng ẩn 256 chiều + BatchNorm mà 3 model pretrained
    kia không có (~0.24M params, 7.6% tổng params của nó). Trong khi đó MobileNetV3 lại xếp
    hạng 1 ở benchmark - không loại trừ được khả năng thứ hạng đó đến từ phần head chứ không
    phải từ backbone. Đã gỡ để bảng benchmark so sánh đúng kiến trúc backbone.
    """
    weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)

    # Full fine-tune (không đóng băng backbone) - torchvision pretrained model
    # đã có requires_grad=True mặc định cho mọi param.

    in_features = model.classifier[0].in_features  # 960
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


# ===================== TRAINING FUNCTIONS =====================
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train model for one epoch (AMP nếu có scaler - xem bean_leaf.training.amp)"""
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

    return running_loss / total, correct / total


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

    return running_loss / total, correct / total


def get_optimizer_scheduler(model, num_epochs=NUM_EPOCHS, lr=None):
    """Create optimizer and scheduler - y hệt efficientnet.get_optimizer_scheduler"""
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=lr or LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    return criterion, optimizer, scheduler
