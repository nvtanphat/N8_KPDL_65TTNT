"""
MobileNetV3Large Model for Bean Leaf Classification
PyTorch Implementation - transfer learning 2 giai đoạn (thay cho bản TensorFlow/Keras cũ):
  Phase 1: đóng băng backbone, chỉ train classification head.
  Phase 2: mở khóa phần lớn backbone (trừ BatchNorm) để fine-tune với LR thấp hơn.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from bean_leaf.training.early_stopping import EarlyStopping

# ===================== CONFIGURATION =====================
NUM_CLASSES = 3
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 30
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 20
PHASE1_LR = 5e-4
PHASE2_LR = 1e-5
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.0
PATIENCE = 7
FREEZE_RATIO = 0.7  # Phase 2: giữ đóng băng 70% block đầu của backbone, chỉ fine-tune phần cuối

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===================== MODEL CREATION =====================
def create_mobilenetv3_model(num_classes=NUM_CLASSES, pretrained=True):
    """MobileNetV3Large pre-trained ImageNet + custom head (GAP -> Dense(256) -> BN -> SiLU -> Dropout -> Dense)"""
    weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)

    in_features = model.classifier[0].in_features  # 960
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.SiLU(),  # Swish
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


def freeze_backbone(model, freeze=True):
    """Phase 1: đóng băng toàn bộ backbone (model.features), chỉ train head"""
    for param in model.features.parameters():
        param.requires_grad = not freeze


def unfreeze_backbone_for_finetune(model, freeze_ratio=FREEZE_RATIO):
    """
    Phase 2: mở khóa phần cuối backbone để fine-tune, giữ đóng băng các block đầu
    (feature tổng quát, ít cần học lại) và LUÔN đóng băng BatchNorm2d (giữ running
    stats đã học từ ImageNet, tránh optimizer phá vỡ khi batch nhỏ).
    """
    blocks = list(model.features.children())
    freeze_until = int(len(blocks) * freeze_ratio)

    for i, block in enumerate(blocks):
        requires_grad = i >= freeze_until
        for param in block.parameters():
            param.requires_grad = requires_grad

    for module in model.features.modules():
        if isinstance(module, nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = False


def _freeze_bn_eval(model):
    """
    model.train() ở đầu mỗi epoch sẽ đưa TẤT CẢ submodule (kể cả BatchNorm2d đã đóng
    băng) về train mode, khiến running_mean/var bị cập nhật lại dù đã set
    requires_grad=False. Phải gọi lại hàm này sau mỗi model.train() để giữ các BN đã
    đóng băng ở eval mode (không cập nhật running stats, không học gamma/beta).
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and not any(p.requires_grad for p in module.parameters()):
            module.eval()


# ===================== TRAINING FUNCTIONS =====================
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    _freeze_bn_eval(model)
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

    return running_loss / total, correct / total


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

    return running_loss / total, correct / total


def _run_phase(model, train_loader, val_loader, device, epochs, lr, early_stopping, phase_name):
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7)

    for epoch in range(epochs):
        print(f"\n[{phase_name}] Epoch {epoch + 1}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"[{phase_name}] Early stopping triggered!")
            break


def train_mobilenetv3(model, train_loader, val_loader, model_path, device):
    """
    Chạy đủ 2 phase transfer learning, lưu checkpoint tốt nhất vào model_path.
    Trả về model đã load lại best weights.
    """
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=model_path)

    print("\n[MobileNetV3] PHASE 1: Training classification head (backbone frozen)")
    freeze_backbone(model, freeze=True)
    _run_phase(model, train_loader, val_loader, device, PHASE1_EPOCHS, PHASE1_LR, early_stopping, "Phase 1")

    if not early_stopping.early_stop:
        print("\n[MobileNetV3] PHASE 2: Fine-tuning (mở khóa phần cuối backbone, BN vẫn đóng băng)")
        unfreeze_backbone_for_finetune(model, freeze_ratio=FREEZE_RATIO)
        early_stopping.counter = 0  # reset để phase 2 có đủ patience riêng
        early_stopping.early_stop = False
        _run_phase(model, train_loader, val_loader, device, PHASE2_EPOCHS, PHASE2_LR, early_stopping, "Phase 2")

    model.load_state_dict(torch.load(model_path, map_location=device))
    return model
