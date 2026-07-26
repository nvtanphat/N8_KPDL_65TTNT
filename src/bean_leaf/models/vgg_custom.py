"""
BeanLeafLite - CNN nhẹ tự thiết kế từ đầu (from scratch, không pretrained) cho Bean Leaf
Classification. Thay thế kiến trúc BeanLeafVGG cũ (5 block conv3x3 "dày", ~4.7M params)
bằng khối Depthwise-Separable + Residual + SE Attention (kiểu MBConv của MobileNetV2/V3):
  - Depthwise-separable conv: tách 1 conv3x3 dày thành depthwise (lọc theo từng kênh) +
    pointwise 1x1 (trộn kênh) - giảm params rất nhiều so với conv3x3 thường ở cùng
    số kênh input/output.
  - Residual connection: cộng thẳng input vào output mỗi block - giúp gradient chảy tốt
    hơn khi train from-scratch (không có pretrained weight để "khởi động" tốt).
  - SE (Squeeze-and-Excitation) Attention: học trọng số quan trọng của từng kênh đặc
    trưng (channel-wise), giúp model "chú ý" vào các kênh mang thông tin đốm bệnh thay vì
    coi mọi kênh ngang nhau - bù lại phần capacity đã cắt giảm.
Kết quả: nhẹ hơn cả MobileNetV3-Large (~3.2M) trong khi vẫn giữ residual+attention để
bù đắp cho việc giảm params, không pretrained.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import accuracy_score

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
GRAD_CLIP = 1.0  # Riêng của BeanLeafLite (OneCycleLR + grad clip) - không thuộc config chung
# OneCycleLR cần biết TRƯỚC tổng số epoch để tính lịch anneal LR về gần 0 ở epoch cuối.
# Nhưng NUM_EPOCHS=100 chỉ là trần tối đa - EarlyStopping (patience=7) thực tế luôn dừng
# sớm hơn nhiều (quan sát thực nghiệm: dừng quanh epoch 40-60). Nếu cấu hình OneCycleLR
# cho 100 epoch, lịch LR bị "cắt ngang" giữa chừng lúc early-stop - LR vẫn còn cao, chưa
# kịp anneal thấp - khiến checkpoint cuối cùng hội tụ kém hơn hẳn so với các model dùng
# CosineAnnealingLR (ít nhạy với việc dừng sớm hơn). Đặt riêng một mốc epoch thực tế hơn
# cho OneCycleLR để nó có cơ hội hoàn thành chu kỳ anneal trước khi patience kích hoạt.
ONECYCLE_TARGET_EPOCHS = 50

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===================== MODEL ARCHITECTURE =====================
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation: nén mỗi feature map thành 1 số (GAP) rồi học 1 cổng
    (bottleneck FC -> sigmoid) quyết định kênh nào quan trọng hơn, nhân ngược lại vào
    feature map gốc. Chi phí params rất rẻ (2*C*C/reduction) so với lợi ích mang lại.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced, kernel_size=1)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        s = self.pool(x)
        s = self.act(self.fc1(s))
        s = self.gate(self.fc2(s))
        return x * s


class LiteResidualBlock(nn.Module):
    """
    Depthwise-separable conv + Residual connection + SE Attention (kiểu MBConv).
    Depthwise (conv3x3 riêng từng kênh, groups=in_channels) + pointwise (1x1 trộn kênh)
    rẻ hơn hẳn 1 conv3x3 "dày" trên toàn bộ in/out channels như VGGBlock cũ.
    Cần shortcut 1x1 (projection) khi đổi số kênh hoặc downsample (stride=2); nếu giữ
    nguyên kích thước/số kênh thì cộng thẳng input (identity) không tốn thêm tham số.
    """
    def __init__(self, in_channels, out_channels, stride=1, se_reduction=2):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.se = SEBlock(out_channels, reduction=se_reduction)

        needs_projection = stride != 1 or in_channels != out_channels
        if needs_projection:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.act(self.bn1(self.depthwise(x)))
        out = self.bn2(self.pointwise(out))
        out = self.se(out)

        return self.act(out + identity)


class BeanLeafLite(nn.Module):
    """
    CNN nhẹ tự thiết kế cho Bean Leaf Classification (from scratch, không pretrained).
    Input: 3 x 384 x 384 (DEFAULT_CONFIG.img_size). Stem giảm size 1 lần, 5 stage
    LiteResidualBlock mỗi stage downsample x2 (tổng x64), rồi head mở rộng channel +
    GAP + classifier 2 lớp FC. ~0.9-1.1M tham số tuỳ đúng con số khi khởi tạo (in bằng
    print_model_summary), nhẹ hơn MobileNetV3-Large (~3.2M).
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )

        # (out_channels, số block trong stage, stride của block đầu tiên trong stage)
        stage_configs = [
            (48, 2, 2),
            (72, 2, 2),
            (120, 3, 2),
            (192, 3, 2),
            (320, 2, 2),
        ]
        stages = []
        in_channels = 32
        for out_channels, num_blocks, first_stride in stage_configs:
            for i in range(num_blocks):
                stride = first_stride if i == 0 else 1
                stages.append(LiteResidualBlock(in_channels, out_channels, stride=stride))
                in_channels = out_channels
        self.stages = nn.Sequential(*stages)

        # Head: mở rộng channel trước GAP (giống MobileNetV3) để tăng capacity cho
        # classifier mà không cần thêm block conv nào nữa.
        head_channels = in_channels * 2
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.SiLU(inplace=True),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(head_channels, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
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
            try:
                scheduler.step()
            except ValueError:
                # OneCycleLR đã hết total_steps (huấn luyện chạy dài hơn ONECYCLE_TARGET_EPOCHS
                # dự kiến) - giữ nguyên LR đã anneal thấp ở bước cuối, không cần step tiếp.
                pass

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
def create_lite_model(num_classes=NUM_CLASSES):
    """Create BeanLeafLite model"""
    model = BeanLeafLite(num_classes=num_classes)
    return model


def get_optimizer_scheduler(model, train_loader, num_epochs=NUM_EPOCHS):
    """Create optimizer and scheduler"""
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    onecycle_epochs = min(num_epochs, ONECYCLE_TARGET_EPOCHS)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=2e-3,
        epochs=onecycle_epochs,
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
