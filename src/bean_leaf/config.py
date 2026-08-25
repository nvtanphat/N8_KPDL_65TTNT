"""
Centralized Configuration - Single Source of Truth cho hyperparameter dùng chung
giữa 5 model classification (BeanLeafLite/EfficientNet-B0/MobileNetV3/ResNet50/ShuffleNetV2).

Đổi 1 giá trị ở đây áp dụng ngay cho toàn bộ model đọc từ DEFAULT_CONFIG.

Mọi model dùng CHUNG toàn bộ giá trị dưới đây và chung 1 recipe optimizer
(AdamW + CosineAnnealingLR) - không model nào có ngoại lệ. Đây là điều kiện để
bảng benchmark là so sánh kiến trúc, chứ không phải so sánh "model nào được tune kỹ hơn".
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class BenchmarkConfig:
    num_classes: int = 3

    # img_size=384: giữ chi tiết vết bệnh nhỏ (đốm góc lá, gỉ sắt) sắc nét hơn.
    img_size: int = 384
    batch_size: int = 32
    # Ngân sách epoch CỐ ĐỊNH, không dừng sớm: CosineAnnealingLR dùng T_max=num_epochs nên
    # LR chỉ anneal về ~0 nếu chạy đủ số epoch này. Trước đây num_epochs=100 làm trần còn
    # EarlyStopping cắt ở epoch 14-37 -> LR mới đi được 14-37% chu kỳ, đứng gần như nguyên
    # ở giá trị ban đầu, và mỗi model bị cắt ở một chỗ khác nhau -> so sánh lệch. 40 epoch
    # là đủ rộng: lần chạy trước mọi model đều đã hội tụ (early-stop) trong khoảng 14-37.
    num_epochs: int = 40
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    # patience=0 = TẮT dừng sớm. Checkpoint vẫn được chọn theo internal-val loss thấp nhất,
    # chỉ là không cắt ngang lịch LR nữa. Đặt >0 để bật lại (dùng khi thử nghiệm nhanh).
    patience: int = 0
    label_smoothing: float = 0.05

    imagenet_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    imagenet_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


DEFAULT_CONFIG = BenchmarkConfig()
