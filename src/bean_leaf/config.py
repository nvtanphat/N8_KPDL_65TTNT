"""
Centralized Configuration - Single Source of Truth cho hyperparameter dùng chung
giữa 3 model classification (VGG/EfficientNet/MobileNetV3).

Đổi 1 giá trị ở đây áp dụng ngay cho toàn bộ model đọc từ DEFAULT_CONFIG.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class BenchmarkConfig:
    num_classes: int = 3

    # img_size=384: giữ chi tiết vết bệnh nhỏ (đốm góc lá, gỉ sắt) sắc nét hơn.
    img_size: int = 384
    batch_size: int = 32
    # Trần epoch cao, để EarlyStopping (patience) tự quyết định dừng sớm thay vì
    # giới hạn cứng - model nào cần nhiều epoch hơn để hội tụ vẫn có đủ "chỗ".
    num_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    patience: int = 7
    label_smoothing: float = 0.05

    imagenet_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    imagenet_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


DEFAULT_CONFIG = BenchmarkConfig()
