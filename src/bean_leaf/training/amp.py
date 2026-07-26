"""
Automatic Mixed Precision (AMP) helper dùng chung cho vòng train của cả 4 model -
activation lưu ở fp16 thay vì fp32, giảm ~40-50% bộ nhớ GPU. Cần thiết vì
DEFAULT_CONFIG.img_size=384 dùng chung cho cả 4 kiến trúc: EfficientNet-B3 ở
384px/batch 32 CUDA OOM trên GPU T4 (14.56GB) nếu train thuần fp32.
"""
import torch


def get_scaler(device):
    """GradScaler chỉ có tác dụng thật trên CUDA; enabled=False trên CPU (no-op an toàn)."""
    return torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))


def autocast_context(device):
    return torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda'))
