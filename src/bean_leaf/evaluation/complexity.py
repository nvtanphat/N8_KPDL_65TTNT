"""
Đo chi phí tính toán của model: số tham số và FLOPs cho 1 ảnh.

Cần thiết vì luận điểm của BeanLeafLite là "nhẹ mà vẫn tốt" - nếu bảng benchmark chỉ có
accuracy thì đúng trục quan trọng nhất của nó lại đang thiếu. Params đo dung lượng lưu trữ,
FLOPs đo chi phí suy luận; hai con số này không tỉ lệ với nhau (depthwise-separable conv
giảm params mạnh hơn giảm FLOPs, còn ResNet50 ngược lại).
"""
import torch
from torch.utils.flop_counter import FlopCounterMode


def count_params(model):
    """Trả về (tổng số tham số, số tham số trainable)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_flops(model, img_size, device=None):
    """
    FLOPs của 1 lần forward trên 1 ảnh (batch=1, 3 x img_size x img_size).

    Dùng torch.utils.flop_counter có sẵn trong PyTorch 2.x - không cần thêm dependency
    (thop/fvcore). Bộ đếm này tính 1 phép nhân-cộng là 2 FLOPs, nên nhiều paper ghi
    "FLOPs" theo nghĩa MACs sẽ ra đúng một nửa con số này; hàm trả về cả 2 để khỏi nhầm.

    Trả về dict {flops, macs} với macs = flops / 2.
    """
    was_training = model.training
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device or next(model.parameters()).device)

    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(x)
    flops = counter.get_total_flops()

    if was_training:
        model.train()
    return {'flops': int(flops), 'macs': int(flops // 2)}


def model_complexity(model, img_size, device=None):
    """Gộp params + FLOPs thành 1 dict JSON-serializable để nhét thẳng vào file kết quả."""
    total, trainable = count_params(model)
    c = count_flops(model, img_size, device=device)
    c.update({'params': total, 'params_trainable': trainable, 'img_size': img_size})
    return c
