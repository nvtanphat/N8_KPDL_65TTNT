"""
Utility functions cho ứng dụng Web phân loại bệnh lá đậu
Hỗ trợ load 5 loại model, tất cả đều chạy trên PyTorch:
MobileNetV3, CNN VGG (custom), EfficientNet-B3, DeiT (timm), YOLO (Ultralytics, PyTorch backend)
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import os
import numpy as np
from PIL import Image
import io

try:
    from app.config import MODELS, MODEL_DIR, CLASS_NAMES
except ImportError:
    from config import MODELS, MODEL_DIR, CLASS_NAMES

# ===================== LAZY IMPORTS =====================
# Import các thư viện khi cần để giảm thời gian load ban đầu

_torch = None
_ultralytics = None


def _get_torch():
    """Lazy load PyTorch"""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _get_ultralytics():
    """Lazy load Ultralytics"""
    global _ultralytics
    if _ultralytics is None:
        from ultralytics import YOLO
        _ultralytics = YOLO
    return _ultralytics


# ===================== MODEL ARCHITECTURES =====================
# Dùng chung định nghĩa kiến trúc với lúc train (package bean_leaf, `pip install -e .`)
# thay vì khai báo lại ở đây - tránh lệch kiến trúc giữa train và inference.

def _create_vgg_model(num_classes=3):
    from bean_leaf.models.bean_leaf_lite import create_lite_model
    return create_lite_model(num_classes=num_classes)


def _create_deit_model(num_classes=3):
    from bean_leaf.models.deit import create_deit_model
    return create_deit_model(num_classes=num_classes, pretrained=False)


def _create_mobilenetv3_model(num_classes=3):
    from bean_leaf.models.mobilenetv3 import create_mobilenetv3_model
    return create_mobilenetv3_model(num_classes=num_classes, pretrained=False)


def _create_efficientnet_model(num_classes=3):
    from bean_leaf.models.efficientnet import create_efficientnet_model
    return create_efficientnet_model(num_classes=num_classes, pretrained=False)


# ===================== MODEL LOADING =====================

def load_model(model_type):
    """
    Load model dựa trên loại model
    
    Args:
        model_type: Tên model ('MobileNetV3', 'BeanLeafLite_Custom', 'EfficientNet_B3', 'DeiT_Transformer', 'YOLO_Segmentation')
    
    Returns:
        Model đã load weights
    """
    if model_type not in MODELS:
        print(f"Model type '{model_type}' không hợp lệ")
        return None
    
    config = MODELS[model_type]
    model_path = os.path.join(MODEL_DIR, config['file'])
    
    if not os.path.exists(model_path):
        print(f"Không tìm thấy file model: {model_path}")
        return None
    
    framework = config.get('framework', 'pytorch').lower()

    try:
        if 'pytorch' in framework or 'timm' in framework:
            architecture = config.get('architecture', 'bean_leaf_lite')
            return _load_pytorch_model(model_path, architecture)
        elif 'ultralytics' in framework or 'yolo' in framework:
            return _load_yolo_model(model_path)
        else:
            print(f"Framework '{framework}' không được hỗ trợ")
            return None
    except Exception as e:
        err_msg = str(e).encode('ascii', errors='backslashreplace').decode('ascii')
        print(f"Loi khi load model {model_type}: {err_msg}")
        return None


def _load_pytorch_model(model_path, architecture):
    """Load PyTorch model"""
    torch = _get_torch()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if architecture == 'bean_leaf_lite':
        model = _create_vgg_model(num_classes=3)
    elif architecture == 'deit':
        model = _create_deit_model(num_classes=3)
    elif architecture == 'mobilenetv3':
        model = _create_mobilenetv3_model(num_classes=3)
    elif architecture == 'efficientnet':
        model = _create_efficientnet_model(num_classes=3)
    else:
        print(f"Architecture '{architecture}' không được hỗ trợ")
        return None
    
    # Load state dict
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
    except Exception as err:
        err_str = str(err).encode('ascii', errors='backslashreplace').decode('ascii')
        print(f"[WARNING] State dict mismatch or load error for {architecture} at {model_path}: {err_str}")
        return None

    model.to(device)
    model.eval()
    
    print(f"Đã load PyTorch model ({architecture}): {model_path}")
    return model


def _load_yolo_model(model_path):
    """Load YOLO model"""
    YOLO = _get_ultralytics()
    model = YOLO(model_path)
    print(f"Đã load YOLO model: {model_path}")
    return model


# ===================== GRAD-CAM =====================
# Chỉ CNN có Conv2d cuối cùng dạng feature-map lưới mới áp dụng được Grad-CAM kiểu
# này. DeiT là Vision Transformer (patch embedding + self-attention, không có feature
# map dạng lưới ý nghĩa ở layer cuối) nên bỏ qua; YOLO đã có mask segmentation riêng.
GRADCAM_ARCHITECTURES = {'bean_leaf_lite', 'mobilenetv3', 'efficientnet'}


def supports_gradcam(model_type):
    """Kiến trúc của model_type có hỗ trợ Grad-CAM hay không (xem GRADCAM_ARCHITECTURES)."""
    architecture = MODELS.get(model_type, {}).get('architecture')
    return architecture in GRADCAM_ARCHITECTURES


def generate_gradcam(model, image, model_type, alpha=0.45):
    """
    Grad-CAM cho model CNN: forward + backward hook trên Conv2d cuối cùng, lấy
    activation và gradient theo lớp được dự đoán, weighted-sum theo gradient trung
    bình từng kênh (global average pooling) rồi ReLU + normalize -> heatmap. Resize
    heatmap về kích thước ảnh gốc, tô màu jet, chồng lên ảnh gốc (alpha-blend).

    Trả về PIL Image (overlay) hoặc None nếu model không có Conv2d nào (kiến trúc
    không hỗ trợ, vd DeiT).
    """
    torch = _get_torch()

    img_tensor = preprocess_image(image, model_type)
    img_tensor.requires_grad_(True)

    target_layer = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    if target_layer is None:
        return None

    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations['value'] = out

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0]

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad()
        outputs = model(img_tensor)
        pred_idx = outputs.argmax(dim=1)
        outputs[0, pred_idx].backward()

        acts = activations['value'][0].detach()
        grads = gradients['value'][0].detach()
        weights = grads.mean(dim=(1, 2))

        heatmap = torch.zeros(acts.shape[1:], device=acts.device)
        for i, w in enumerate(weights):
            heatmap += w * acts[i]

        heatmap = torch.relu(heatmap)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        heatmap = heatmap.cpu().numpy()
    finally:
        handle_f.remove()
        handle_b.remove()

    return _overlay_heatmap(image, heatmap, alpha=alpha)


def _overlay_heatmap(image, heatmap, alpha=0.45):
    """Resize heatmap [0,1] (H_feat, W_feat) về kích thước ảnh gốc, tô màu jet, chồng
    lên ảnh gốc bằng alpha-blend. Trả về PIL Image RGB cùng kích thước ảnh gốc."""
    import matplotlib.cm as cm

    heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize(image.size, Image.BILINEAR)
    heatmap_resized = np.array(heatmap_img) / 255.0

    colored = np.uint8(cm.jet(heatmap_resized)[:, :, :3] * 255)
    colored_img = Image.fromarray(colored).convert('RGB')

    return Image.blend(image.convert('RGB'), colored_img, alpha)


# ===================== IMAGE PROCESSING =====================

def read_image(image_bytes):
    """Đọc ảnh từ bytes"""
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def preprocess_image(image, model_type):
    """
    Tiền xử lý ảnh cho model cụ thể
    
    Args:
        image: PIL Image
        model_type: Loại model
    
    Returns:
        Tensor/array đã được tiền xử lý
    """
    config = MODELS[model_type]
    img_size = config['img_size']
    framework = config.get('framework', 'pytorch').lower()

    # Resize ảnh
    image_resized = image.resize(img_size, Image.Resampling.LANCZOS)

    if 'pytorch' in framework or 'timm' in framework:
        torch = _get_torch()
        # PyTorch: normalize và channel first
        img_array = np.array(image_resized, dtype=np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        # Channel first: (H, W, C) -> (C, H, W)
        img_array = img_array.transpose(2, 0, 1)
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return img_tensor.to(device)
    
    elif 'ultralytics' in framework or 'yolo' in framework:
        # YOLO xử lý ảnh trong nội bộ
        return image
    
    return image


# ===================== PREDICTION =====================

def predict(model, image, model_type):
    """
    Dự đoán loại bệnh từ ảnh
    
    Args:
        model: Model đã load
        image: PIL Image
        model_type: Loại model
    
    Returns:
        Dict chứa kết quả dự đoán
    """
    config = MODELS[model_type]
    framework = config.get('framework', 'pytorch').lower()

    if 'pytorch' in framework or 'timm' in framework:
        return _predict_pytorch(model, image, model_type)
    elif 'ultralytics' in framework or 'yolo' in framework:
        return _predict_yolo(model, image, model_type)

    return {'class': 'unknown', 'confidence': 0.0, 'probabilities': {}}


def _predict_pytorch(model, image, model_type):
    """Dự đoán với PyTorch model"""
    torch = _get_torch()
    
    img_tensor = preprocess_image(image, model_type)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    pred_idx = probabilities.argmax().item()
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probabilities[pred_idx].item() * 100
    
    probs_dict = {CLASS_NAMES[i]: probabilities[i].item() * 100 for i in range(len(CLASS_NAMES))}
    
    return {
        'class': pred_class,
        'confidence': confidence,
        'probabilities': probs_dict
    }


def _predict_yolo(model, image, model_type):
    """Dự đoán với YOLO model (segmentation)"""
    # YOLO segmentation - hạ conf threshold để detect tốt hơn
    results = model.predict(image, verbose=False, conf=0.1, imgsz=640)
    
    if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
        # Lấy detection có confidence cao nhất
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        
        best_idx = np.argmax(confidences)
        pred_class_idx = classes[best_idx]
        confidence = float(confidences[best_idx]) * 100
        
        # Map YOLO class index to class name
        yolo_names = results[0].names
        pred_class = yolo_names.get(pred_class_idx, 'unknown')
        
        # Convert to standard class name format
        pred_class_mapped = _map_yolo_class(pred_class)
        
        # Tạo probabilities (ước lượng từ detection confidences)
        probabilities = {cls: 0.0 for cls in CLASS_NAMES}
        probabilities[pred_class_mapped] = confidence
        
        return {
            'class': pred_class_mapped,
            'confidence': confidence,
            'probabilities': probabilities,
            'segmentation_result': results[0]  # Thêm kết quả segmentation
        }
    
    # Không phát hiện được bệnh
    return {
        'class': 'healthy',
        'confidence': 0.0,
        'probabilities': {cls: 0.0 for cls in CLASS_NAMES}
    }


def _map_yolo_class(yolo_class_name):
    """Map tên class từ YOLO sang format chuẩn"""
    yolo_class_lower = yolo_class_name.lower().replace(' ', '_').replace('-', '_')
    
    # Mapping các biến thể tên (YOLO model có: Angular_Leaf_Spot, Bean_Rust, Healthy)
    mappings = {
        'healthy': 'healthy',
        'angular_leaf_spot': 'angular_leaf_spot',
        'bean_rust': 'bean_rust',
        'angular': 'angular_leaf_spot',
        'rust': 'bean_rust',
    }
    
    return mappings.get(yolo_class_lower, 'healthy')
