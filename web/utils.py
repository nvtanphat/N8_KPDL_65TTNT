"""
Utility functions cho ứng dụng Web phân loại bệnh lá đậu
Hỗ trợ load 4 loại model: MobileNetV3 (Keras), CNN VGG (PyTorch), DeiT (PyTorch), YOLO (Ultralytics)
"""
import os
import numpy as np
from PIL import Image
import io

from config import MODELS, MODEL_DIR, CLASS_NAMES

# ===================== LAZY IMPORTS =====================
# Import các thư viện khi cần để giảm thời gian load ban đầu

_tf = None
_torch = None
_timm = None
_ultralytics = None


def _get_tensorflow():
    """Lazy load TensorFlow"""
    global _tf
    if _tf is None:
        import tensorflow as tf
        _tf = tf
    return _tf


def _get_torch():
    """Lazy load PyTorch"""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _get_timm():
    """Lazy load timm"""
    global _timm
    if _timm is None:
        import timm
        _timm = timm
    return _timm


def _get_ultralytics():
    """Lazy load Ultralytics"""
    global _ultralytics
    if _ultralytics is None:
        from ultralytics import YOLO
        _ultralytics = YOLO
    return _ultralytics


# ===================== MODEL ARCHITECTURES =====================

def _create_vgg_model(num_classes=3):
    """Tạo model CNN VGG custom (from scratch)"""
    torch = _get_torch()
    import torch.nn as nn
    
    class VGGBlock(nn.Module):
        """Block cơ bản của VGG"""
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
        """Custom VGG model cho Bean Leaf Classification"""
        def __init__(self, num_classes=3):
            super(BeanLeafVGG, self).__init__()
            self.block1 = VGGBlock(3, 32)
            self.block2 = VGGBlock(32, 64)
            self.block3 = VGGBlock(64, 128)
            self.block4 = VGGBlock(128, 256)
            self.block5 = VGGBlock(256, 512)
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
    
    return BeanLeafVGG(num_classes=num_classes)


def _create_deit_model(num_classes=3):
    """Tạo model DeiT"""
    timm = _get_timm()
    model = timm.create_model(
        'deit3_small_patch16_384.fb_in1k',
        pretrained=False,
        num_classes=num_classes
    )
    return model


# ===================== MODEL LOADING =====================

def load_model(model_type):
    """
    Load model dựa trên loại model
    
    Args:
        model_type: Tên model ('MobileNetV3', 'CNN_VGG_Custom', 'DeiT_Transformer', 'YOLO_Segmentation')
    
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
    
    framework = config.get('framework', 'keras').lower()
    
    try:
        if 'keras' in framework or 'tensorflow' in framework:
            return _load_keras_model(model_path)
        elif 'pytorch' in framework or 'timm' in framework:
            architecture = config.get('architecture', 'vgg_custom')
            return _load_pytorch_model(model_path, architecture)
        elif 'ultralytics' in framework or 'yolo' in framework:
            return _load_yolo_model(model_path)
        else:
            print(f"Framework '{framework}' không được hỗ trợ")
            return None
    except Exception as e:
        print(f"Lỗi khi load model {model_type}: {e}")
        return None


def _load_keras_model(model_path):
    """Load Keras/TensorFlow model"""
    tf = _get_tensorflow()
    model = tf.keras.models.load_model(model_path)
    print(f"Đã load Keras model: {model_path}")
    return model


def _load_pytorch_model(model_path, architecture):
    """Load PyTorch model"""
    torch = _get_torch()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if architecture == 'vgg_custom':
        model = _create_vgg_model(num_classes=3)
    elif architecture == 'deit':
        model = _create_deit_model(num_classes=3)
    else:
        print(f"Architecture '{architecture}' không được hỗ trợ")
        return None
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
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
    framework = config.get('framework', 'keras').lower()
    
    # Resize ảnh
    image_resized = image.resize(img_size, Image.Resampling.LANCZOS)
    
    if 'keras' in framework or 'tensorflow' in framework:
        # Keras/TensorFlow: [0, 1] range, batch dimension
        img_array = np.array(image_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    elif 'pytorch' in framework or 'timm' in framework:
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
    framework = config.get('framework', 'keras').lower()
    
    if 'keras' in framework or 'tensorflow' in framework:
        return _predict_keras(model, image, model_type)
    elif 'pytorch' in framework or 'timm' in framework:
        return _predict_pytorch(model, image, model_type)
    elif 'ultralytics' in framework or 'yolo' in framework:
        return _predict_yolo(model, image, model_type)
    
    return {'class': 'unknown', 'confidence': 0.0, 'probabilities': {}}


def _predict_keras(model, image, model_type):
    """Dự đoán với Keras model"""
    img_array = preprocess_image(image, model_type)
    predictions = model.predict(img_array, verbose=0)[0]
    
    pred_idx = np.argmax(predictions)
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(predictions[pred_idx]) * 100
    
    probabilities = {CLASS_NAMES[i]: float(p) * 100 for i, p in enumerate(predictions)}
    
    return {
        'class': pred_class,
        'confidence': confidence,
        'probabilities': probabilities
    }


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
