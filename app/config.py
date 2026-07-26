"""
Cấu hình cho ứng dụng Web phân loại bệnh lá đậu
"""
import os

# Đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Tên các lớp phân loại - THỨ TỰ PHẢI khớp với torchvision.datasets.ImageFolder,
# tức sắp xếp alphabet theo tên thư mục con (angular_leaf_spot < bean_rust < healthy).
CLASS_NAMES = ['angular_leaf_spot', 'bean_rust', 'healthy']
CLASS_LABELS = {
    'healthy': 'Khỏe mạnh',
    'angular_leaf_spot': 'Đốm góc cạnh',
    'bean_rust': 'Rỉ sắt'
}

# Thông tin về các loại bệnh
DISEASE_INFO = {
    'healthy': {
        'name': 'Lá Khỏe Mạnh',
        'description': 'Lá đậu khỏe mạnh, không có dấu hiệu bệnh.',
        'severity': 'Bình thường',
        'symptoms': [],
        'recommendation': 'Tiếp tục chăm sóc cây bình thường, duy trì tưới nước và bón phân hợp lý.'
    },
    'angular_leaf_spot': {
        'name': 'Bệnh Đốm Góc Cạnh (Angular Leaf Spot)',
        'description': (
            'Bệnh đốm góc cạnh do vi khuẩn Pseudomonas syringae pv. phaseolicola gây ra. '
            'Đây là một trong những bệnh phổ biến nhất trên cây đậu, gây thiệt hại nghiêm trọng cho năng suất.'
        ),
        'severity': 'Nghiêm trọng',
        'symptoms': [
            'Xuất hiện các đốm nhỏ màu nâu có góc cạnh trên lá',
            'Đốm bệnh thường bị giới hạn bởi các gân lá',
            'Lá bị héo và rụng sớm',
            'Quanh đốm bệnh có viền vàng nhạt',
            'Bệnh lây lan nhanh trong điều kiện ẩm ướt'
        ],
        'recommendation': (
            'Cần loại bỏ các lá bị bệnh ngay lập tức. '
            'Sử dụng thuốc diệt khuẩn có chứa đồng. '
            'Tránh tưới nước lên lá và đảm bảo thông thoáng cho cây.'
        )
    },
    'bean_rust': {
        'name': 'Bệnh Rỉ Sắt (Bean Rust)',
        'description': (
            'Bệnh rỉ sắt do nấm Uromyces appendiculatus gây ra. '
            'Bệnh xuất hiện chủ yếu trong mùa mưa và điều kiện độ ẩm cao.'
        ),
        'severity': 'Trung bình đến Nghiêm trọng',
        'symptoms': [
            'Xuất hiện các đốm nhỏ màu nâu đỏ như rỉ sắt trên mặt dưới lá',
            'Các đốm bệnh phình lên và có bột bào tử màu nâu',
            'Lá chuyển vàng và rụng sớm',
            'Cây sinh trưởng kém, năng suất giảm',
            'Bệnh lây lan nhanh qua gió và mưa'
        ],
        'recommendation': (
            'Phun thuốc trừ nấm có hoạt chất như Mancozeb hoặc Chlorothalonil. '
            'Thu dọn và tiêu hủy lá bệnh. '
            'Trồng giống kháng bệnh nếu có thể.'
        )
    }
}

# Cấu hình model - 5 models từ folder models/
MODELS = {
    'MobileNetV3': {
        'file': 'best_mobilenetv3.pth',
        'img_size': (384, 384),
        'framework': 'PyTorch',
        'architecture': 'mobilenetv3',
        'description': '⚡ [Mô hình Nhẹ Nhất - Edge/Mobile] MobileNetV3-Large siêu nhẹ (~3.2M params), suy luận tức thì (~mấy ms/ảnh), tối ưu cho thiết bị di động & nông dân ngoài thực địa.',
        'dataset': 'Bean Leaf Dataset - 3 classes (1,296 images)',
    },
    'BeanLeafLite_Custom': {
        'file': 'best_beanleaflite.pth',
        'img_size': (384, 384),
        'framework': 'PyTorch',
        'architecture': 'bean_leaf_lite',
        'description': '🛠️ [Mô hình Custom Đổi Mới] BeanLeafLite: CNN tự thiết kế từ đầu (from scratch) với Depthwise-Separable + Residual + SE Attention, ~1M tham số.',
        'dataset': 'Bean Leaf Dataset - 3 classes (1,296 images)',
    },
    'DeiT_Transformer': {
        'file': 'best_deit.pth',
        'img_size': (384, 384),
        'framework': 'PyTorch/timm',
        'architecture': 'deit',
        'description': '🥇 [Mô hình Độ Chính Xác Cao Nhất - SOTA 99.25%] Vision Transformer (DeiT-Small) khai thác cơ chế Self-Attention cho kết quả chẩn đoán chính xác tuyệt đối trên Server/Cloud.',
        'dataset': 'Bean Leaf Dataset - 3 classes (1,296 images)',
    },
    'EfficientNet_B3': {
        'file': 'best_efficientnet.pth',
        'img_size': (384, 384),
        'framework': 'PyTorch',
        'architecture': 'efficientnet',
        'description': '⚖️ [Mô hình Cân Bằng Nhất] EfficientNet-B3 fine-tuned từ ImageNet, cân bằng lý tưởng giữa khả năng tổng quát hóa và tài nguyên tính toán.',
        'dataset': 'Bean Leaf Dataset - 3 classes (1,296 images)',
    },
    'YOLO_Segmentation': {
        'file': 'best_yolov8_segmentation.pt',
        'img_size': (640, 640),
        'framework': 'Ultralytics',
        'architecture': 'yolov8',
        'description': '🎯 [Mô hình Phân Vùng Ổ Bệnh] YOLOv8-seg cho Instance Segmentation, phát hiện vị trí ổ bệnh và vẽ mặt nạ (polygon mask) khoanh vùng trực quan.',
        'dataset': 'Bean Leaf Segmentation Dataset (Roboflow)',
    }
}
