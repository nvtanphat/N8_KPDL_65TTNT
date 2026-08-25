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

# Cấu hình model - 5 models CNN từ folder models/
MODELS = {
    'BeanLeafLite_Custom': {
        'file': 'best_beanleaflite.pth',
        'img_size': (384, 384),
        'framework': 'PyTorch',
        'architecture': 'bean_leaf_lite',
        'description': '🛠️ [Mô hình Custom Đổi Mới] BeanLeafLite: CNN tự thiết kế từ đầu (from scratch) với Depthwise-Separable + Residual + SE Attention, ~0.94M tham số.',
        'dataset': 'Bean Leaf Dataset - 3 classes (1,296 images)',
    },
    'ShuffleNetV2': {
        'file': 'best_shufflenetv2.pth',
        'img_size': (384, 384),
        'framework': 'PyTorch',
        'architecture': 'shufflenetv2',
        'description': '⚡ [Mô hình Siêu Nhẹ] ShuffleNetV2 (x1.0): Tối ưu hóa xáo trộn kênh đặc trưng (Channel Shuffle), ~2.3M tham số cho thiết bị di động.',
        'dataset': 'Bean Leaf Dataset - 3 classes (1,296 images)',
    },
    'MobileNetV3': {
        'file': 'best_mobilenetv3.pth',
        'img_size': (384, 384),
        'framework': 'PyTorch',
        'architecture': 'mobilenetv3',
        'description': '📱 [Mô hình Edge/Mobile] MobileNetV3-Large siêu nhẹ (~3.2M params), suy luận tốc độ cao trên thiết bị nông dân ngoài thực địa.',
        'dataset': 'Bean Leaf Dataset - 3 classes (1,296 images)',
    },
    'EfficientNet_B0': {
        'file': 'best_efficientnet_b0.pth',
        'img_size': (384, 384),
        'framework': 'PyTorch',
        'architecture': 'efficientnet',
        'description': '⚖️ [Mô hình Cân Bằng] EfficientNet-B0 fine-tuned từ ImageNet (~5.3M params), tối ưu giữa dung lượng và độ chính xác.',
        'dataset': 'Bean Leaf Dataset - 3 classes (1,296 images)',
    },
    'ResNet50': {
        'file': 'best_resnet50.pth',
        'img_size': (384, 384),
        'framework': 'PyTorch',
        'architecture': 'resnet50',
        'description': '🏛️ [Mô hình Residual Tiêu Chuẩn] ResNet50 (~25.6M params), kiến trúc mạng cuộn sâu với kết nối tắt (Skip Connections).',
        'dataset': 'Bean Leaf Dataset - 3 classes (1,296 images)',
    }
}
