"""
Cấu hình cho ứng dụng Web phân loại bệnh lá đậu
"""
import os

# Đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')  # Thay đổi từ 'results' sang 'model'
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Tên các lớp phân loại
CLASS_NAMES = ['healthy', 'angular_leaf_spot', 'bean_rust']
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

# Cấu hình model - 4 models từ folder model
MODELS = {
    'MobileNetV3': {
        'file': 'best_mobilenetv3_tanphat.keras',
        'img_size': (224, 224),
        'framework': 'TensorFlow/Keras',
        'developer': 'Nguyễn Văn Tấn Phát',
        'description': 'Model MobileNetV3Large fine-tuned với transfer learning từ ImageNet. Tối ưu cho thiết bị di động.',
        'dataset': 'Bean Leaf Dataset - 3 classes (1,296 images)',
    },
    'CNN_VGG_Custom': {
        'file': 'model_cratch_hoangloc.pth',
        'img_size': (400, 400),
        'framework': 'PyTorch',
        'architecture': 'vgg_custom',
        'developer': 'Nguyễn Hoàng Lộc',
        'description': 'Model CNN VGG tự xây dựng từ đầu (from scratch). Kiến trúc 5 blocks với BatchNorm.',
        'dataset': 'Bean Leaf Dataset - 3 classes (1,296 images)',
    },
    'DeiT_Transformer': {
        'file': 'model_deit_tanphat.pth',
        'img_size': (384, 384),
        'framework': 'PyTorch/timm',
        'architecture': 'deit',
        'developer': 'Nguyễn Văn Tấn Phát',
        'description': 'Model DeiT (Data-efficient Image Transformer). Vision Transformer hiện đại với attention mechanism.',
        'dataset': 'Bean Leaf Dataset - 3 classes (1,296 images)',
    },
    'YOLO_Segmentation': {
        'file': 'model_segemnt_yolo.pt',
        'img_size': (640, 640),
        'framework': 'Ultralytics',
        'architecture': 'yolov8',
        'developer': 'Nhóm 8',
        'description': 'Model YOLOv8 cho Instance Segmentation. Phát hiện và phân vùng chính xác vùng bệnh trên lá.',
        'dataset': 'Bean Leaf Segmentation Dataset (Roboflow)',
    }
}
