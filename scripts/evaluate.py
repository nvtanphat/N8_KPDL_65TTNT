"""
Đánh giá offline 3 model classification đã train, trên tập test độc lập (data/val -
chưa từng dùng để chọn checkpoint/early-stopping trong lúc train, xem README mục
Training: Train / Internal-Val / Test). Đọc checkpoint trực tiếp từ models/ (không
cần train lại) và lưu toàn bộ metric (accuracy, precision/recall/F1 từng lớp,
confusion matrix) ra 1 file JSON để dùng lại (viết báo cáo, hiển thị web app...) mà
không phải chạy lại inference mỗi lần.

Cách chạy:
    python scripts/evaluate.py
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import InterpolationMode
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from bean_leaf.config import DEFAULT_CONFIG
from bean_leaf.data.transforms import build_val_transform
from bean_leaf.evaluation.metrics import _run_inference
from bean_leaf.models import bean_leaf_lite, efficientnet, mobilenetv3, resnet50, shufflenetv2

# Khớp tên checkpoint + 'architecture' trong app/config.py (MODELS) - dùng chung 1
# nguồn checkpoint với web app, không tải/lưu riêng.
MODEL_REGISTRY = {
    'bean_leaf_lite': {
        'create': bean_leaf_lite.create_lite_model,
        'checkpoint': 'best_beanleaflite.pth',
        'interpolation': InterpolationMode.BILINEAR,
    },
    'shufflenetv2': {
        'create': lambda num_classes: shufflenetv2.create_shufflenetv2_model(num_classes=num_classes, pretrained=False),
        'checkpoint': 'best_shufflenetv2.pth',
        'interpolation': InterpolationMode.BILINEAR,
    },
    'mobilenetv3': {
        'create': lambda num_classes: mobilenetv3.create_mobilenetv3_model(num_classes=num_classes, pretrained=False),
        'checkpoint': 'best_mobilenetv3.pth',
        'interpolation': InterpolationMode.BILINEAR,
    },
    'efficientnet': {
        'create': lambda num_classes: efficientnet.create_efficientnet_model(num_classes=num_classes, pretrained=False),
        'checkpoint': 'best_efficientnet_b0.pth',
        'interpolation': InterpolationMode.BILINEAR,
    },
    'resnet50': {
        'create': lambda num_classes: resnet50.create_resnet50_model(num_classes=num_classes, pretrained=False),
        'checkpoint': 'best_resnet50.pth',
        'interpolation': InterpolationMode.BILINEAR,
    },
}


def evaluate_one(name, entry, val_dir, device, model_dir):
    """Load checkpoint + chạy inference trên val_dir, trả về dict metric (JSON-serializable)."""
    checkpoint_path = os.path.join(model_dir, entry['checkpoint'])
    if not os.path.exists(checkpoint_path):
        print(f"[SKIP] {name}: khong tim thay checkpoint {checkpoint_path}")
        return None

    model = entry['create'](num_classes=DEFAULT_CONFIG.num_classes)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)

    transform = build_val_transform(DEFAULT_CONFIG.img_size, entry['interpolation'])
    dataset = datasets.ImageFolder(val_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=DEFAULT_CONFIG.batch_size, shuffle=False)

    all_labels, all_scores = _run_inference(model, loader, device)
    all_preds = np.argmax(all_scores, axis=1)

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=dataset.classes, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds).tolist()

    print(f"[{name}] Test Accuracy: {acc:.4f} ({len(all_labels)} anh)")

    return {
        'checkpoint': entry['checkpoint'],
        'test_accuracy': float(acc),
        'num_samples': int(len(all_labels)),
        'class_names': dataset.classes,
        'classification_report': report,
        'confusion_matrix': cm,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Danh gia offline cac model classification tren test set (data/val)',
    )
    parser.add_argument('--val_dir', type=str, default=os.path.join('data', 'val'))
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--output', type=str, default=os.path.join('outputs', 'evaluation_metrics.json'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Test set (khong dung de chon checkpoint): {args.val_dir}")
    print("=" * 60)

    results = {}
    for name, entry in MODEL_REGISTRY.items():
        result = evaluate_one(name, entry, args.val_dir, device, args.model_dir)
        if result is not None:
            results[name] = result

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVED] {args.output}")
    print("\n=== Tong ket Test Accuracy ===")
    for name, result in results.items():
        print(f"{name}: {result['test_accuracy']:.4f}")


if __name__ == '__main__':
    main()
