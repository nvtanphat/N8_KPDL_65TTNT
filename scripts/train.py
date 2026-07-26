"""
Bean Leaf Classification - Main Training Script
Hỗ trợ 4 model PyTorch: VGG (custom), EfficientNet-B3, MobileNetV3, DeiT

Hyperparameter dùng chung (img_size, batch_size, epochs, lr, weight_decay, patience,
label_smoothing) đọc từ bean_leaf.config.DEFAULT_CONFIG (Single Source of Truth). Cả 4
model dùng chung 1 vòng lặp huấn luyện duy nhất (for epoch ... + EarlyStopping); chỉ
VGG (OneCycleLR step theo batch + gradient clipping) và DeiT (warmup/EMA) cần nhánh
riêng cho cơ chế đặc thù kiến trúc không có tương đương ở model khác. EfficientNet-B3
và MobileNetV3 dùng y hệt 1 recipe (AdamW + CosineAnnealingLR, full fine-tune từ đầu)
nên đi chung 1 nhánh - đảm bảo so sánh công bằng giữa các kiến trúc.

Train/Val/Test: thư mục train được tách thêm 1 lần nữa thành train_subset +
internal_val_subset (stratified) để early-stopping/chọn checkpoint lúc train. Thư mục
val gốc (--data_dir/val) giữ nguyên làm TEST SET độc lập - không tham gia bất kỳ quyết
định nào lúc train, chỉ đánh giá đúng 1 lần sau khi train xong. Dùng val vừa để
early-stop vừa để báo cáo kết quả sẽ cho con số thiên vị lạc quan (đã chọn đúng
checkpoint tốt nhất trên chính tập đó).
"""

import argparse
import os

import torch
import torch.nn as nn
from torchvision.transforms import InterpolationMode

from bean_leaf.config import DEFAULT_CONFIG
from bean_leaf.data.dataset import create_df, get_train_val_test_loaders
from bean_leaf.data import eda
from bean_leaf.data.kaggle_download import download_dataset
from bean_leaf.data.transforms import build_train_transform, build_val_transform
from bean_leaf.training.amp import get_scaler
from bean_leaf.training.early_stopping import EarlyStopping
from bean_leaf.utils.paths import get_default_output_dir

from bean_leaf.models import bean_leaf_lite, efficientnet, mobilenetv3, deit

NUM_CLASSES = DEFAULT_CONFIG.num_classes
ALL_MODELS = ['vgg', 'efficientnet', 'mobilenet', 'deit']
device = bean_leaf_lite.device

# Mỗi model: module chứa config + factory, và interpolation phù hợp với kiến trúc
MODEL_REGISTRY = {
    'vgg': {'module': bean_leaf_lite, 'create': bean_leaf_lite.create_lite_model, 'interpolation': InterpolationMode.BILINEAR},
    'efficientnet': {'module': efficientnet, 'create': efficientnet.create_efficientnet_model, 'interpolation': InterpolationMode.BILINEAR},
    'mobilenet': {'module': mobilenetv3, 'create': mobilenetv3.create_mobilenetv3_model, 'interpolation': InterpolationMode.BILINEAR},
    'deit': {'module': deit, 'create': deit.create_deit_model, 'interpolation': InterpolationMode.BICUBIC},
}


def get_model_dataloaders(model_name, train_dir, val_dir):
    """train_loader + internal_val_loader (early-stopping) + test_loader (val_dir gốc, đánh giá 1 lần)."""
    entry = MODEL_REGISTRY[model_name]
    train_tf = build_train_transform(DEFAULT_CONFIG.img_size, entry['interpolation'])
    val_tf = build_val_transform(DEFAULT_CONFIG.img_size, entry['interpolation'])
    return get_train_val_test_loaders(train_dir, val_dir, train_tf, val_tf, DEFAULT_CONFIG.batch_size)


def _evaluate_on_test(model_name, model, test_loader, device):
    """Đánh giá lần cuối trên test set (val_dir gốc) - chỉ gọi 1 lần sau khi train xong."""
    module = MODEL_REGISTRY[model_name]['module']
    criterion = nn.CrossEntropyLoss()
    result = module.validate(model, test_loader, criterion, device)
    test_loss, test_acc = result[0], result[1]
    return test_loss, test_acc


# ===================== TRAINING FUNCTIONS =====================
def train_model(model_name, train_loader, internal_val_loader, test_loader, model, output_dir):
    """Train a specific model, skip nếu checkpoint đã tồn tại. Trả về (model, test_acc)."""
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    model_path = os.path.join(model_output_dir, f'best_{model_name}_model.pth')

    if os.path.exists(model_path):
        print(f"[SKIP] Model '{model_name}' already exists at {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        test_loss, test_acc = _evaluate_on_test(model_name, model, test_loader, device)
        print(f"[TEST] Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")
        return model, test_acc

    print(f"\n[TRAIN] Training {model_name}...")
    print("=" * 60)

    epochs = DEFAULT_CONFIG.num_epochs
    # AMP dùng chung: cần thiết vì img_size=384 áp cho cả 4 model có thể vượt VRAM GPU
    # (thực tế đã CUDA OOM với EfficientNet-B3 384px/batch32 khi train thuần fp32).
    scaler = get_scaler(device)

    if model_name == 'deit':
        criterion, optimizer, scheduler, model_ema = deit.get_optimizer_scheduler(model, train_loader, epochs)
        train_fn, val_fn = deit.train_one_epoch, deit.validate
    elif model_name == 'vgg':
        criterion, optimizer, scheduler = bean_leaf_lite.get_optimizer_scheduler(model, train_loader, epochs)
        train_fn, val_fn = bean_leaf_lite.train_one_epoch, bean_leaf_lite.validate
        model_ema = None
    else:  # efficientnet & mobilenet: cùng 1 recipe (AdamW + CosineAnnealingLR, full fine-tune)
        module = MODEL_REGISTRY[model_name]['module']
        criterion, optimizer, scheduler = module.get_optimizer_scheduler(model, epochs)
        train_fn, val_fn = module.train_one_epoch, module.validate
        model_ema = None

    early_stopping = EarlyStopping(patience=DEFAULT_CONFIG.patience, verbose=True, path=model_path)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)

        if model_name == 'deit':
            train_loss, train_acc = train_fn(model, model_ema, train_loader, criterion, optimizer, scheduler, device, scaler)
            # Với EMA_DECAY cao (0.9998) + ít epoch, EMA hội tụ chậm hơn hẳn model gốc
            # (thực tế đo được: EMA kẹt ~35% trong khi model gốc đạt ~99% cùng lúc) - phải so
            # sánh cả 2 và lấy bên thắng, không thể mặc định luôn dùng EMA để chấm điểm/lưu.
            val_loss, val_acc, _, _ = val_fn(model, internal_val_loader, criterion, device)
            ema_val_loss, ema_val_acc, _, _ = val_fn(model_ema.module, internal_val_loader, criterion, device)
            if ema_val_acc >= val_acc:
                best_of_epoch, val_loss, val_acc = model_ema.module, ema_val_loss, ema_val_acc
            else:
                best_of_epoch = model
        elif model_name == 'vgg':
            # OneCycleLR đã được step theo từng batch bên trong train_fn, không step lại ở đây
            train_loss, train_acc = train_fn(model, train_loader, criterion, optimizer, scheduler, device, scaler)
            val_loss, val_acc, _, _ = val_fn(model, internal_val_loader, criterion, device)
        else:  # efficientnet & mobilenet: CosineAnnealingLR, step 1 lần/epoch, không phụ thuộc val_loss
            train_loss, train_acc = train_fn(model, train_loader, criterion, optimizer, device, scaler)
            val_loss, val_acc = val_fn(model, internal_val_loader, criterion, device)
            scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"[Internal Val] Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        # DeiT: lưu đúng model (raw hoặc EMA) vừa thắng ở epoch này
        early_stopping(val_loss, best_of_epoch if model_name == 'deit' else model)
        if early_stopping.early_stop:
            print("\nEarly stopping triggered!")
            break

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"\n[SAVED] Model saved to {model_path}")

    test_loss, test_acc = _evaluate_on_test(model_name, model, test_loader, device)
    print(f"[TEST] Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")
    return model, test_acc


# ===================== MAIN =====================
def main():
    parser = argparse.ArgumentParser(description='Bean Leaf Classification')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--download', action='store_true',
                        help='Tải dataset từ Kaggle vào --data_dir trước khi train '
                             '(cần cấu hình Kaggle API credential trước, xem data/README.md)')
    parser.add_argument('--model', type=str, choices=ALL_MODELS + ['all'],
                        default='all', help='Model to train')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to save checkpoints (default: ./outputs, '
                             'or $BEAN_LEAF_OUTPUT_DIR if set)')
    parser.add_argument('--eda', action='store_true', help='Run EDA before training')
    args = parser.parse_args()

    output_dir = args.output_dir or get_default_output_dir()

    if args.download:
        download_dataset(args.data_dir)

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    os.makedirs(output_dir, exist_ok=True)

    if args.eda:
        print("\n=== Running EDA ===")
        train_df = create_df(train_dir)
        eda.plot_class_distribution(train_df, title='Training Set Distribution')

    models_to_train = ALL_MODELS if args.model == 'all' else [args.model]
    test_results = {}

    for model_name in models_to_train:
        train_loader, internal_val_loader, test_loader = get_model_dataloaders(model_name, train_dir, val_dir)

        model = MODEL_REGISTRY[model_name]['create'](NUM_CLASSES).to(device)
        model, test_acc = train_model(model_name, train_loader, internal_val_loader, test_loader, model, output_dir)
        test_results[model_name] = test_acc

        print(f"\n{model_name.upper()} completed!")

    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETED!")
    print("=" * 60)
    print("\nTest Accuracy (val_dir gốc, đánh giá 1 lần - không dùng để chọn checkpoint):")
    for model_name, test_acc in test_results.items():
        print(f"  {model_name}: {test_acc:.4f}")


if __name__ == '__main__':
    main()
