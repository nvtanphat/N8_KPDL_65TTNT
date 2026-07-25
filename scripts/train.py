"""
Bean Leaf Classification - Main Training Script
Hỗ trợ 4 model PyTorch: VGG (custom), EfficientNet-B3, MobileNetV3, DeiT

Mỗi kiến trúc sở hữu hyperparameter riêng của nó (IMG_SIZE, BATCH_SIZE, optimizer...)
ngay trong module bean_leaf.models.<tên_model> - script này chỉ điều phối:
augmentation & dataloader dùng chung, vòng train/early-stopping dùng chung.
"""

import argparse
import os

import torch
from torchvision.transforms import InterpolationMode

from bean_leaf.data.dataset import create_df, get_dataloaders
from bean_leaf.data import eda
from bean_leaf.data.transforms import build_train_transform, build_val_transform
from bean_leaf.training.early_stopping import EarlyStopping
from bean_leaf.utils.paths import get_default_output_dir

from bean_leaf.models import vgg_custom, efficientnet, mobilenetv3, deit

NUM_CLASSES = 3
ALL_MODELS = ['vgg', 'efficientnet', 'mobilenet', 'deit']
device = vgg_custom.device

# Mỗi model: module chứa config + factory, và interpolation phù hợp với kiến trúc
MODEL_REGISTRY = {
    'vgg': {'module': vgg_custom, 'create': vgg_custom.create_vgg_model, 'interpolation': InterpolationMode.BILINEAR},
    'efficientnet': {'module': efficientnet, 'create': efficientnet.create_efficientnet_model, 'interpolation': InterpolationMode.BILINEAR},
    'mobilenet': {'module': mobilenetv3, 'create': mobilenetv3.create_mobilenetv3_model, 'interpolation': InterpolationMode.BILINEAR},
    'deit': {'module': deit, 'create': deit.create_deit_model, 'interpolation': InterpolationMode.BICUBIC},
}


def get_model_dataloaders(model_name, train_dir, val_dir):
    """Dataloader dùng augmentation dùng chung, resize theo IMG_SIZE riêng của từng kiến trúc."""
    entry = MODEL_REGISTRY[model_name]
    module = entry['module']
    train_tf = build_train_transform(module.IMG_SIZE, entry['interpolation'])
    val_tf = build_val_transform(module.IMG_SIZE, entry['interpolation'])
    return get_dataloaders(train_dir, val_dir, train_tf, val_tf, module.BATCH_SIZE)


# ===================== TRAINING FUNCTIONS =====================
def train_model(model_name, train_loader, val_loader, model, output_dir):
    """Train a specific model, skip nếu checkpoint đã tồn tại"""
    module = MODEL_REGISTRY[model_name]['module']
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    model_path = os.path.join(model_output_dir, f'best_{model_name}_model.pth')

    if os.path.exists(model_path):
        print(f"[SKIP] Model '{model_name}' already exists at {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    print(f"\n[TRAIN] Training {model_name}...")
    print("=" * 60)

    # MobileNetV3: transfer learning 2 phase tự quản lý vòng lặp + early stopping riêng
    if model_name == 'mobilenet':
        model = mobilenetv3.train_mobilenetv3(model, train_loader, val_loader, model_path, device)
        print(f"\n[SAVED] Model saved to {model_path}")
        return model

    epochs = module.NUM_EPOCHS

    if model_name == 'deit':
        criterion, optimizer, scheduler, model_ema = deit.get_optimizer_scheduler(model, train_loader, epochs)
        train_fn, val_fn = deit.train_one_epoch, deit.validate
    elif model_name == 'vgg':
        criterion, optimizer, scheduler = vgg_custom.get_optimizer_scheduler(model, train_loader, epochs)
        train_fn, val_fn = vgg_custom.train_one_epoch, vgg_custom.validate
        model_ema = None
    else:  # efficientnet
        criterion, optimizer, scheduler = efficientnet.get_optimizer_scheduler(model, epochs)
        train_fn, val_fn = efficientnet.train_one_epoch, efficientnet.validate
        model_ema = None

    early_stopping = EarlyStopping(patience=module.PATIENCE, verbose=True, path=model_path)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)

        if model_name == 'deit':
            train_loss, train_acc = train_fn(model, model_ema, train_loader, criterion, optimizer, scheduler, device)
            val_loss, val_acc, _, _ = val_fn(model_ema.module, val_loader, criterion, device)
        elif model_name == 'vgg':
            # OneCycleLR đã được step theo từng batch bên trong train_fn, không step lại ở đây
            train_loss, train_acc = train_fn(model, train_loader, criterion, optimizer, scheduler, device)
            val_loss, val_acc, _, _ = val_fn(model, val_loader, criterion, device)
        else:  # efficientnet: CosineAnnealingLR, step 1 lần/epoch, không phụ thuộc val_loss
            train_loss, train_acc = train_fn(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = val_fn(model, val_loader, criterion, device)
            scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # DeiT được đánh giá trên model_ema.module nên phải lưu đúng weight đã chấm điểm đó
        early_stopping(val_loss, model_ema.module if model_name == 'deit' else model)
        if early_stopping.early_stop:
            print("\nEarly stopping triggered!")
            break

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"\n[SAVED] Model saved to {model_path}")
    return model


# ===================== MAIN =====================
def main():
    parser = argparse.ArgumentParser(description='Bean Leaf Classification')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--model', type=str, choices=ALL_MODELS + ['all'],
                        default='all', help='Model to train')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to save checkpoints (default: ./outputs, '
                             'or $BEAN_LEAF_OUTPUT_DIR if set)')
    parser.add_argument('--eda', action='store_true', help='Run EDA before training')
    args = parser.parse_args()

    output_dir = args.output_dir or get_default_output_dir()

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    os.makedirs(output_dir, exist_ok=True)

    if args.eda:
        print("\n=== Running EDA ===")
        train_df = create_df(train_dir)
        eda.plot_class_distribution(train_df, title='Training Set Distribution')

    models_to_train = ALL_MODELS if args.model == 'all' else [args.model]

    for model_name in models_to_train:
        train_loader, val_loader = get_model_dataloaders(model_name, train_dir, val_dir)

        model = MODEL_REGISTRY[model_name]['create'](NUM_CLASSES).to(device)
        model = train_model(model_name, train_loader, val_loader, model, output_dir)

        print(f"\n{model_name.upper()} completed!")

    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETED!")
    print("=" * 60)


if __name__ == '__main__':
    main()
