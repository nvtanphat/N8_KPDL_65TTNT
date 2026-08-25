"""
Bean Leaf Classification - Main Training Script
Hỗ trợ 3 model PyTorch: VGG (custom), EfficientNet-B3, MobileNetV3
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import argparse
import json
import os
import statistics

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import InterpolationMode

from bean_leaf.config import DEFAULT_CONFIG
from bean_leaf.data.dataset import create_df, get_kfold_loaders, get_train_val_test_loaders
from bean_leaf.data import eda
from bean_leaf.data.kaggle_download import download_dataset
from bean_leaf.data.transforms import build_train_transform, build_val_transform
from bean_leaf.training.amp import get_scaler
from bean_leaf.training.early_stopping import EarlyStopping
from sklearn.metrics import confusion_matrix

from bean_leaf.evaluation.complexity import model_complexity
from bean_leaf.evaluation.metrics import collect_predictions
from bean_leaf.utils.paths import get_default_output_dir
from bean_leaf.utils.seed import set_seed

from bean_leaf.models import bean_leaf_lite, efficientnet, mobilenetv3, resnet50, shufflenetv2

NUM_CLASSES = DEFAULT_CONFIG.num_classes
ALL_MODELS = ['vgg', 'efficientnet', 'mobilenet', 'resnet50', 'shufflenetv2']
device = bean_leaf_lite.device

# Mỗi model: module chứa config + factory, và interpolation phù hợp với kiến trúc
MODEL_REGISTRY = {
    'vgg': {'module': bean_leaf_lite, 'create': bean_leaf_lite.create_lite_model, 'interpolation': InterpolationMode.BILINEAR},
    'efficientnet': {'module': efficientnet, 'create': efficientnet.create_efficientnet_model, 'interpolation': InterpolationMode.BILINEAR},
    'mobilenet': {'module': mobilenetv3, 'create': mobilenetv3.create_mobilenetv3_model, 'interpolation': InterpolationMode.BILINEAR},
    'resnet50': {'module': resnet50, 'create': resnet50.create_resnet50_model, 'interpolation': InterpolationMode.BILINEAR},
    'shufflenetv2': {'module': shufflenetv2, 'create': shufflenetv2.create_shufflenetv2_model, 'interpolation': InterpolationMode.BILINEAR},
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
def train_model(model_name, train_loader, internal_val_loader, test_loader, model, output_dir,
                run_dir=None, lr=None):
    """
    Train a specific model, skip nếu checkpoint đã tồn tại.
    run_dir: thư mục checkpoint riêng cho 1 fold khi chạy k-fold (mặc định <output_dir>/<model_name>).
    lr: learning rate ghi đè (từ --lr_sweep); None = dùng DEFAULT_CONFIG.learning_rate.
    Trả về (model, stats) với stats = {test_acc, test_loss, epochs_run}.
    """
    model_output_dir = run_dir or os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    model_path = os.path.join(model_output_dir, f'best_{model_name}_model.pth')

    if os.path.exists(model_path):
        print(f"[SKIP] Model '{model_name}' already exists at {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        test_loss, test_acc = _evaluate_on_test(model_name, model, test_loader, device)
        print(f"[TEST] Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")
        return model, {'test_acc': test_acc, 'test_loss': test_loss, 'epochs_run': 0}

    print(f"\n[TRAIN] Training {model_name}...")
    print("=" * 60)

    epochs = DEFAULT_CONFIG.num_epochs
    # AMP dùng chung cho cả 5 model: img_size=384 ở batch 32 có thể vượt VRAM GPU
    # (thực tế đã CUDA OOM với EfficientNet-B3 384px/batch32 khi train thuần fp32).
    scaler = get_scaler(device)

    # Mọi model đi chung 1 pipeline: cùng AdamW + CosineAnnealingLR(T_max=epochs), cùng cách
    # step scheduler (1 lần/epoch), cùng tiêu chí lưu checkpoint. Không còn nhánh riêng cho
    # model nào - đó là điều kiện để bảng benchmark so sánh kiến trúc chứ không so sánh recipe.
    module = MODEL_REGISTRY[model_name]['module']
    criterion, optimizer, scheduler = module.get_optimizer_scheduler(model, epochs, lr=lr)
    train_fn, val_fn = module.train_one_epoch, module.validate

    # patience=0 -> đặt ngưỡng lớn hơn tổng số epoch: EarlyStopping không bao giờ kích hoạt
    # nhưng vẫn làm nhiệm vụ lưu checkpoint tốt nhất theo internal-val loss.
    patience = DEFAULT_CONFIG.patience if DEFAULT_CONFIG.patience > 0 else epochs + 1
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_path)

    epochs_run = 0
    for epoch in range(epochs):
        epochs_run = epoch + 1
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)

        train_loss, train_acc = train_fn(model, train_loader, criterion, optimizer, device, scaler)
        # bean_leaf_lite.validate trả thêm (preds, labels) ở cuối - lấy 2 phần tử đầu cho đồng nhất
        result = val_fn(model, internal_val_loader, criterion, device)
        val_loss, val_acc = result[0], result[1]
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"[Internal Val] Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("\nEarly stopping triggered!")
            break

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"\n[SAVED] Model saved to {model_path}")

    test_loss, test_acc = _evaluate_on_test(model_name, model, test_loader, device)
    print(f"[TEST] Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")
    return model, {'test_acc': test_acc, 'test_loss': test_loss, 'epochs_run': epochs_run}


# ===================== LEARNING-RATE SWEEP =====================
def _train_for_sweep(model_name, train_loader, internal_val_loader, lr, epochs, seed):
    """Train ngắn 1 lần với 1 learning rate, trả về (best_val_acc, val_loss tại epoch đó)."""
    set_seed(seed)
    model = MODEL_REGISTRY[model_name]['create'](NUM_CLASSES).to(device)
    module = MODEL_REGISTRY[model_name]['module']
    criterion, optimizer, scheduler = module.get_optimizer_scheduler(model, epochs, lr=lr)
    scaler = get_scaler(device)

    best = (-1.0, float('inf'))  # (val_acc cao nhất, val_loss tại đúng epoch đó)
    for epoch in range(epochs):
        module.train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        result = module.validate(model, internal_val_loader, criterion, device)
        val_loss, val_acc = result[0], result[1]
        scheduler.step()
        if (val_acc, -val_loss) > (best[0], -best[1]):
            best = (val_acc, val_loss)
        print(f"    epoch {epoch + 1}/{epochs}: val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
              flush=True)
    return best


def sweep_learning_rate(model_name, train_loader, internal_val_loader, lrs, epochs, seed):
    """
    Quét learning rate cho 1 model: mỗi LR train lại từ đầu trên CÙNG một split, CÙNG số epoch,
    CÙNG seed. Chọn LR theo internal-val accuracy cao nhất (tie-break bằng val_loss thấp hơn).

    Vì sao cần: ép mọi model dùng chung lr=3e-4 nghe có vẻ công bằng nhưng không phải - 3e-4 là
    LR kinh điển để fine-tune model pretrained, quá nhỏ để train from-scratch. Thực nghiệm đã
    cho thấy điều đó: khi gỡ OneCycleLR(max_lr=2e-3) riêng của BeanLeafLite để về recipe chung,
    test acc tụt từ 0.9549 xuống 0.8977. Công bằng đúng nghĩa là mọi model được thử CÙNG SỐ LẦN
    trên cùng một lưới LR, rồi so kết quả tốt nhất của từng model.

    Test set (val_dir) không tham gia bước này - chọn LR chỉ dựa trên internal-val.
    """
    print()
    print("=" * 60)
    print(f"[SWEEP] {model_name}: quét {len(lrs)} learning rate x {epochs} epoch")
    print("=" * 60)

    results = []
    for lr in lrs:
        print()
        print(f"[SWEEP] {model_name} lr={lr:g}")
        val_acc, val_loss = _train_for_sweep(model_name, train_loader, internal_val_loader,
                                             lr, epochs, seed)
        results.append({'lr': lr, 'val_acc': val_acc, 'val_loss': val_loss})
        print(f"  -> lr={lr:g}: best val_acc={val_acc:.4f} (val_loss={val_loss:.4f})", flush=True)

    best = max(results, key=lambda r: (r['val_acc'], -r['val_loss']))
    print()
    print(f"[SWEEP] {model_name}: chọn lr={best['lr']:g} "
          f"(val_acc={best['val_acc']:.4f})")
    return best['lr'], results


# ===================== K-FOLD CROSS-VALIDATION =====================
def run_kfold(model_name, train_dir, val_dir, output_dir, n_splits, seed,
              lr_grid=None, sweep_epochs=15):
    """
    Train model_name n_splits lần trên các fold khác nhau của train_dir, mỗi fold 1 seed riêng.

    Trả về 2 nhóm số liệu bổ trợ nhau:
    - test accuracy từng fold trên val_dir (test set độc lập, 133 ảnh): mean ± std cho biết kết
      quả dao động bao nhiêu giữa các lần chạy - 1 lần chạy đơn lẻ không nói được điều này.
    - out-of-fold accuracy: gộp dự đoán internal-val của tất cả fold lại, phủ TOÀN BỘ train_dir
      (~1034 ảnh) nên khoảng tin cậy hẹp hơn nhiều so với 133 ảnh của test set.
    """
    entry = MODEL_REGISTRY[model_name]
    train_tf = build_train_transform(DEFAULT_CONFIG.img_size, entry['interpolation'])
    val_tf = build_val_transform(DEFAULT_CONFIG.img_size, entry['interpolation'])

    fold_results = []
    oof_true, oof_pred = [], []

    # Chi phí tính toán đo 1 lần, không phụ thuộc fold. Đo ở đúng img_size dùng khi train
    # để con số FLOPs khớp với điều kiện benchmark, không phải con số 224px của paper gốc.
    complexity = model_complexity(MODEL_REGISTRY[model_name]['create'](NUM_CLASSES),
                                  DEFAULT_CONFIG.img_size, device='cpu')
    print(f"[COST] {model_name}: {complexity['params']/1e6:.2f}M params | "
          f"{complexity['flops']/1e9:.2f} GFLOPs ({complexity['macs']/1e9:.2f} GMACs) "
          f"@ {DEFAULT_CONFIG.img_size}px")

    # Materialize để dùng lại đúng split của fold 1 cho bước sweep - mọi model và mọi LR
    # phải thấy cùng một tập dữ liệu thì việc chọn LR mới so sánh được với nhau.
    folds = list(get_kfold_loaders(train_dir, val_dir, train_tf, val_tf, DEFAULT_CONFIG.batch_size,
                                   n_splits=n_splits, seed=seed))

    oof_index = []
    lr, sweep_results = None, None
    if lr_grid:
        _, sweep_train, sweep_val, _, _ = folds[0]
        lr, sweep_results = sweep_learning_rate(model_name, sweep_train, sweep_val,
                                                lr_grid, sweep_epochs, seed)

    for fold, train_loader, internal_val_loader, test_loader, _internal_val_idx in folds:
        # Seed lệch theo fold: mỗi fold vẫn tái lập được, nhưng khởi tạo trọng số khác nhau nên
        # std giữa các fold phản ánh cả dao động do split lẫn do khởi tạo - đúng thứ cần đo.
        fold_seed = seed + fold
        set_seed(fold_seed)

        print()
        print("=" * 60)
        print(f"[FOLD {fold}/{n_splits}] model={model_name} seed={fold_seed}")
        print("=" * 60)

        model = MODEL_REGISTRY[model_name]['create'](NUM_CLASSES).to(device)
        run_dir = os.path.join(output_dir, model_name, f'fold{fold}')
        model, stats = train_model(model_name, train_loader, internal_val_loader, test_loader,
                                   model, output_dir, run_dir=run_dir, lr=lr)

        y_true, y_pred = collect_predictions(model, internal_val_loader, device)
        oof_true.append(y_true)
        oof_pred.append(y_pred)
        oof_index.append(_internal_val_idx)
        fold_acc = float((y_true == y_pred).mean())

        stats.update({'fold': fold, 'seed': fold_seed, 'oof_acc': fold_acc,
                      'oof_n': int(len(y_true))})
        fold_results.append(stats)
        print(f"[FOLD {fold}] test_acc={stats['test_acc']:.4f} | oof_acc={fold_acc:.4f} "
              f"| epochs={stats['epochs_run']}")

    # Sắp lại theo index gốc của ImageFolder: các fold trả về theo thứ tự fold, mà muốn so
    # sánh có cặp giữa 2 model thì phần tử thứ i của 2 mảng phải là cùng một tấm ảnh.
    order = np.argsort(np.concatenate(oof_index))
    oof_true = np.concatenate(oof_true)[order]
    oof_pred = np.concatenate(oof_pred)[order]
    test_accs = [r['test_acc'] for r in fold_results]

    summary = {
        'model': model_name,
        'n_splits': n_splits,
        'base_seed': seed,
        'img_size': DEFAULT_CONFIG.img_size,
        'batch_size': DEFAULT_CONFIG.batch_size,
        'num_epochs': DEFAULT_CONFIG.num_epochs,
        'patience': DEFAULT_CONFIG.patience,
        'complexity': complexity,
        'learning_rate': lr or DEFAULT_CONFIG.learning_rate,
        'lr_sweep': sweep_results,
        'sweep_epochs': sweep_epochs if lr_grid else None,
        'folds': fold_results,
        'test_acc_mean': float(statistics.mean(test_accs)),
        # stdev cần >=2 mẫu; n_splits luôn >=2 nên không cần guard, nhưng để rõ ý đồ vẫn ghi chú:
        # đây là std mẫu (ddof=1), không phải std tổng thể.
        'test_acc_std': float(statistics.stdev(test_accs)),
        'test_acc_min': float(min(test_accs)),
        'test_acc_max': float(max(test_accs)),
        'oof_acc': float((oof_true == oof_pred).mean()),
        'oof_n': int(len(oof_true)),
        'oof_confusion_matrix': confusion_matrix(oof_true, oof_pred).tolist(),
        # Nhãn thật + dự đoán của TỪNG ảnh, xếp theo index gốc của ImageFolder(train_dir).
        # Nhờ mọi model dùng chung seed/n_splits nên thứ tự này giống hệt nhau giữa các model,
        # cho phép so sánh có cặp (McNemar) thay vì chỉ đặt 2 con số mean cạnh nhau - với
        # 1034 ảnh, McNemar phân biệt được chênh lệch mà nhìn mean±std không kết luận nổi.
        'oof_true': oof_true.tolist(),
        'oof_pred': oof_pred.tolist(),
    }

    summary_path = os.path.join(output_dir, model_name, f'kfold_{model_name}.json')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print(f"K-FOLD SUMMARY - {model_name} ({n_splits} folds, base seed {seed})")
    print("=" * 60)
    for r in fold_results:
        print(f"  fold {r['fold']} (seed {r['seed']}): test_acc={r['test_acc']:.4f} "
              f"| oof_acc={r['oof_acc']:.4f} | epochs={r['epochs_run']}")
    print()
    print(f"  Test acc (val_dir, {n_splits} lần chạy): "
          f"{summary['test_acc_mean']:.4f} ± {summary['test_acc_std']:.4f} "
          f"(min {summary['test_acc_min']:.4f}, max {summary['test_acc_max']:.4f})")
    print(f"  Out-of-fold acc (toàn bộ train_dir, {summary['oof_n']} ảnh): {summary['oof_acc']:.4f}")
    if sweep_results:
        grid = ', '.join(f"{r['lr']:g}" for r in sweep_results)
        print(f"  Learning rate: {summary['learning_rate']:g} (chọn từ lưới [{grid}])")
    else:
        print(f"  Learning rate: {summary['learning_rate']:g} (mặc định, không sweep)")
    print(f"  Chi phí: {complexity['params']/1e6:.2f}M params | "
          f"{complexity['flops']/1e9:.2f} GFLOPs @ {DEFAULT_CONFIG.img_size}px")
    print()
    print(f"  Đã lưu: {summary_path}")
    return summary


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
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed cố định cho khởi tạo trọng số / shuffle / augmentation. '
                             'Không có seed thì 2 lần chạy cùng code vẫn ra kết quả khác nhau, '
                             'rõ nhất ở BeanLeafLite vì model này train from-scratch.')
    parser.add_argument('--kfold', type=int, default=0, metavar='K',
                        help='Chạy StratifiedKFold K fold trên train_dir thay vì holdout 85/15 '
                             '1 lần. Mỗi fold train lại từ đầu với seed riêng, cho ra mean±std '
                             'của test acc + out-of-fold acc trên toàn bộ train_dir. K=0 (mặc '
                             'định) = giữ nguyên cách chạy cũ.')
    parser.add_argument('--lr_sweep', type=str, default='', metavar='LR1,LR2,...',
                        help='Quét learning rate trước khi chạy k-fold, vd "1e-4,3e-4,1e-3,3e-3". '
                             'Mỗi model được thử CÙNG SỐ LẦN trên cùng lưới này, chọn theo '
                             'internal-val (test set không tham gia). Ép mọi model dùng chung 1 LR '
                             'không phải là công bằng: 3e-4 hợp để fine-tune model pretrained '
                             'nhưng quá nhỏ với model train from-scratch.')
    parser.add_argument('--sweep_epochs', type=int, default=15,
                        help='Số epoch cho mỗi lần thử trong sweep (mặc định 15). Ngân sách rút '
                             'gọn để sweep không đắt hơn chính k-fold, áp dụng như nhau cho mọi model.')
    args = parser.parse_args()

    lr_grid = [float(x) for x in args.lr_sweep.split(',') if x.strip()] if args.lr_sweep else None
    if lr_grid and not args.kfold:
        parser.error('--lr_sweep cần đi kèm --kfold (LR chọn ra được dùng cho các fold).')

    if args.kfold == 1 or args.kfold < 0:
        parser.error('--kfold phải >= 2 (hoặc 0 để tắt); 1 fold không tính được std.')

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

    if args.kfold:
        for model_name in models_to_train:
            run_kfold(model_name, train_dir, val_dir, output_dir, args.kfold, args.seed,
                      lr_grid=lr_grid, sweep_epochs=args.sweep_epochs)
        return

    # Chạy thường: 1 holdout 85/15, seed cố định để lần chạy sau tái lập được.
    set_seed(args.seed)
    print(f'[SEED] {args.seed}')
    test_results = {}

    for model_name in models_to_train:
        train_loader, internal_val_loader, test_loader = get_model_dataloaders(model_name, train_dir, val_dir)

        model = MODEL_REGISTRY[model_name]['create'](NUM_CLASSES).to(device)
        model, stats = train_model(model_name, train_loader, internal_val_loader, test_loader, model, output_dir)
        test_results[model_name] = stats['test_acc']

        print(f"\n{model_name.upper()} completed!")

    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETED!")
    print("=" * 60)
    print("\nTest Accuracy (val_dir gốc, đánh giá 1 lần - không dùng để chọn checkpoint):")
    for model_name, test_acc in test_results.items():
        print(f"  {model_name}: {test_acc:.4f}")


if __name__ == '__main__':
    main()
