"""
Dựng lại dự đoán out-of-fold cho từng ảnh từ checkpoint của các fold, rồi so sánh các model
bằng kiểm định McNemar.

Vì sao cần: bảng benchmark chỉ có mean ± std không xếp hạng được nhóm dẫn đầu - độ lệch chuẩn
giữa các fold (± 0.73-1.49%) lớn hơn khoảng cách giữa các model (0.10-0.87 điểm). McNemar so
từng cặp model TRÊN CÙNG MỘT ẢNH, chỉ đếm những ảnh mà hai model bất đồng, nên nhạy hơn hẳn
việc đặt hai con số accuracy cạnh nhau.

Không cần train lại: các fold tái lập được từ seed cố định, checkpoint từng fold đã có sẵn.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torchvision.transforms import InterpolationMode

from bean_leaf.config import DEFAULT_CONFIG
from bean_leaf.data.dataset import get_kfold_loaders
from bean_leaf.data.transforms import build_train_transform, build_val_transform
from bean_leaf.evaluation.metrics import collect_predictions
from bean_leaf.models import bean_leaf_lite, efficientnet, mobilenetv3, resnet50, shufflenetv2

MODELS = {
    'vgg': bean_leaf_lite.create_lite_model,
    'efficientnet': lambda n: efficientnet.create_efficientnet_model(n, pretrained=False),
    'mobilenet': lambda n: mobilenetv3.create_mobilenetv3_model(n, pretrained=False),
    'resnet50': lambda n: resnet50.create_resnet50_model(n, pretrained=False),
    'shufflenetv2': lambda n: shufflenetv2.create_shufflenetv2_model(n, pretrained=False),
}


def find_checkpoint(roots, model_name, fold):
    """Tìm best_<model>_model.pth của 1 fold trong các thư mục output đã tải về."""
    rel = os.path.join(model_name, f'fold{fold}', f'best_{model_name}_model.pth')
    for root in roots:
        for dirpath, _dirnames, _files in os.walk(root):
            candidate = os.path.join(dirpath, rel)
            if os.path.isfile(candidate):
                return candidate
    return None


def build_oof(model_name, train_dir, val_dir, roots, n_splits, seed, device):
    """Chạy inference từng fold trên đúng phần dữ liệu fold đó không nhìn thấy lúc train."""
    train_tf = build_train_transform(DEFAULT_CONFIG.img_size, InterpolationMode.BILINEAR)
    val_tf = build_val_transform(DEFAULT_CONFIG.img_size, InterpolationMode.BILINEAR)

    preds, trues, idxs = [], [], []
    for fold, _train_loader, internal_val_loader, _test_loader, internal_val_idx in get_kfold_loaders(
            train_dir, val_dir, train_tf, val_tf, DEFAULT_CONFIG.batch_size,
            n_splits=n_splits, seed=seed):
        ckpt = find_checkpoint(roots, model_name, fold)
        if ckpt is None:
            print(f"  [SKIP] {model_name} fold {fold}: không tìm thấy checkpoint")
            return None

        model = MODELS[model_name](DEFAULT_CONFIG.num_classes).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
        y_true, y_pred = collect_predictions(model, internal_val_loader, device)
        trues.append(y_true)
        preds.append(y_pred)
        idxs.append(internal_val_idx)
        print(f"  fold {fold}: {len(y_true)} ảnh, acc={np.mean(y_true == y_pred):.4f}", flush=True)

    # Sắp theo index gốc của ImageFolder để phần tử thứ i của mọi model là cùng một tấm ảnh
    order = np.argsort(np.concatenate(idxs))
    return {
        'y_true': np.concatenate(trues)[order].tolist(),
        'y_pred': np.concatenate(preds)[order].tolist(),
    }


def mcnemar(pred_a, pred_b, truth):
    """
    McNemar cho 2 model trên cùng tập ảnh. Chỉ 2 ô lệch mới mang thông tin:
      b = số ảnh A đúng / B sai,  c = số ảnh A sai / B đúng.
    Dùng binomial test 2 phía (chính xác, không cần xấp xỉ chi-square vốn kém tin cậy khi b+c nhỏ).
    """
    from scipy.stats import binomtest

    a_ok = np.asarray(pred_a) == np.asarray(truth)
    b_ok = np.asarray(pred_b) == np.asarray(truth)
    n01 = int(np.sum(a_ok & ~b_ok))
    n10 = int(np.sum(~a_ok & b_ok))
    if n01 + n10 == 0:
        return {'n01': 0, 'n10': 0, 'p_value': 1.0}
    p = binomtest(n01, n01 + n10, 0.5).pvalue
    return {'n01': n01, 'n10': n10, 'p_value': float(p)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--checkpoint_root', action='append', required=True,
                        help='Thư mục chứa <model>/fold<k>/best_<model>_model.pth (lặp lại được)')
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=os.path.join('outputs', 'kfold', 'oof_predictions.json'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    print(f"Device: {device} | seed={args.seed} n_splits={args.n_splits}")

    oof = {}
    for name in MODELS:
        print(f"\n[{name}]", flush=True)
        result = build_oof(name, train_dir, val_dir, args.checkpoint_root,
                           args.n_splits, args.seed, device)
        if result is not None:
            oof[name] = result

    if len(oof) < 2:
        print("Cần ít nhất 2 model để so sánh.", file=sys.stderr)
        return 1

    names = list(oof)
    truth = oof[names[0]]['y_true']
    for n in names[1:]:
        assert oof[n]['y_true'] == truth, f"{n} có nhãn thật khác - không ghép cặp được"

    comparisons = {}
    print("\n" + "=" * 72)
    print("McNemar - so từng cặp trên cùng 1034 ảnh (n01 = chỉ A đúng, n10 = chỉ B đúng)")
    print("=" * 72)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = mcnemar(oof[a]['y_pred'], oof[b]['y_pred'], truth)
            verdict = "KHAC BIET" if r['p_value'] < 0.05 else "khong tach duoc"
            comparisons[f"{a}_vs_{b}"] = r
            print(f"  {a:14s} vs {b:14s}  n01={r['n01']:3d} n10={r['n10']:3d} "
                  f"p={r['p_value']:.4f}  -> {verdict}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({'oof': oof, 'mcnemar': comparisons}, f, indent=2)
    print(f"\n[SAVED] {args.output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
