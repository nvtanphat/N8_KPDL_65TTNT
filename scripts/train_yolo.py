"""
Train YOLOv8 Segmentation - script riêng cho model detect/segment vùng bệnh.

Khác với scripts/train.py (4 model classification, đọc data_dir/train,val dạng
ImageFolder), YOLO cần 1 file data.yaml (định dạng segmentation, export từ
Roboflow - xem README phần Dataset) nên có entrypoint và cách gọi dữ liệu riêng.
"""

import argparse
import os

import torch

from bean_leaf.models.yolo_seg import (
    create_yolo_model,
    train_yolo_model,
    evaluate_yolo_model,
    load_yolo_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8-seg cho bean leaf disease')
    parser.add_argument('--data_yaml', type=str, required=True,
                         help='Đường dẫn tới data.yaml (định dạng YOLO segmentation)')
    parser.add_argument('--model_size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--patience', type=int, default=20)
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else 'cpu'

    model = create_yolo_model(args.model_size)
    results = train_yolo_model(
        model, args.data_yaml,
        epochs=args.epochs, batch_size=args.batch_size,
        img_size=args.img_size, patience=args.patience, device=device,
    )

    best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    print(f"\n[SAVED] Best model: {best_model_path}")
    print("Copy file này vào models/model_segemnt_yolo.pt để web app dùng được "
          "(xem models/README.md).")

    best_model = load_yolo_checkpoint(best_model_path)
    val_results = evaluate_yolo_model(
        best_model, args.data_yaml, args.img_size, args.batch_size, device,
    )
    print(f"\nBox mAP@0.5: {val_results.box.map50:.4f}")
    print(f"Mask mAP@0.5: {val_results.seg.map50:.4f}")


if __name__ == '__main__':
    main()
