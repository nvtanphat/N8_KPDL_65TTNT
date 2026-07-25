# Data

Thư mục này chứa dataset khi train, **không commit ảnh vào git** (đã khai báo trong `.gitignore`).

Cấu trúc cần có:

```
data/
├── train/
│   ├── healthy/
│   ├── angular_leaf_spot/
│   └── bean_rust/
└── val/
    ├── healthy/
    ├── angular_leaf_spot/
    └── bean_rust/
```

Nguồn dataset: [Kaggle - Bean Leaf Lesions Classification](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification)

## Tải tự động qua Kaggle CLI

1. Cấu hình Kaggle API credential (chỉ cần làm 1 lần, không tự động hoá được vì gắn với
   tài khoản cá nhân): tạo token tại [kaggle.com/settings](https://www.kaggle.com/settings) →
   *Create New Token*, tải về `kaggle.json`, đặt vào `~/.kaggle/kaggle.json`
   (Windows: `C:\Users\<user>\.kaggle\kaggle.json`) — hoặc set env `KAGGLE_USERNAME`/`KAGGLE_KEY`.
2. Tải trực tiếp bằng `scripts/train.py --download` (tự tải nếu `data/train` chưa tồn tại,
   bỏ qua nếu đã có sẵn), hoặc tải riêng không train:

```bash
python -c "from bean_leaf.data.kaggle_download import download_dataset; download_dataset('data')"
```

Xem [README.md](../README.md#huấn-luyện-mô-hình-training) mục Training để chạy cả tải + train
bằng 1 lệnh.
