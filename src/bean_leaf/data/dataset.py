import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets


def create_df(data_dir):
    """
    Tạo DataFrame từ cấu trúc thư mục (mỗi subfolder là 1 class). Dùng cho EDA.
    """
    filepaths = []
    labels = []
    if os.path.exists(data_dir):
        for category in os.listdir(data_dir):
            category_path = os.path.join(data_dir, category)
            if os.path.isdir(category_path):
                for img_name in os.listdir(category_path):
                    filepaths.append(os.path.join(category_path, img_name))
                    labels.append(category)
    return pd.DataFrame({'path_full': filepaths, 'category_str': labels})


def get_train_val_test_loaders(train_dir, val_dir, train_transform, val_transform, batch_size,
                                internal_val_ratio=0.15, seed=42, num_workers=0):
    """
    Tách thư mục train (ImageFolder) thành train_subset + internal_val_subset (stratified
    theo nhãn) - internal_val chỉ dùng để early-stopping/chọn checkpoint lúc train, KHÔNG
    dùng để báo cáo kết quả. Thư mục val gốc giữ nguyên làm TEST SET độc lập: không tham
    gia bất kỳ quyết định nào trong lúc train (không ảnh hưởng early-stopping, không chọn
    checkpoint theo nó), chỉ đánh giá đúng 1 lần sau khi train xong - nếu dùng val để vừa
    early-stop vừa báo cáo, con số báo cáo sẽ bị thiên vị (chọn đúng checkpoint tốt nhất
    trên chính tập đó rồi lại lấy điểm của tập đó làm "kết quả cuối cùng").

    2 instance ImageFolder riêng biệt cùng trỏ vào train_dir để train_subset dùng
    augmentation (train_transform) còn internal_val_subset dùng transform không augment
    (val_transform) - ImageFolder liệt kê file theo thứ tự cố định trong 1 lần chạy nên
    index từ train_test_split áp dụng nhất quán cho cả 2 instance.
    """
    full_train_aug = datasets.ImageFolder(train_dir, transform=train_transform)
    full_train_plain = datasets.ImageFolder(train_dir, transform=val_transform)

    labels = [label for _, label in full_train_aug.samples]
    train_idx, internal_val_idx = train_test_split(
        range(len(labels)), test_size=internal_val_ratio, stratify=labels, random_state=seed,
    )

    train_dataset = Subset(full_train_aug, train_idx)
    internal_val_dataset = Subset(full_train_plain, internal_val_idx)
    test_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    internal_val_loader = DataLoader(
        internal_val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, internal_val_loader, test_loader


def _make_loaders(full_train_aug, full_train_plain, test_dataset, train_idx, internal_val_idx,
                  batch_size, num_workers):
    """Dựng 3 DataLoader từ index đã chia - dùng chung cho holdout và k-fold."""
    train_loader = DataLoader(
        Subset(full_train_aug, train_idx), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    internal_val_loader = DataLoader(
        Subset(full_train_plain, internal_val_idx), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, internal_val_loader, test_loader


def get_kfold_loaders(train_dir, val_dir, train_transform, val_transform, batch_size,
                      n_splits=5, seed=42, num_workers=0):
    """
    Generator yield (fold_idx, train_loader, internal_val_loader, test_loader, internal_val_idx)
    cho từng fold của StratifiedKFold trên train_dir.

    Khác get_train_val_test_loaders ở chỗ chia train_dir thành n_splits phần thay vì 1 lần
    holdout 85/15: mỗi ảnh trong train_dir được làm internal-val đúng 1 lần, nên gộp dự đoán
    của tất cả fold lại (out-of-fold) sẽ có ước lượng accuracy trên TOÀN BỘ train_dir thay vì
    chỉ trên 15% - khoảng tin cậy hẹp hơn nhiều so với test set 133 ảnh.

    val_dir vẫn giữ nguyên vai trò TEST SET độc lập và được đánh giá lại ở từng fold: độ lệch
    chuẩn của test accuracy giữa các fold chính là thước đo "kết quả dao động bao nhiêu giữa
    các lần chạy" - thứ mà 1 lần chạy đơn lẻ không thể nói được.

    internal_val_idx được trả về để map dự đoán out-of-fold ngược lại đúng ảnh gốc
    (internal_val_loader dùng shuffle=False nên thứ tự dự đoán khớp thứ tự index này).
    """
    full_train_aug = datasets.ImageFolder(train_dir, transform=train_transform)
    full_train_plain = datasets.ImageFolder(train_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    labels = [label for _, label in full_train_aug.samples]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, internal_val_idx) in enumerate(skf.split(range(len(labels)), labels), start=1):
        train_loader, internal_val_loader, test_loader = _make_loaders(
            full_train_aug, full_train_plain, test_dataset,
            train_idx, internal_val_idx, batch_size, num_workers,
        )
        yield fold, train_loader, internal_val_loader, test_loader, internal_val_idx
