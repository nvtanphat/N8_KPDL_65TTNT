import os

import pandas as pd
from torch.utils.data import DataLoader
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


def get_dataloaders(train_dir, val_dir, train_transform, val_transform, batch_size, num_workers=0):
    """
    Tạo train/val DataLoader từ 2 thư mục dạng ImageFolder (train/<class>/*.jpg),
    đúng cách mà tất cả notebook (04, 05, 03) đang load dữ liệu.
    """
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
