from PIL import Image
from torchvision import transforms

from bean_leaf.data.dataset import create_df, get_train_val_test_loaders


def _make_image_folder(root, class_counts):
    """Tạo cấu trúc <root>/<class>/*.jpg với số ảnh theo class_counts (dict)."""
    for cls, count in class_counts.items():
        cls_dir = root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            Image.new("RGB", (8, 8)).save(cls_dir / f"img{i}.jpg")


def test_get_train_val_test_loaders_splits_train_and_keeps_val_as_test(tmp_path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    _make_image_folder(train_dir, {"healthy": 20, "bean_rust": 20})
    _make_image_folder(val_dir, {"healthy": 5, "bean_rust": 5})

    tf = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
    train_loader, internal_val_loader, test_loader = get_train_val_test_loaders(
        str(train_dir), str(val_dir), tf, tf, batch_size=4, internal_val_ratio=0.2,
    )

    # train_dir (40 ảnh) phải được tách thành train_subset + internal_val_subset,
    # không được lẫn với test set (val_dir gốc, 10 ảnh, giữ nguyên không bị tách).
    assert len(train_loader.dataset) + len(internal_val_loader.dataset) == 40
    assert len(internal_val_loader.dataset) == 8  # 20% của 40
    assert len(test_loader.dataset) == 10

    # internal_val không được trùng ảnh với train (khác index trong cùng ImageFolder gốc)
    train_indices = set(train_loader.dataset.indices)
    internal_val_indices = set(internal_val_loader.dataset.indices)
    assert train_indices.isdisjoint(internal_val_indices)


def test_create_df_reads_class_subfolders(tmp_path):
    for cls in ("healthy", "bean_rust"):
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        Image.new("RGB", (8, 8)).save(cls_dir / "img0.jpg")

    df = create_df(str(tmp_path))

    assert len(df) == 2
    assert set(df["category_str"]) == {"healthy", "bean_rust"}


def test_create_df_missing_dir_returns_empty():
    df = create_df("this/path/does/not/exist")
    assert len(df) == 0
