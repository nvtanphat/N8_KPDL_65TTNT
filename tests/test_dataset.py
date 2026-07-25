from PIL import Image

from bean_leaf.data.dataset import create_df


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
