import os

# Thư mục gốc của repo (2 cấp lên từ src/bean_leaf/utils/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def get_default_output_dir():
    """
    Thư mục lưu checkpoint khi train, theo thứ tự ưu tiên:
    1. Biến môi trường BEAN_LEAF_OUTPUT_DIR (để override trên máy/CI khác nhau)
    2. <repo_root>/outputs (mặc định)
    """
    return os.environ.get('BEAN_LEAF_OUTPUT_DIR', os.path.join(REPO_ROOT, 'outputs'))
