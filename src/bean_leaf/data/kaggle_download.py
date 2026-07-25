"""
Tải Bean Leaf Lesions Dataset từ Kaggle qua Kaggle CLI (`kaggle datasets download`).

Yêu cầu cài `kaggle` (có trong requirements.txt) và đã cấu hình API credential trước
(~/.kaggle/kaggle.json hoặc env KAGGLE_USERNAME/KAGGLE_KEY) - đây là bước không thể tự
động hoá vì gắn với tài khoản cá nhân. Xem hướng dẫn tạo token:
https://github.com/Kaggle/kaggle-api#api-credentials
"""
import os
import subprocess
import sys

KAGGLE_DATASET = "marquis03/bean-leaf-lesions-classification"


def download_dataset(dest_dir):
    """
    Tải + giải nén dataset vào dest_dir, kết quả có dest_dir/train và dest_dir/val
    (đúng cấu trúc mà scripts/train.py cần). Bỏ qua nếu dest_dir/train đã tồn tại,
    để có thể gọi lại nhiều lần (vd trong CI) mà không tải lại mỗi lần.
    """
    train_dir = os.path.join(dest_dir, 'train')
    if os.path.isdir(train_dir):
        print(f"[SKIP] Dataset đã tồn tại tại {train_dir}")
        return

    os.makedirs(dest_dir, exist_ok=True)
    print(f"[DOWNLOAD] Tải '{KAGGLE_DATASET}' từ Kaggle vào {dest_dir} ...")

    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", dest_dir, "--unzip"],
            check=True,
        )
    except FileNotFoundError:
        print(
            "Không tìm thấy lệnh 'kaggle'. Cài bằng: pip install kaggle\n"
            "rồi cấu hình API token tại ~/.kaggle/kaggle.json "
            "(xem https://github.com/Kaggle/kaggle-api#api-credentials)",
            file=sys.stderr,
        )
        raise
    except subprocess.CalledProcessError:
        print(
            "Tải dataset thất bại - kiểm tra lại Kaggle API credential "
            "(~/.kaggle/kaggle.json hoặc env KAGGLE_USERNAME/KAGGLE_KEY).",
            file=sys.stderr,
        )
        raise

    if not os.path.isdir(train_dir):
        raise RuntimeError(
            f"Tải xong nhưng không thấy {train_dir} - cấu trúc dataset trên Kaggle "
            "có thể đã đổi, kiểm tra lại thủ công."
        )

    print(f"[DONE] Dataset đã sẵn sàng tại {dest_dir}")
