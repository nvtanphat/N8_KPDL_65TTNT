import os
import subprocess
import sys

# stdout mặc định bị buffer khi không gắn với TTY (đúng môi trường chạy kernel Kaggle) -
# nếu script crash giữa chừng, output chưa flush sẽ mất trắng, log tải về sẽ rỗng.
# Bật line-buffering ngay từ đầu để mọi print() đều được ghi ra log kịp thời.
sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("PYTHONUNBUFFERED", "1")

KAGGLE_INPUT = "/kaggle/input"


def run(cmd):
    print(f"$ {cmd}", flush=True)
    # check=True: lỗi ở bất kỳ bước nào sẽ raise ngay với traceback rõ ràng,
    # thay vì os.system() im lặng bỏ qua và chạy tiếp bước sau.
    subprocess.run(cmd, shell=True, check=True)


def find_dataset_dir(max_depth=4):
    """
    Tự dò thư mục dataset đã mount trong /kaggle/input, đệ quy tối đa max_depth cấp.
    Kernel tạo qua API (dataset_sources trong kernel-metadata.json) có thể mount lồng
    thêm 1 cấp trung gian kiểu /kaggle/input/datasets/<owner>/<slug>/..., khác với
    /kaggle/input/<slug>/... như trong môi trường notebook UI thông thường - nên
    không thể chỉ tìm ở cấp 1. Trả về thư mục đầu tiên (đệ quy) có chứa 'train/'.
    """
    if not os.path.isdir(KAGGLE_INPUT):
        raise RuntimeError(f"Không thấy {KAGGLE_INPUT} - kernel có bật dataset_sources chưa?")

    all_dirs = []
    for root, dirs, _files in os.walk(KAGGLE_INPUT):
        depth = root[len(KAGGLE_INPUT):].count(os.sep)
        if depth >= max_depth:
            dirs[:] = []  # không đi sâu hơn nữa
            continue
        all_dirs.append(root)
        if "train" in dirs:
            return root

    raise RuntimeError(
        f"Không tìm thấy thư mục nào chứa 'train/' bên trong (đã quét {len(all_dirs)} thư mục "
        f"trong {KAGGLE_INPUT}, tối đa {max_depth} cấp). Danh sách đã quét: {all_dirs}. "
        f"Kiểm tra lại dataset_sources trong kernel-metadata.json."
    )


print("[1/4] Cloning repository từ GitHub...", flush=True)
run("git clone https://github.com/nvtanphat/bean-leaf-disease.git")
os.chdir("bean-leaf-disease")

print("[2/4] Cài đặt dependencies...", flush=True)
run("pip install -r requirements.txt -q")
run("pip install -e . -q")

print("[3/4] Dò đường dẫn dataset đã mount...", flush=True)
data_dir = find_dataset_dir()
print(f"Dùng --data_dir {data_dir}", flush=True)
run(f"ls -la {data_dir}/train")

print("[4/4] Bắt đầu train 4 model bằng GPU trên Kaggle...", flush=True)
run(f"python -u scripts/train.py --data_dir {data_dir} --model all")
