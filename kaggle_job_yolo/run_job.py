import os
import subprocess
import sys

# stdout mặc định bị buffer khi không gắn với TTY (đúng môi trường chạy kernel Kaggle) -
# nếu script crash giữa chừng, output chưa flush sẽ mất trắng, log tải về sẽ rỗng.
sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Key public đi kèm dataset Roboflow (xem notebooks/06_yolo_segmentation.ipynb) - chỉ dùng để
# tải đúng dataset segmentation công khai này, không phải secret riêng tư. Override qua env
# ROBOFLOW_API_KEY nếu key này hết hạn hoặc bạn muốn dùng key của workspace riêng.
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "Efp5ZmBckO5PsBl99j41")

# Tinh chỉnh so với baseline (notebook gốc dùng "n" - nano, nhẹ nhất nhưng mAP thấp nhất):
# "s" (small) cho mAP cao hơn rõ rệt với chi phí thêm không đáng kể trên GPU T4.
MODEL_SIZE = os.environ.get("YOLO_MODEL_SIZE", "s")
EPOCHS = int(os.environ.get("YOLO_EPOCHS", "100"))
PATIENCE = int(os.environ.get("YOLO_PATIENCE", "20"))


def run(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


print("[1/4] Cloning repository từ GitHub...", flush=True)
run("git clone https://github.com/nvtanphat/bean-leaf-disease.git")
os.chdir("bean-leaf-disease")

print("[2/4] Cài đặt dependencies...", flush=True)
run("pip install -r requirements.txt -q")
run("pip install -e . -q")
run("pip install roboflow -q")

print("[3/4] Tải dataset segmentation từ Roboflow...", flush=True)
from roboflow import Roboflow  # noqa: E402 (cần cài xong ở bước trên mới import được)

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("alebachew-m").project("final_instance_segmentation")
version = project.version(1)
dataset = version.download("yolov8")
data_yaml = os.path.join(dataset.location, "data.yaml")
print(f"Dataset location: {dataset.location}", flush=True)
run(f"cat {data_yaml}")

print(f"[4/4] Train YOLOv8{MODEL_SIZE}-seg ({EPOCHS} epochs, patience={PATIENCE}) trên GPU Kaggle...", flush=True)
run(
    f"python -u scripts/train_yolo.py --data_yaml {data_yaml} "
    f"--model_size {MODEL_SIZE} --epochs {EPOCHS} --patience {PATIENCE}"
)
