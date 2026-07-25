import os

print("[1/3] Cloning repository từ GitHub...")
os.system("git clone https://github.com/nvtanphat/bean-leaf-disease.git")
os.chdir("bean-leaf-disease")

print("[2/3] Cài đặt dependencies...")
os.system("pip install -r requirements.txt -q")
os.system("pip install -e . -q")

print("[3/3] Bắt đầu train bằng GPU trên Kaggle...")
# Chạy train 4 model với dataset sẵn có trên Kaggle (mount qua dataset_sources trong kernel-metadata.json)
os.system("python scripts/train.py --data_dir /kaggle/input/bean-leaf-lesions-classification --model all")
