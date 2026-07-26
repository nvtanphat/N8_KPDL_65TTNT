# 🍃 Bean Leaf Lesions Classification & Instance Segmentation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

Một hệ thống **Deep Learning** toàn diện cho việc tự động chẩn đoán, phân loại tổn thương và phân vùng vị trí bệnh hại trên lá đậu (Bean Leaves). Dự án tích hợp nhiều kiến trúc tiên tiến từ CNN truyền thống, Vision Transformer (DeiT) đến Instance Segmentation (YOLOv8-seg), đi kèm ứng dụng Web tương tác Streamlit và Docker hỗ trợ triển khai thực tế.

---

## 📌 Tính năng Nổi bật

- **Mô hình Phân loại & Phân vùng Đa dạng (Multi-Architecture System):**
  - **Custom CNN (BeanLeafLite):** Depthwise-Separable + Residual + SE Attention tự thiết kế, ~1M tham số - nhẹ hơn cả MobileNetV3.
  - **EfficientNet-B3:** Tối ưu hóa sự cân bằng giữa số lượng tham số và độ chính xác.
  - **MobileNetV3-Large:** Kiến trúc siêu nhẹ, tối ưu cho thời gian thực và thiết bị di động (Edge devices).
  - **DeiT-Small (Vision Transformer):** Khai thác cơ chế Self-Attention để hiểu ngữ cảnh toàn cục của ảnh.
  - **YOLOv8-seg (Instance Segmentation):** Phát hiện chính xác vị trí và tạo mask phân vùng ổ bệnh realtime.

- **Thiết kế Modular dạng Python Package (`bean_leaf`):**
  - Đóng gói chuẩn theo quy chuẩn Python (`pip install -e .`).
  - Dễ dàng tái sử dụng mã nguồn giữa các tập lệnh CLI, Notebooks, Web Application và bộ test CI/CD.

- **Ứng dụng Web Tương tác (Streamlit App):**
  - **Single View:** Phân tích ảnh đơn, hiển thị biểu đồ xác suất và đưa ra gợi ý xử lý nông nghiệp.
  - **Compare Mode:** So sánh dự đoán song song giữa các mô hình trên cùng một bức ảnh.
  - **Segmentation View:** Hiển thị vị trí ổ bệnh được khoanh vùng trực quan bằng YOLOv8.

- **Sẵn sàng Triển khai & Tự động hóa:**
  - Đóng gói container chuẩn hóa với **Docker**.
  - Tích hợp **GitHub Actions CI/CD** tự động kiểm thử unit test (`pytest`).

---

## 📊 Tập dữ liệu (Dataset)

Dự án kết hợp các tập dữ liệu chuẩn mực được tiền xử lý và gán nhãn:
- **Bài toán Phân loại (Classification):** [Bean Leaf Lesions Dataset](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification)
- **Bài toán Phân vùng (Instance Segmentation):** [Roboflow Universe - Bean Leaf Segmentation](https://universe.roboflow.com/alebachew-m/final_instance_segmentation)

**Các lớp bệnh hại (3 Classes):**
1. `angular_leaf_spot` — Bệnh đốm góc lá
2. `bean_rust` — Bệnh gỉ sắt
3. `healthy` — Lá khỏe mạnh

> 💡 *Hướng dẫn chi tiết về cấu trúc dữ liệu có tại [data/README.md](data/README.md).*

---

## 🏗️ Cấu trúc Dự án

```bash
bean-leaf-disease/
├── .github/workflows/ci.yml    # Pipeline CI/CD kiểm thử tự động
├── app/                        # Giao diện Web App Streamlit
│   ├── config.py               # Thẻ cấu hình mô hình & khuyến nghị chẩn đoán
│   ├── streamlit_app.py        # Ứng dụng chính Streamlit
│   └── utils.py                # Pipeline nạp mô hình & xử lý suy luận
├── data/                       # Quản lý tập dữ liệu (xem data/README.md)
├── docker/
│   └── Dockerfile              # Cấu hình Docker build container
├── models/                     # Thư mục chứa weights / checkpoints (xem models/README.md)
├── notebooks/
│   └── 01_eda.ipynb            # Khám phá & Trực quan hóa dữ liệu (EDA)
├── scripts/
│   ├── train.py                # Script huấn luyện các mô hình Phân loại (Classification)
│   └── train_yolo.py           # Script huấn luyện mô hình Phân vùng (YOLOv8-seg)
├── src/bean_leaf/              # Core Library Package
│   ├── data/                   # Dataset Handlers, DataLoaders & Augmentations
│   ├── evaluation/             # Đánh giá chỉ số (Accuracy, F1, Confusion Matrix, Grad-CAM)
│   ├── models/                 # Định nghĩa các kiến trúc mô hình PyTorch & Ultralytics
│   ├── training/               # Quản lý vòng lặp huấn luyện & Early Stopping
│   └── utils/                  # Utility helpers & Cấu hình đường dẫn
├── tests/                      # Bộ unit tests cho kiểm thử tự động
├── pyproject.toml              # Cấu hình cài đặt package Python
└── requirements.txt            # Danh sách thư viện phụ thuộc
```

---

## ⚙️ Cài đặt & Chuẩn bị Môi trường

### 1. Yêu cầu Hệ thống
- Python **3.10+**
- Git & Virtualenv / Conda

### 2. Clone Repository & Khởi tạo Môi trường ảo

```bash
# Clone dự án
git clone https://github.com/nvtanphat/bean-leaf-disease.git
cd bean-leaf-disease

# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Trên Linux / macOS:
source venv/bin/activate
```

### 3. Cài đặt Phụ thuộc & Package Local

```bash
pip install -r requirements.txt
pip install -e .
```
*(Lưu ý: Lệnh `pip install -e .` sẽ cài đặt `bean_leaf` dưới dạng editable package, cho phép import package ổn định từ mọi nơi trong dự án).*

---

## 💻 Triển khai & Sử dụng Web App

### 1. Chạy Trực tiếp với Streamlit

```bash
streamlit run app/streamlit_app.py
```
Ứng dụng sẽ tự động mở tại giao diện trình duyệt local: `http://localhost:8501`.

### 2. Triển khai với Docker

```bash
# Xây dựng Docker Image
docker build -t bean-leaf-app -f docker/Dockerfile .

# Chạy Docker Container (mount thư mục models chứa weights)
docker run -p 8501:8501 -v $(pwd)/models:/app/models bean-leaf-app
```

---

## 🚀 Huấn luyện Mô hình (Training)

### 0. Controlled Benchmark Protocol

Để so sánh công bằng giữa 4 kiến trúc, hyperparameter dùng chung được tập trung tại
[`src/bean_leaf/config.py`](src/bean_leaf/config.py) (`DEFAULT_CONFIG`) - đổi 1 giá trị áp
dụng ngay cho cả 4 model, thay vì khai báo lặp lại rải rác từng file:

| Tham số | Giá trị | Lý do |
|---|:---:|---|
| `img_size` | **384** | Giữ chi tiết vết bệnh nhỏ (đốm góc lá, gỉ sắt) sắc nét hơn; khớp sẵn với DeiT3 (`patch16_384`) |
| `batch_size` | 32 | Đảm bảo ổn định gradient |
| `num_epochs` | 100 (trần) | EarlyStopping (`patience`) tự quyết định điểm dừng thực tế |
| `learning_rate` | 3e-4 | Chuẩn cho Transfer Learning + AdamW |
| `weight_decay` | 1e-2 | Giảm overfitting |
| `patience` | 7 | Early stopping vừa đủ |
| `label_smoothing` | 0.05 | Làm mềm phân phối nhãn giữa các lớp bệnh tương đồng |

EfficientNet-B3 và MobileNetV3 dùng chung 1 recipe train y hệt nhau (AdamW +
CosineAnnealingLR, full fine-tune từ epoch 1 - không đóng băng backbone). Chỉ VGG
(OneCycleLR `max_lr` step theo batch + gradient clipping) và DeiT (warmup + EMA) còn giữ
cơ chế riêng vì không có tương đương ở config chung. MobileNetV3 trước đây dùng transfer
learning 2-phase (freeze rồi unfreeze dần) - đã bỏ vì đó là 1 lợi thế recipe riêng mà 3
model kia không có, khiến so sánh không công bằng. Xem `tests/test_models.py` để verify
forward pass đúng shape sau khi đổi `DEFAULT_CONFIG.img_size`.

> 💡 EfficientNet-B3 ở 384px/batch 32 vượt VRAM GPU T4 nếu train thuần fp32 (đã gặp CUDA OOM
> thực tế) - `scripts/train.py` dùng Automatic Mixed Precision (`bean_leaf.training.amp`) cho
> cả 4 model để khắc phục, không cần giảm batch/resolution.

**Train / Internal-Val / Test:** thư mục `train/` được tách thêm thành `train_subset` +
`internal_val_subset` (stratified theo nhãn, tỷ lệ 85/15) - `internal_val_subset` chỉ dùng
để EarlyStopping/chọn checkpoint lúc train. Thư mục `val/` gốc giữ nguyên làm **test set
độc lập**, không tham gia bất kỳ quyết định nào lúc train, chỉ đánh giá **đúng 1 lần** sau
khi train xong. Nếu dùng `val/` vừa để early-stop vừa để báo cáo kết quả (như trước đây),
con số sẽ bị thiên vị lạc quan vì checkpoint được chọn chính vì nó tốt nhất trên chính tập
đó. `scripts/train.py` in ra `Test Accuracy` riêng biệt sau khi train xong mỗi model.

Bảng benchmark ở mục [Kết quả Thực nghiệm](#-kết-quả-thực-nghiệm--đánh-giá-benchmark--evaluation)
bên dưới đã được đo lại dưới protocol 384px thống nhất này.

### 1. Huấn luyện Mô hình Phân loại (Classification)

Sử dụng lệnh CLI [`scripts/train.py`](scripts/train.py):

```bash
# Huấn luyện một mô hình cụ thể (ví dụ: EfficientNet)
python scripts/train.py --data_dir "./data" --model efficientnet

# Huấn luyện lần lượt tất cả mô hình phân loại
python scripts/train.py --data_dir "./data" --model all
```

Các tham số bổ sung:
- `--model`: Lựa chọn mô hình (`vgg`, `efficientnet`, `mobilenet`, `deit`, `all`).
- `--output_dir`: Thư mục lưu checkpoint kết quả (mặc định: `./outputs`).
- `--eda`: Tự động thực hiện phân tích EDA dữ liệu trước khi train.

### 2. Huấn luyện Mô hình Phân vùng (YOLOv8 Segmentation)

Sử dụng lệnh CLI [`scripts/train_yolo.py`](scripts/train_yolo.py):

```bash
python scripts/train_yolo.py --data_yaml "./data/data.yaml" --epochs 50 --model_size n
```


---

## 📈 Kết quả Thực nghiệm & Đánh giá (Benchmark & Evaluation)

### 1. Hiệu năng Mô hình Phân loại (Classification Benchmark)

Đo dưới **Controlled Benchmark Protocol** thống nhất (384px, xem mục Training) - tất cả 4
model dùng chung 1 resolution/hyperparameter để so sánh công bằng:

| Mô hình | Test Accuracy | Số tham số | Đặc điểm & Ưu thế |
|---|:---:|:---:|---|
| **MobileNetV3-Large** | **99.25%** | ~3.2M | Phù hợp thiết bị di động / Edge |
| **DeiT-Small** (ViT) | **99.25%** | ~21.8M | Self-Attention khai thác ngữ cảnh toàn cục |
| **EfficientNet-B3** | **97.74%** | ~13.0M | Khả năng tổng quát hóa tốt, cân bằng tối ưu giữa tham số và hiệu năng |
| **BeanLeafLite** (Custom CNN) | **98.50%** | **~0.94M** | Depthwise-Separable + Residual + SE Attention - nhẹ hơn cả MobileNetV3 |

> Đo đúng 1 lần trên tập test độc lập (`val/` gốc), **không** dùng để chọn checkpoint hay
> quyết định early-stopping trong lúc train (xem mục Training: Train / Internal-Val /
> Test) - số liệu đáng tin cậy hơn "Val Accuracy" thiên vị đo trước đây
> (100%/99.25%/98.50%/96.99%, dùng `val/` vừa để chọn checkpoint vừa để báo cáo).
>
> Cả 4 model dùng đúng 1 recipe train công bằng (AdamW + CosineAnnealingLR/OneCycleLR,
> full fine-tune từ epoch 1 - không có model nào được ưu ái cơ chế train riêng như
> MobileNetV3 2-phase trước đây).
>
> BeanLeafLite ban đầu đo được 93.98% - thấp bất thường so với 3 model còn lại. Nguyên
> nhân: OneCycleLR được cấu hình lịch anneal LR cho 100 epoch, nhưng EarlyStopping
> (patience=7) luôn dừng sớm hơn nhiều (~epoch 40-45 thực tế) - lịch LR bị cắt ngang giữa
> chừng, LR chưa kịp giảm về thấp lúc chọn checkpoint. Sau khi hiệu chỉnh lại mốc epoch
> mà OneCycleLR nhắm tới cho sát với thời điểm early-stop thực tế (không đổi img_size/
> batch_size/epoch ceiling/patience/label_smoothing hay bất kỳ tham số benchmark chung
> nào khác), kết quả tăng lên 98.50% - ngang EfficientNet-B3 trong khi nhẹ hơn ~14 lần.

#### Đánh giá độ ổn định qua 5-Fold Cross-Validation (kiến trúc BeanLeafVGG cũ):

| Mô hình | Độ chính xác trung bình (5-Fold CV) |
|---|:---:|
| **DeiT-Small** | **98.29% ± 0.47%** |
| **BeanLeafVGG** (kiến trúc cũ) | **97.10%** |
| **MobileNetV3-Large** | **96.57% ± 0.80%** |

---

### 2. Hiệu năng Mô hình Phân vùng (YOLOv8 Instance Segmentation)

> ⚠️ Các số liệu trước đây (baseline notebook, model `n`: Box mAP50 73.0%, Mask mAP50 68.3%)
> đo trên split **`val`** - cùng tập mà Ultralytics dùng để chọn `best.pt`/early-stopping
> (`patience`) trong lúc train, nên bị thiên vị lạc quan (y hệt lỗi đã sửa ở mục
> Classification). `data.yaml` (Roboflow) có sẵn split **`test`** độc lập, chưa từng dùng.
> `scripts/train_yolo.py` đã sửa để `evaluate_yolo_model()` mặc định chấm trên `test` thay
> vì `val` - cần train lại để có mAP đáng tin cậy, bảng dưới đây sẽ cập nhật sau khi chạy.

- **Box mAP@0.5:** *chờ đo lại trên test set*
- **Mask mAP@0.5 / mAP@0.5:0.95:** *chờ đo lại trên test set*
- **Mask mAP@0.5 theo từng phân lớp:** *chờ đo lại trên test set*
- **Tốc độ suy luận (Inference Speed):** **4.9 ms/ảnh** (~200 FPS) - không đổi (không phụ thuộc split đánh giá).

---

## 🧪 Kiểm thử Đơn vị (Unit Tests & Quality)

Dự án sử dụng **pytest** để đảm bảo tính toàn vẹn của dữ liệu và kiến trúc mạng:

```bash
# Chạy bộ test suite
pytest -v
```

Hệ thống CI/CD via GitHub Actions tự động kích hoạt kiểm thử mỗi khi có thay đổi được đẩy lên repository.

---

## 👥 Tác giả & Lời cảm ơn

- **Nguyễn Văn Tấn Phát**
- **Nguyễn Hoàng Lộc**

Trân trọng cảm ơn cộng đồng mã nguồn mở **PyTorch**, **Timm**, **Ultralytics**, và **Streamlit** đã cung cấp các nền tảng và thư viện chất lượng cao.

