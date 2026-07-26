# 🍃 Bean Leaf Lesions Classification & Instance Segmentation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![CI/CD](https://github.com/nvtanphat/bean-leaf-disease/actions/workflows/ci.yml/badge.svg)](https://github.com/nvtanphat/bean-leaf-disease/actions)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Hệ thống **Deep Learning** toàn diện cho việc chẩn đoán, phân loại tổn thương và phân vùng vị trí bệnh hại trên lá đậu (Bean Leaves). Dự án tích hợp các kiến trúc tiên tiến từ CNN truyền thống, Vision Transformer (DeiT), Instance Segmentation (YOLOv8-seg) cho đến kiến trúc tự thiết kế **BeanLeafLite** (~0.94M params), đi kèm ứng dụng Web Streamlit và Docker container hỗ trợ triển khai thực tế.

---

## ✨ Điểm Nổi Bật Nâng Cao (Engineering & Research Highlights)

> [!TIP]
> **1. Kiến trúc Đổi mới Siêu nhẹ (`BeanLeafLite` ~0.94M Params):**  
> Tự thiết kế từ đầu (from scratch) kiến trúc kết hợp *Depthwise-Separable Conv + Residual Skip Connections + SE Channel Attention*. Mô hình đạt **98.50% Test Accuracy** với kích thước siêu nhỏ (~0.94M params) — nhẹ hơn EfficientNet-B3 gấp 14 lần nhưng đạt hiệu năng tương đương.

> [!NOTE]
> **2. Phương pháp luận Đánh giá Độc lập & Minh bạch (Unbiased Benchmark):**  
> Đánh giá nghiêm ngặt theo **Controlled Benchmark Protocol (384px)**. Tách riêng tập `test` độc lập hoàn toàn khỏi tập `internal-val` dùng chọn checkpoint, đảm bảo số liệu báo cáo trung thực, không thiên vị.

> [!IMPORTANT]
> **3. Chuẩn hóa Cấu trúc Kiến trúc Phần mềm (Production-Grade Architecture):**  
> - **Single Source of Truth (`config.py`):** Quản lý siêu tham số tập trung qua Dataclass `DEFAULT_CONFIG`.  
> - **Modular Package:** Đóng gói chuẩn Python (`pip install -e .`).  
> - **MLOps Ready:** Tích hợp Automatic Mixed Precision (AMP), Pytest Unit Tests, Dockerfile và GitHub Actions CI/CD.

---

## 📌 Tính năng Hệ thống

- **Hỗ trợ Đa kiến trúc (Multi-Architecture Ecosystem):**
  - **BeanLeafLite (Custom CNN):** 🛠️ Kiến trúc nhẹ tự thiết kế (~0.94M params, Acc **98.50%**).
  - **MobileNetV3-Large:** ⚡ Tối ưu hóa suy luận thời gian thực cho thiết bị di động / Edge (Acc **97.74%**).
  - **DeiT-Small (Vision Transformer):** 🥇 Khai thác cơ chế Self-Attention khai thác ngữ cảnh toàn cục (Acc **100.00%**).
  - **EfficientNet-B3:** ⚖️ Cân bằng tối ưu giữa tham số và khả năng tổng quát hóa (Acc **98.50%**).
  - **YOLOv8-seg (Instance Segmentation):** 🎯 Khoanh vùng và vẽ mặt nạ (polygon mask) tổn thương đốm lá realtime.

- **Ứng dụng Web Tương tác (Streamlit Interactive App):**
  - 🖼️ **Single View:** Phân tích ảnh đơn, hiển thị biểu đồ xác suất & gợi ý hướng điều trị nông nghiệp.
  - ⚖️ **Compare Mode:** So sánh dự đoán song song của tất cả các mô hình trên cùng một bức ảnh.
  - 🎯 **Segmentation View:** Trực quan hóa mặt nạ phân vùng vị trí ổ bệnh bằng YOLOv8-seg.

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
├── data/                       # Quản lý tập dữ liệu
├── docker/
│   └── Dockerfile              # Cấu hình Docker build container
├── models/                     # Thư mục chứa weights / checkpoints
├── notebooks/
│   └── 01_eda.ipynb            # Khám phá & Trực quan hóa dữ liệu (EDA)
├── scripts/
│   ├── train.py                # Script huấn luyện các mô hình Phân loại (Classification)
│   └── train_yolo.py           # Script huấn luyện mô hình Phân vùng (YOLOv8-seg)
├── src/bean_leaf/              # Core Library Package
│   ├── config.py               # Single Source of Truth cho siêu tham số toàn hệ thống
│   ├── data/                   # Dataset Handlers, DataLoaders & Augmentations
│   ├── evaluation/             # Đánh giá chỉ số (Accuracy, F1, Confusion Matrix, Grad-CAM)
│   ├── models/                 # Định nghĩa các kiến trúc mô hình (PyTorch & Ultralytics)
│   ├── training/               # Quản lý vòng lặp huấn luyện & AMP Mixed Precision
│   └── utils/                  # Utility helpers & Cấu hình đường dẫn
├── tests/                      # Bộ unit tests cho kiểm thử tự động (pytest)
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

# Tạo & kích hoạt môi trường ảo
python -m venv venv
source venv/bin/activate  # Hoặc .\venv\Scripts\Activate.ps1 trên Windows
```

### 3. Cài đặt Phụ thuộc & Package Local

```bash
pip install -r requirements.txt
pip install -e .
```

---

## 💻 Triển khai & Sử dụng Web App

![Giao diện Web App Streamlit Chẩn đoán & Grad-CAM](docs/assets/web_demo.png)

### 1. Chạy Trực tiếp với Streamlit
```bash
streamlit run app/streamlit_app.py
```

### 2. Triển khai với Docker
```bash
docker build -t bean-leaf-app -f docker/Dockerfile .
docker run -p 8501:8501 -v $(pwd)/models:/app/models bean-leaf-app
```

---

## 🚀 Huấn luyện Mô hình (Training)

### 0. Controlled Benchmark Protocol

Để so sánh công bằng giữa 4 kiến trúc, hyperparameter dùng chung được tập trung tại [`src/bean_leaf/config.py`](src/bean_leaf/config.py) (`DEFAULT_CONFIG`) - đổi 1 giá trị áp dụng ngay cho cả 4 model:

| Tham số | Giá trị | Lý do & Ý nghĩa Kỹ thuật |
|---|:---:|---|
| `img_size` | **384** | Giữ chi tiết vết bệnh nhỏ (đốm góc lá, gỉ sắt) sắc nét hơn |
| `batch_size` | 32 | Đảm bảo ổn định gradient |
| `num_epochs` | 100 | EarlyStopping (`patience`) tự quyết định điểm dừng thực tế |
| `learning_rate` | 3e-4 | Chuẩn cho Transfer Learning + AdamW |
| `weight_decay` | 1e-2 | Giảm hiện tượng overfitting |
| `patience` | 7 | Early stopping vừa đủ |
| `label_smoothing` | 0.05 | Làm mềm phân phối nhãn giữa các lớp bệnh tương đồng |

> 💡 **Automatic Mixed Precision (AMP):** EfficientNet-B3 ở 384px/batch 32 vượt VRAM GPU T4 nếu train thuần fp32 — `scripts/train.py` dùng Automatic Mixed Precision (`bean_leaf.training.amp`) cho cả 4 model để khắc phục.

**Quy trình Phân tách Dữ liệu:**
Thư mục `train/` được tách thành `train_subset` + `internal_val_subset` (stratified, tỷ lệ 85/15) — `internal_val_subset` chỉ dùng để EarlyStopping. Thư mục `val/` gốc giữ nguyên làm **test set độc lập**, đánh giá **đúng 1 lần** sau khi train.

### 1. Huấn luyện Mô hình Phân loại (Classification)
```bash
python scripts/train.py --data_dir "./data" --model all
```

### 2. Huấn luyện Mô hình Phân vùng (YOLOv8 Segmentation)
```bash
python scripts/train_yolo.py --data_yaml "./data/data.yaml" --epochs 50 --model_size n
```

---

## 📈 Kết quả Thực nghiệm & Đánh giá (Benchmark & Evaluation)

### 1. Hiệu năng Mô hình Phân loại (Classification Benchmark)
Kết quả đo đạc độc lập trên tập test dưới **Controlled Benchmark Protocol** thống nhất (384px):

| Mô hình | Test Acc | Params | Phân nhóm Tối ưu & Ưu thế Kiến trúc |
|---|:---:|:---:|---|
| **DeiT-Small** (ViT) | **100.00%** | ~21.8M | 🥇 **[Cloud SOTA]** Self-Attention khai thác ngữ cảnh toàn cục |
| **BeanLeafLite** (Custom CNN) | **98.50%** | **~0.94M** | 🛠️ **[Custom Innovation]** Depthwise + Residual + SE — siêu nhẹ |
| **EfficientNet-B3** | **98.50%** | ~13.0M | ⚖️ **[Balanced Standard]** Cân bằng tham số và hiệu năng |
| **MobileNetV3-Large** | **97.74%** | ~3.2M | ⚡ **[Edge / Mobile]** Phù hợp thiết bị di động, suy luận siêu nhanh |

> [!NOTE]
> **Đánh giá Độc lập:** Kết quả được đo đúng 1 lần trên tập test độc lập (`val/` gốc), không dùng để chọn checkpoint, đảm bảo số liệu trung thực. Có thể tái tạo lại bất kỳ lúc nào bằng `python scripts/evaluate.py` (đọc thẳng checkpoint trong `models/`, không cần train lại) — kết quả (kèm precision/recall/F1 từng lớp, confusion matrix) được lưu ra `outputs/evaluation_metrics.json`.

#### Đánh giá độ ổn định qua 5-Fold Cross-Validation:
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

