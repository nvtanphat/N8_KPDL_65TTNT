# 🍃 Bean Leaf Lesions Classification & Instance Segmentation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

Một hệ thống **Deep Learning** toàn diện cho việc tự động chẩn đoán, phân loại tổn thương và phân vùng vị trí bệnh hại trên lá đậu (Bean Leaves). Dự án tích hợp nhiều kiến trúc tiên tiến từ CNN truyền thống, Vision Transformer (DeiT) đến Instance Segmentation (YOLOv8-seg), đi kèm ứng dụng Web tương tác Streamlit và Docker hỗ trợ triển khai thực tế.

---

## 📌 Tính năng Nổi bật

- **Mô hình Phân loại & Phân vùng Đa dạng (Multi-Architecture System):**
  - **Custom CNN (BeanLeafVGG):** Kiến trúc CNN nhẹ tự thiết kế làm baseline.
  - **EfficientNet-B3:** Tối ưu hóa sự cân bằng giữa số lượng tham số và độ chính xác.
  - **MobileNetV3-Large:** Kiến trúc siêu nhẹ, tối ưu cho thời gian thực và thiết bị di động (Edge devices).
  - **DeiT-Small (Vision Transformer):** Khai thác cơ chế Self-Attention cho độ chính xác SOTA (**99.25%**).
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

Kết quả đánh giá độc lập trên tập kiểm thử (Test Set):

| Mô hình | Validation Accuracy | Số tham số | Đặc điểm & Ưu thế |
|---|:---:|:---:|---|
| **DeiT-Small** (ViT) | **99.25%** | ~21.8M | Đạt kết quả SOTA nhờ cơ chế Self-Attention khai thác ngữ cảnh toàn cục |
| **BeanLeafVGG** (Custom CNN) | **98.50%** | ~4.7M | Kiến trúc CNN tùy chỉnh tối ưu, đạt độ chính xác cao dù train từ đầu |
| **EfficientNet-B3** | **96.24%** | ~13.0M | Khả năng tổng quát hóa tốt, cân bằng tối ưu giữa tham số và hiệu năng |
| **MobileNetV3-Large** | **94.74%** | ~3.2M | Kiến trúc siêu nhẹ, đáp ứng tức thì cho thiết bị di động / Edge |

#### Đánh giá độ ổn định qua 5-Fold Cross-Validation:

| Mô hình | Độ chính xác trung bình (5-Fold CV) |
|---|:---:|
| **DeiT-Small** | **98.29% ± 0.47%** |
| **BeanLeafVGG** | **97.10%** |
| **MobileNetV3-Large** | **96.57% ± 0.80%** |

---

### 2. Hiệu năng Mô hình Phân vùng (YOLOv8 Instance Segmentation)

- **Box mAP@0.5:** 68.0%
- **Mask mAP@0.5:** 68.0% | **Mask mAP@0.5:0.95:** 48.4%
- **Mask mAP@0.5 theo từng phân lớp:**
  - `healthy`: **93.0%** (Ranh giới lá phân biệt rõ ràng)
  - `angular_leaf_spot`: **65.0%** (Vùng bệnh đốm dạng góc đa giác)
  - `bean_rust`: **42.0%** (Đốm nhỏ rải rác)
- **Tốc độ suy luận (Inference Speed):** **4.9 ms/ảnh** (~200 FPS), phục vụ tốt cho phân tích thời gian thực.

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

