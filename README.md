# 🍃 Bean Leaf Lesions Classification & Segmentation

> **Tác giả:** Nguyễn Văn Tấn Phát & Nguyễn Hoàng Lộc  
> *(Ghi chú: Dự án được khởi đầu và phát triển từ đồ án môn học Khai phá dữ liệu - Data Mining)*

---

## 📌 Giới thiệu (Overview)

Dự án ứng dụng các mô hình **Deep Learning** tiên tiến (bao gồm Convolutional Neural Networks - CNNs, Vision Transformers - ViT, và Instance Segmentation) để tự động phát hiện, phân loại tổn thương và phân vùng vị trí bệnh hại trên lá đậu (Bean Leaves).

### ✨ Đặc điểm nổi bật
- **PyTorch Ecosystem:** Toàn bộ quy trình huấn luyện và suy luận được xây dựng đồng bộ trên nền tảng PyTorch và Torchvision.
- **Kiến trúc Modular & Scalable:** Gói mã nguồn `src/bean_leaf/` được thiết kế dạng Python Package chuẩn (`pip install -e .`), giúp tái sử dụng linh hoạt giữa CLI, Notebooks, Web App và CI/CD.
- **Hỗ trợ Đa mô hình (Multi-architecture):** Huấn luyện & đánh giá 5 kiến trúc từ baseline CNN tự dựng, MobileNetV3 (Lightweight), EfficientNet-B3, Vision Transformer (DeiT) tới YOLOv8-seg (Segmentation).
- **Ứng dụng Web Tương tác (Streamlit & Docker):** Giao diện Streamlit hỗ trợ tải ảnh, dự đoán real-time, biểu đồ xác suất, gợi ý xử lý và chế độ so sánh song song (*Compare Mode*) giữa các mô hình.

---

## 📊 Dataset (Tập dữ liệu)

Dự án sử dụng **Bean Leaf Lesions Dataset**:
- **Nguồn dữ liệu:** [Kaggle - Bean Leaf Lesions Classification](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification)
- **Dữ liệu Phân vùng (Instance Segmentation):** [Roboflow Universe - Bean Leaf Segmentation](https://universe.roboflow.com/alebachew-m/final_instance_segmentation)
- **Các lớp bài toán (3 classes):**
  1. `angular_leaf_spot` (Bệnh đốm góc lá)
  2. `bean_rust` (Bệnh gỉ sắt)
  3. `healthy` (Lá khỏe mạnh)

> 💡 **Chi tiết về cách chuẩn bị dữ liệu:** Xem hướng dẫn tại [data/README.md](data/README.md).

---

## 🏗️ Kiến trúc các Mô hình (Model Architectures)

| Mô hình | Loại | Mô tả / Đặc điểm | PyTorch Module |
|---|---|---|---|
| **BeanLeafVGG** | CNN Custom | Mạng CNN tự thiết kế (baseline model) | [`bean_leaf.models.vgg_custom`](src/bean_leaf/models/vgg_custom.py) |
| **EfficientNet-B3** | CNN Transfer | Tối ưu hóa sự cân bằng giữa độ chính xác và tham số | [`bean_leaf.models.efficientnet`](src/bean_leaf/models/efficientnet.py) |
| **MobileNetV3-Large** | CNN Lightweight | Kiến trúc siêu nhẹ, tối ưu cho thiết bị di động / Edge | [`bean_leaf.models.mobilenetv3`](src/bean_leaf/models/mobilenetv3.py) |
| **DeiT-Small** | Vision Transformer | Data-efficient Image Transformer (sử dụng `timm`) | [`bean_leaf.models.deit`](src/bean_leaf/models/deit.py) |
| **YOLOv8-seg** | Instance Segmentation | Phát hiện vị trí & vẽ mask phân vùng vùng bệnh (Ultralytics) | [`bean_leaf.models.yolo_seg`](src/bean_leaf/models/yolo_seg.py) |

---

## 📁 Cấu trúc Dự án (Project Structure)

```bash
bean-leaf-disease/
├── .github/workflows/ci.yml    # CI workflow tự động kiểm thử (pytest)
├── app/                        # Ứng dụng Web Streamlit
│   ├── config.py               # Cấu hình danh sách mô hình & khuyến nghị
│   ├── streamlit_app.py        # Giao diện ứng dụng chính
│   └── utils.py                # Wrapper load model và suy luận (inference)
├── data/                       # Thư mục dữ liệu (xem data/README.md)
├── docker/
│   └── Dockerfile              # Dockerfile đóng gói Web App
├── models/                     # Checkpoint đã train (xem models/README.md)
├── notebooks/
│   └── 01_eda.ipynb            # Notebook khám phá và trực quan hóa dữ liệu (EDA)
├── scripts/
│   ├── train.py                # CLI entrypoint train 4 mô hình Classification
│   └── train_yolo.py           # CLI entrypoint train YOLOv8 Segmentation
├── src/bean_leaf/              # Core Package (cài qua `pip install -e .`)
│   ├── data/                   # Dataset class, DataLoaders, Augmentations
│   ├── evaluation/             # Metrics (Accuracy, F1, Confusion Matrix, Grad-CAM)
│   ├── models/                 # Định nghĩa các kiến trúc mô hình (PyTorch)
│   ├── training/               # EarlyStopping, Trainer Utilities
│   └── utils/                  # Quản lý đường dẫn và cấu hình
├── tests/                      # Bộ kiểm thử đơn vị (Pytest smoke tests)
├── pyproject.toml              # Cấu hình package Python
└── requirements.txt            # Danh sách các thư viện phụ thuộc
```

---

## ⚙️ Cài đặt (Installation)

### 1. Clone Repository & Tạo Môi trường ảo

```bash
git clone https://github.com/nvtanphat/bean-leaf-disease.git
cd bean-leaf-disease

# Tạo venv (khuyến nghị Python 3.10+)
python -m venv venv

# Kích hoạt venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Cài đặt Dependencies & Local Package

```bash
pip install -r requirements.txt
pip install -e .
```
> 🔹 **Lưu ý:** Lệnh `pip install -e .` cho phép import trực tiếp package `bean_leaf` từ bất kỳ đâu (`scripts/`, `app/`, `tests/`) mà không cần tùy chỉnh `sys.path`.

---

## 🚀 Huấn luyện Mô hình (Training)

### 1. Huấn luyện các mô hình Classification (PyTorch)

Sử dụng script [`scripts/train.py`](scripts/train.py) để train các mô hình Classification:

```bash
python scripts/train.py --data_dir "./data" --model [tên_model] [options]
```

**Các tham số chính:**
- `--data_dir`: Đường dẫn thư mục chứa dataset (`train/`, `val/`, `test/`).
- `--model`: Chọn mô hình cần huấn luyện: `vgg`, `efficientnet`, `mobilenet`, `deit`, hoặc `all` (train lần lượt tất cả).
- `--epochs`: Số lượng epochs (mặc định: `25`).
- `--batch_size`: Batch size (mặc định: `32`).
- `--output_dir`: Đường dẫn lưu checkpoints (mặc định: `./outputs`).
- `--eda`: (Option) Tự động chạy phân tích dữ liệu trước khi huấn luyện.

**Ví dụ:**
```bash
# Train duy nhất mô hình EfficientNet-B3
python scripts/train.py --data_dir "./data" --model efficientnet --epochs 30

# Train toàn bộ 4 mô hình phân loại
python scripts/train.py --data_dir "./data" --model all
```

> 📌 Checkpoint mô hình xuất ra sẽ nằm tại `outputs/<model_name>/best_<model_name>_model.pth`. Sau khi train xong, copy checkpoint vào thư mục `models/` để Web App có thể đọc (xem chi tiết tại [models/README.md](models/README.md)).

---

### 2. Huấn luyện mô hình YOLOv8 Segmentation

Sử dụng script [`scripts/train_yolo.py`](scripts/train_yolo.py) với dataset định dạng Ultralytics (`data.yaml`):

```bash
python scripts/train_yolo.py --data_yaml "./data/data.yaml" --epochs 50 --model_size n
```

---

## 💻 Ứng dụng Web & Triển khai (Inference & Deployment)

### 1. Chạy Web App trực tiếp với Streamlit

```bash
streamlit run app/streamlit_app.py
```

Các tính năng chính trên Web App:
- 🖼️ **Single View:** Phân tích ảnh với 1 mô hình đã chọn, hiển thị xác suất dự đoán và thông tin điều trị bệnh.
- ⚖️ **Compare Mode:** Chế độ so sánh song song dự đoán của tất cả mô hình trên cùng 1 bức ảnh.
- 🎯 **Segmentation View:** Chế độ phân vùng phát hiện vị trí tổn thương đốm bệnh bằng YOLOv8-seg.

---

### 2. Triển khai ứng dụng qua Docker

Đóng gói và khởi chạy Web App bằng Docker container:

```bash
# Build Docker image
docker build -t bean-leaf-app -f docker/Dockerfile .

# Khởi chạy container trên port 8501
docker run -p 8501:8501 -v $(pwd)/models:/app/models bean-leaf-app
```
Sau đó truy cập ứng dụng tại: `http://localhost:8501`.

---

## 📈 Kết quả Thực nghiệm & Đánh giá (Experimental Results)

Dưới đây là tổng hợp kết quả đánh giá thực nghiệm chi tiết trích xuất từ báo cáo thử nghiệm trên cùng tập dữ liệu kiểm thử:

### 1. Phân loại bệnh (Classification Performance)

#### Bảng so sánh tổng hợp các chỉ số hiệu năng trên tập Kiểm thử:

| Mô hình (Model) | Accuracy | Precision | Recall | F1-Score | Số tham số | Đặc điểm chính |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **DeiT-Small** (ViT) | **100.00%** | **100.00%** | **100.00%** | **100.00%** | ~21.8M | Đạt kết quả SOTA tuyệt đối nhờ cơ chế Self-Attention khai thác ngữ cảnh toàn cục. |
| **BeanLeafVGG** (Custom CNN) | **99.25%** | **99.00%** | **99.00%** | **99.00%** | ~4.7M | CNN tự xây dựng từ scratch với kết quả xuất sắc dù không dùng weights pretrained. |
| **EfficientNet-B3** | **99.25%** | **99.00%** | **99.00%** | **99.00%** | ~13.0M | Khả năng tổng quát hóa tốt, cân bằng giữa độ chính xác và số lượng tham số. |
| **MobileNetV3-Large** | **98.50%** | **98.00%** | **98.00%** | **98.00%** | ~3.2M | Mô hình siêu nhẹ, tối ưu cho ứng dụng thời gian thực và thiết bị di động / edge. |

#### Độ ổn định qua 5-Fold Cross-Validation:

| Mô hình (Model) | Độ chính xác trung bình (5-Fold CV) |
|---|:---:|
| **DeiT-Small** | **98.29% ± 0.47%** |
| **BeanLeafVGG** (CNN tự xây) | **97.10%** |
| **MobileNetV3-Large** | **96.57% ± 0.80%** |

---

### 2. Phân vùng bệnh (YOLOv8-seg Instance Segmentation)

- **Box mAP@0.5:** 68.0%
- **Mask mAP@0.5:** 68.0% | **Mask mAP@0.5:0.95:** 48.4%
- **Mask mAP@0.5 theo từng lớp:**
  - `healthy` (Lá khỏe mạnh): **93.0%** (Ranh giới viền lá rõ ràng)
  - `angular_leaf_spot` (Đốm góc): **65.0%** (Đốm vết bệnh lớn dạng hình đa giác)
  - `bean_rust` (Gỉ sắt): **42.0%** (Đốm nhỏ li ti, màu sắc tương đồng nền lá)
- **Tốc độ suy luận (Inference Speed):** **4.9 ms/ảnh** (~200 FPS), đáp ứng hoàn hảo yêu cầu realtime trên thiết bị di động.

---

## 🧪 Kiểm thử Đơn vị (Unit Testing)

Dự án tích hợp bộ test tự động sử dụng **pytest** để kiểm tra tính toàn vẹn của dataset và các mô hình PyTorch (shape output, loss computation):

```bash
pytest -v
```

> 🔄 **CI/CD:** Hệ thống GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) tự động kích hoạt `pytest` trên mỗi lần push hoặc tạo Pull Request vào nhánh `main`.

---

## 📝 Báo cáo Đồ án

Báo cáo đầy đủ với phân tích định tính & định lượng, bảng so sánh metrics (Accuracy, Precision, Recall, F1-Score) và trực quan Grad-CAM đã được tổng hợp chi tiết tại mục [Kết quả Thực nghiệm](#-kết-quả-thực-nghiệm--đánh-giá-experimental-results).

---

## 📜 License & Lời cảm ơn

Dự án ban đầu được khởi xướng từ đồ án môn học Khai phá dữ liệu (Data Mining) và tiếp tục được nâng cấp thành một hệ thống dự án độc lập. Cảm ơn các cộng đồng mã nguồn mở **PyTorch**, **Timm**, **Ultralytics**, và **Streamlit** đã cung cấp các nền tảng tuyệt vời.
