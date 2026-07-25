# Bean Leaf Lesions Classification (Phân loại Bệnh trên Lá Đậu)

> **Đồ án môn học: Khai phá dữ liệu (Data Mining)**

## Giới thiệu (Overview)

Dự án này tập trung vào việc áp dụng các kỹ thuật **Deep Learning** tiên tiến (CNNs và Vision Transformers) để tự động phân loại các tổn thương bệnh trên lá đậu. Hệ thống giúp hỗ trợ nông dân và các chuyên gia nông nghiệp phát hiện sớm bệnh hại, từ đó đưa ra biện pháp xử lý kịp thời.

Dự án bao gồm quy trình trọn vẹn từ:

1. **EDA & Preprocessing:** Khám phá và xử lý dữ liệu ảnh.
2. **Model Training:** Huấn luyện và tinh chỉnh (Fine-tuning) nhiều kiến trúc mô hình khác nhau.
3. **Evaluation:** So sánh hiệu năng giữa các mô hình.
4. **Deployment:** Triển khai ứng dụng Web tương tác để demo khả năng dự đoán thực tế.

Toàn bộ pipeline (training lẫn inference) chạy trên **PyTorch** — không còn phụ thuộc TensorFlow/Keras.

## Dataset

Dữ liệu được sử dụng trong dự án là **Bean Leaf Lesions Classification Dataset**.

* **Nguồn:** [Kaggle - Bean Leaf Lesions Classification](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification)
* **Số lớp (Classes):** 3 lớp (Bao gồm 2 loại bệnh và lá khỏe mạnh).
* **Đặc điểm:** Ảnh chụp lá đậu với các điều kiện ánh sáng và góc chụp khác nhau.
* **Data segmentation:** https://universe.roboflow.com/alebachew-m/final_instance_segmentation

Xem [data/README.md](data/README.md) để biết cấu trúc thư mục dữ liệu cần có.

## Các mô hình được sử dụng (Model Architectures)

1. **CNN tự build (BeanLeafVGG):** Mạng CNN cổ điển với kiến trúc sâu, dùng làm baseline.
2. **EfficientNet-B3:** Mô hình tối ưu hóa sự cân bằng giữa độ chính xác và chi phí tính toán.
3. **MobileNetV3Large:** Mô hình nhẹ (lightweight), tối ưu cho các thiết bị di động/edge devices.
4. **DeiT (Data-efficient Image Transformers):** Ứng dụng kiến trúc Vision Transformer vào bài toán phân loại ảnh.
5. **YOLOv8-seg:** Sử dụng cho bài toán phát hiện + phân vùng bệnh (Detection/Segmentation) - *Thử nghiệm mở rộng*.

Cả 5 mô hình đều được định nghĩa và train bằng PyTorch (VGG/EfficientNet/MobileNetV3/DeiT trong `src/bean_leaf/models/`, YOLO qua Ultralytics).

## Cấu trúc dự án (Project Structure)

```bash
bean-leaf-disease/
├── data/                       # Dataset (không commit ảnh, xem data/README.md)
├── models/                     # Checkpoint đã train (không commit, xem models/README.md)
├── notebooks/                  # Jupyter Notebooks cho EDA và thử nghiệm model
│   ├── 01_eda.ipynb
│   ├── 02_mobilenet_v3.ipynb
│   ├── 03_deit.ipynb
│   ├── 04_cnn_from_scratch.ipynb
│   ├── 05_efficientnet_b3.ipynb
│   └── 06_yolo_segmentation.ipynb
├── src/bean_leaf/              # Package chính (cài qua `pip install -e .`)
│   ├── data/
│   │   ├── dataset.py          # create_df + DataLoader (PyTorch/torchvision)
│   │   └── eda.py              # Visualize phân bố lớp, augmentation
│   ├── models/
│   │   ├── vgg_custom.py       # BeanLeafVGG (from scratch)
│   │   ├── efficientnet.py     # EfficientNet-B3
│   │   ├── mobilenetv3.py      # MobileNetV3Large
│   │   ├── deit.py             # DeiT (timm)
│   │   └── yolo_seg.py         # YOLOv8-seg (Ultralytics API riêng, xem ghi chú trong file)
│   ├── training/
│   │   └── early_stopping.py   # EarlyStopping dùng chung cho 4 model classification
│   ├── evaluation/
│   │   └── metrics.py          # Classification report, confusion matrix, ROC/AUC, Grad-CAM
│   └── utils/
│       └── paths.py            # Quản lý output dir (env var thay vì hardcode)
├── scripts/
│   ├── train.py                # Train 4 model classification: python scripts/train.py --model ...
│   └── train_yolo.py           # Train YOLOv8-seg: python scripts/train_yolo.py --data_yaml ...
├── app/                        # Ứng dụng Web (Streamlit)
│   ├── streamlit_app.py
│   ├── config.py
│   └── utils.py                # Load model + predict cho cả 5 kiến trúc
├── tests/                      # Smoke test cho model/dataset (pytest)
├── docker/Dockerfile           # Đóng gói app Streamlit để deploy
├── reports/N8_report.pdf       # Báo cáo đồ án
├── requirements.txt
└── pyproject.toml
```

## Cài đặt (Installation)

1. **Clone dự án:**
```bash
git clone https://github.com/your-username/bean-leaf-disease.git
cd bean-leaf-disease
```

2. **Tạo môi trường ảo (Khuyến nghị):**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Cài đặt thư viện + package nội bộ (editable install):**
```bash
pip install -r requirements.txt
pip install -e .
```
`pip install -e .` giúp `import bean_leaf` hoạt động từ bất kỳ đâu (scripts/, tests/, notebooks/) mà không cần chỉnh `sys.path`.

## Huấn luyện mô hình (Training)

Sử dụng `scripts/train.py` để huấn luyện các mô hình PyTorch (VGG, EfficientNet, MobileNetV3, DeiT).

**Cú pháp:**

```bash
python scripts/train.py --data_dir "đường_dẫn_đến_dataset" --model [tên_model] [options]
```

**Tham số:**

* `--data_dir`: Đường dẫn đến folder chứa dữ liệu (cấu trúc `train/`, `val/`).
* `--model`: Chọn model để train. Các tùy chọn: `vgg`, `efficientnet`, `mobilenet`, `deit`, hoặc `all` (train tất cả).
* `--output_dir`: Nơi lưu checkpoint (mặc định `./outputs`, có thể override bằng biến môi trường `BEAN_LEAF_OUTPUT_DIR`).
* `--eda`: (Tùy chọn) Chạy phân tích dữ liệu trước khi train.

**Ví dụ:**

```bash
# Train toàn bộ các model
python scripts/train.py --data_dir "./data" --model all

# Chỉ train MobileNetV3
python scripts/train.py --data_dir "./data" --model mobilenet
```

*Lưu ý: Quá trình train tích hợp sẵn Early Stopping và Learning Rate Scheduler để tối ưu hóa kết quả.*

> **MobileNetV3 đã được viết lại bằng PyTorch** (trước đây dùng TensorFlow/Keras). Checkpoint
> `.keras` cũ không còn tương thích — cần chạy lại lệnh trên để có `best_mobilenetv3_model.pth`
> trước khi dùng được trong web app (xem [models/README.md](models/README.md)).

### Huấn luyện YOLOv8-seg (riêng)

YOLO không dùng chung pipeline trên vì Ultralytics tự quản lý train/augmentation qua 1 file
`data.yaml` (định dạng segmentation, export từ Roboflow - xem mục Dataset), không phải cấu trúc
`train/`, `val/` dạng ImageFolder:

```bash
python scripts/train_yolo.py --data_yaml "đường_dẫn_đến/data.yaml" --model_size n
```

Kết quả train nằm ở `runs/segment/train/weights/best.pt` (do Ultralytics tự tạo) — copy file này
vào `models/model_segemnt_yolo.pt` để web app dùng được (xem [models/README.md](models/README.md)).

## Sử dụng Web App (Inference)

Ứng dụng web được xây dựng bằng **Streamlit**, cho phép tải ảnh lên và nhận diện bệnh theo thời gian thực.

1. **Chạy ứng dụng:**
```bash
streamlit run app/streamlit_app.py
```

2. **Tính năng trên Web:**
* **Single View:** Chọn 1 model cụ thể để phân tích ảnh.
* **Compare Mode:** So sánh kết quả dự đoán (Confidence score) giữa tất cả các model (VGG, MobileNet, EfficientNet, DeiT) trên cùng 1 ảnh.
* **Visualization:** Hiển thị biểu đồ xác suất và thông tin chi tiết về bệnh (mức độ nghiêm trọng, khuyến nghị xử lý).

3. **Chạy bằng Docker (tùy chọn):**
```bash
docker build -t bean-leaf-app -f docker/Dockerfile .
docker run -p 8501:8501 -v $(pwd)/models:/app/models bean-leaf-app
```

## Kiểm thử (Tests)

```bash
pytest
```

Chạy smoke test cho từng kiến trúc model (forward pass đúng shape) và cho `create_df`. CI (`.github/workflows/ci.yml`) tự động chạy `pytest` trên mỗi push/PR vào `main`.
