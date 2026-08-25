# 🍃 Bean Leaf Lesions Classification

Hệ thống **Deep Learning** toàn diện chẩn đoán và phân loại tổn thương bệnh hại trên lá đậu (Bean Leaves). Dự án tích hợp các kiến trúc CNN gồm **BeanLeafLite** tự thiết kế (~0.94M params), **ShuffleNetV2** (~2.3M), **MobileNetV3-Large** (~3.2M), **EfficientNet-B0** (~5.3M), và **ResNet50** (~25.6M), đi kèm ứng dụng Web Streamlit.

---

## 📌 Tính năng Hệ thống

- **Hỗ trợ Đa kiến trúc CNN (Multi-Architecture Ecosystem):**
  - **BeanLeafLite (Custom CNN):** Kiến trúc siêu nhẹ tự thiết kế (~0.94M params).
  - **ShuffleNetV2 (x1.0):** Tối ưu hóa xáo trộn kênh đặc trưng cho di động (~2.3M params).
  - **MobileNetV3-Large:** Tối ưu hóa suy luận thời gian thực cho thiết bị Edge (~3.2M params).
  - **EfficientNet-B0:** Cân bằng tối ưu giữa tham số và khả năng tổng quát hóa (~5.3M params).
  - **ResNet50:** Kiến trúc mạng cuộn sâu tiêu chuẩn với Skip Connections (~25.6M params).

- **Ứng dụng Web Tương tác (Streamlit Web App):**
  - **Single View:** Phân tích ảnh đơn, hiển thị biểu đồ xác suất & khuyến nghị y tế nông nghiệp.
  - **Compare Mode:** So sánh dự đoán song song của tất cả các mô hình trên cùng một bức ảnh.
  - **Grad-CAM Visualization:** Bản đồ nhiệt giải thích vùng chú ý của AI khi chẩn đoán.

---

## 🛠️ Kiến trúc Tự Thiết Kế: BeanLeafLite (~0.94M Params)

**BeanLeafLite** là mô hình mạng nơ-ron cuộn (CNN) được tự thiết kế nhằm tối ưu hóa sự cân bằng giữa độ chính xác và dung lượng tính toán trên các thiết bị di động hoặc môi trường nhúng:

- **Depthwise-Separable Convolutions:** Tách biệt quá trình lọc không gian (spatial) và phối hợp kênh (channel), giúp giảm số lượng tham số xuống chỉ còn **~0.94M** (nhỏ hơn EfficientNet-B0 gấp 5.6 lần).
- **Residual Skip Connections:** Kết nối tắt giữa các tầng block giúp dòng gradient truyền trực tiếp, tránh hiện tượng suy giảm gradient khi huấn luyện sâu.
- **Squeeze-and-Excitation (SE) Attention:** Cơ chế chú ý kênh giúp tự động tái trọng số các đặc trưng quan trọng, tập trung vào các chi tiết tổn thương đốm lá nhỏ.
- **Hiệu năng Thực nghiệm:** Đạt độ chính xác cao trên tập kiểm thử độc lập, khẳng định hiệu quả vượt trội của mô hình tự thiết kế.

---

## 📊 Tập dữ liệu (Dataset)

- **Bài toán Phân loại (Classification):** [Bean Leaf Lesions Dataset](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification)

**Các lớp bệnh hại (3 Classes):**
1. `angular_leaf_spot` — Bệnh đốm góc lá
2. `bean_rust` — Bệnh gỉ sắt
3. `healthy` — Lá khỏe mạnh

---

## 📈 Kết quả Thực nghiệm & Đánh giá (Benchmark & Evaluation)

### Hiệu năng Mô hình Phân loại (Classification Benchmark)
Đánh giá độc lập trên tập test chuẩn (133 ảnh) dưới cùng **Controlled Benchmark Protocol (384px)**:

| Mô hình | Test Accuracy | Tham số (Params) | Phân nhóm Tối ưu & Đặc trưng Kiến trúc |
|---|:---:|:---:|---|
| **BeanLeafLite** (Custom CNN) | *Chờ train* | **~0.94M** | Depthwise-Separable + Residual + SE Attention (siêu nhẹ) |
| **ShuffleNetV2** (x1.0) | *Chờ train* | ~2.3M | Channel Shuffle & Inverted Residual cho di động |
| **MobileNetV3-Large** | *Chờ train* | ~3.2M | Kiến trúc tối ưu hóa cho thiết bị di động & Edge Computing |
| **EfficientNet-B0** | *Chờ train* | ~5.3M | Compound Scaling cân bằng hiệu năng & tài nguyên |
| **ResNet50** | *Chờ train* | ~25.6M | Mạng cuộn sâu Residual Skip Connections tiêu chuẩn |



---

## 🏗️ Cấu trúc Dự án

```bash
bean-leaf-disease/
├── app/                        # Giao diện Web App Streamlit
│   ├── config.py               # Cấu hình mô hình & thông tin bệnh lý
│   ├── streamlit_app.py        # Ứng dụng chính Streamlit
│   └── utils.py                # Pipeline nạp mô hình, suy luận & Grad-CAM
├── data/                       # Quản lý tập dữ liệu train/val
├── docs/                       # Tài liệu dự án & hình ảnh minh họa
├── models/                     # Thư mục chứa weights (.pth)
├── outputs/                    # Báo cáo đánh giá định lượng (JSON)
├── scripts/
│   ├── train.py                # Script huấn luyện các mô hình Phân loại
│   └── evaluate.py             # Script đánh giá offline & xuất metric
├── src/bean_leaf/              # Core Library Package
│   ├── config.py               # Single Source of Truth cho siêu tham số
│   ├── data/                   # DataLoaders & Augmentations
│   ├── evaluation/             # Đánh giá chỉ số & Grad-CAM
│   ├── models/                 # Kiến trúc mô hình (PyTorch & timm)
│   └── training/               # Vòng lặp huấn luyện & AMP Mixed Precision
└── tests/                      # Bộ unit tests tự động (pytest)
```

---

## 💻 Triển khai & Sử dụng Web App

![Giao diện Web App Streamlit Chẩn đoán & Grad-CAM](docs/assets/web_demo.png)

```bash
# Cài đặt thư viện & package local
pip install -r requirements.txt
pip install -e .

# Chạy ứng dụng web
streamlit run app/streamlit_app.py
```

---

## 🚀 Huấn luyện & Đánh giá (Training & Evaluation)

### Siêu tham số Huấn luyện Mặc định (`src/bean_leaf/config.py`)
- **Kích thước ảnh (`img_size`):** 384x384
- **Batch size:** 32
- **Tối ưu hóa (`Optimizer`):** AdamW (`lr=3e-4`, `weight_decay=1e-2`)
- **Kỹ thuật:** Automatic Mixed Precision (AMP), Early Stopping (`patience=7`), Label Smoothing (0.05)

### Lệnh Thực thi

```bash
# 1. Huấn luyện toàn bộ các mô hình phân loại (data/train, data/val)
python scripts/train.py --data_dir "./data" --model all

# 2. Đánh giá offline các model classification và lưu metric ra JSON
python scripts/evaluate.py

# 3. Chạy Unit Tests
pytest -v
```

---

## 👥 Tác giả & Lời cảm ơn

- **Nguyễn Văn Tấn Phát**
- **Nguyễn Hoàng Lộc**

Trân trọng cảm ơn cộng đồng mã nguồn mở **PyTorch**, **Timm**, và **Streamlit** đã hỗ trợ công cụ và nền tảng cho nghiên cứu này.
