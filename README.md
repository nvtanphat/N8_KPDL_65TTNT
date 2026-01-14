# Bean Leaf Lesions Classification (Phân loại Bệnh trên Lá Đậu)

> **Đồ án môn học: Khai phá dữ liệu (Data Mining)**

## Giới thiệu (Overview)

Dự án này tập trung vào việc áp dụng các kỹ thuật **Deep Learning** tiên tiến (CNNs và Vision Transformers) để tự động phân loại các tổn thương bệnh trên lá đậu. Hệ thống giúp hỗ trợ nông dân và các chuyên gia nông nghiệp phát hiện sớm bệnh hại, từ đó đưa ra biện pháp xử lý kịp thời.

Dự án bao gồm quy trình trọn vẹn từ:

1. **EDA & Preprocessing:** Khám phá và xử lý dữ liệu ảnh.
2. **Model Training:** Huấn luyện và tinh chỉnh (Fine-tuning) nhiều kiến trúc mô hình khác nhau.
3. **Evaluation:** So sánh hiệu năng giữa các mô hình.
4. **Deployment:** Triển khai ứng dụng Web tương tác để demo khả năng dự đoán thực tế.

## Dataset

Dữ liệu được sử dụng trong dự án là **Bean Leaf Lesions Classification Dataset**.

* **Nguồn:** [Kaggle - Bean Leaf Lesions Classification](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification)
* **Số lớp (Classes):** 3 lớp (Bao gồm 2 loại bệnh và lá khỏe mạnh).
* **Đặc điểm:** Ảnh chụp lá đậu với các điều kiện ánh sáng và góc chụp khác nhau.

##  Các mô hình được sử dụng (Model Architectures)

1. **CNN tự build (BeanLeafVGG):** Mạng CNN cổ điển với kiến trúc sâu, dùng làm baseline.
2. **EfficientNet-B3:** Mô hình tối ưu hóa sự cân bằng giữa độ chính xác và chi phí tính toán (Scale width, depth, resolution).
3. **MobileNetV3:** Mô hình nhẹ (lightweight), tối ưu cho các thiết bị di động/edge devices.
4. **DeiT (Data-efficient Image Transformers):** Ứng dụng kiến trúc Transformer (Vision Transformer) vào bài toán phân loại ảnh.
5. **YOLO:** Sử dụng cho bài toán phát hiện vùng bệnh (Detection) - *Thử nghiệm mở rộng*.

## Cấu trúc dự án (Project Structure)

```bash
N8_KPDL_65TTNT/
├── notebook/                  # Các Jupyter Notebooks cho EDA và thử nghiệm model
│   ├── 01_eda.ipynb           # Phân tích khám phá dữ liệu
│   ├── 02_mobilenetv3_tanphat.ipynb
│   ├── 03_deit_tanphat.ipynb
│   ├── 04_cnnfromscratch_hoangloc.ipynb
│   ├── 05_efficientnetb3_hoangloc.ipynb
│   └── 06_detectionyolo_tanphat.ipynb
├── src/                       # Mã nguồn huấn luyện chính (PyTorch)
│   ├── main.py                # Script điều phối quá trình train
│   ├── preprocessing.py       # Xử lý dữ liệu và augmentation
│   ├── eda.py                 # Hàm hỗ trợ visualize dữ liệu
│   ├── evaluation.py          # Đánh giá model (Metrics, Confusion Matrix)
│   ├── model_nguyenhoangloc1.py   # Định nghĩa VGG
│   ├── model_nguyenhoangloc2.py   # Định nghĩa EfficientNet
│   ├── model_nguyenvantanphat1.py # Các model bổ sung
│   └── model_nguyenvantanphat2.py # Định nghĩa DeiT
├── web/                       # Ứng dụng Web (Streamlit)
│   ├── app.py                 # Giao diện chính
│   ├── config.py              # Cấu hình web
│   └── utils.py               # Hàm load model và dự đoán
├── requirements.txt           # Các thư viện cần thiết
└── README.md                  # Tài liệu hướng dẫn

```

##  Cài đặt (Installation)

1. **Clone dự án:**
```bash
git clone https://github.com/your-username/N8_KPDL_65TTNT.git
cd N8_KPDL_65TTNT

```


2. **Tạo môi trường ảo (Khuyến nghị):**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

```


3. **Cài đặt thư viện:**
```bash
pip install -r requirements.txt

```



##  Huấn luyện mô hình (Training)

Sử dụng script `src/main.py` để huấn luyện các mô hình PyTorch (VGG, EfficientNet, DeiT).

**Cú pháp:**

```bash
python src/main.py --data_dir "đường_dẫn_đến_dataset" --model [tên_model] [options]

```

**Tham số:**

* `--data_dir`: Đường dẫn đến folder chứa dữ liệu (cấu trúc `train/`, `val/`).
* `--model`: Chọn model để train. Các tùy chọn: `vgg`, `efficientnet`, `deit`, hoặc `all` (train tất cả).
* `--eda`: (Tùy chọn) Chạy phân tích dữ liệu trước khi train.

**Ví dụ:**

```bash
# Train toàn bộ các model
python src/main.py --data_dir "./dataset" --model all

# Chỉ train EfficientNet
python src/main.py --data_dir "./dataset" --model efficientnet

```

*Lưu ý: Quá trình train tích hợp sẵn Early Stopping và Learning Rate Scheduler để tối ưu hóa kết quả.*

##  Sử dụng Web App (Inference)

Ứng dụng web được xây dựng bằng **Streamlit**, cho phép tải ảnh lên và nhận diện bệnh theo thời gian thực.

1. **Chạy ứng dụng:**
```bash
streamlit run web/app.py

```


2. **Tính năng trên Web:**
* **Single View:** Chọn 1 model cụ thể để phân tích ảnh.
* **Compare Mode:** So sánh kết quả dự đoán (Confidence score) giữa tất cả các model (VGG, MobileNet, EfficientNet, DeiT) trên cùng 1 ảnh.
* **Visualization:** Hiển thị biểu đồ xác suất và thông tin chi tiết về bệnh (mức độ nghiêm trọng, khuyến nghị xử lý).

