# 🍃 Bean Leaf Lesions Classification

Hệ thống **Deep Learning** toàn diện chẩn đoán và phân loại tổn thương bệnh hại trên lá đậu (Bean Leaves). Dự án tích hợp các kiến trúc CNN gồm **BeanLeafLite** tự thiết kế (0.91M params), **ShuffleNetV2** (1.26M), **MobileNetV3-Large** (2.97M), **EfficientNet-B0** (4.01M), và **ResNet50** (23.51M), đi kèm ứng dụng Web Streamlit.

---

## 📌 Tính năng Hệ thống

- **Hỗ trợ Đa kiến trúc CNN (Multi-Architecture Ecosystem):**
  - **BeanLeafLite (Custom CNN):** Kiến trúc siêu nhẹ tự thiết kế (0.91M params, 0.36 GFLOPs).
  - **ShuffleNetV2 (x1.0):** Tối ưu hóa xáo trộn kênh đặc trưng cho di động (1.26M params).
  - **MobileNetV3-Large:** Tối ưu hóa suy luận thời gian thực cho thiết bị Edge (2.97M params).
  - **EfficientNet-B0:** Cân bằng tối ưu giữa tham số và khả năng tổng quát hóa (4.01M params).
  - **ResNet50:** Kiến trúc mạng cuộn sâu tiêu chuẩn với Skip Connections (23.51M params).

  > Số tham số và FLOPs trên đo bằng `torch.utils.flop_counter` trên mô hình đã thay lớp phân
  > loại về **3 lớp**, ở đúng độ phân giải 384x384 dùng khi benchmark - khác với con số thường
  > trích trong paper gốc (tính cho 1000 lớp ImageNet ở 224x224). FLOPs tính 1 phép nhân-cộng
  > là 2 FLOPs; paper nào ghi "FLOPs" theo nghĩa MACs sẽ ra đúng một nửa con số này.

- **Ứng dụng Web Tương tác (Streamlit Web App):**
  - **Single View:** Phân tích ảnh đơn, hiển thị biểu đồ xác suất & khuyến nghị y tế nông nghiệp.
  - **Compare Mode:** So sánh dự đoán song song của tất cả các mô hình trên cùng một bức ảnh.
  - **Grad-CAM Visualization:** Bản đồ nhiệt giải thích vùng chú ý của AI khi chẩn đoán.

---

## 🛠️ Kiến trúc Tự Thiết Kế: BeanLeafLite (0.91M Params, 0.36 GFLOPs)

**BeanLeafLite** là mô hình mạng nơ-ron cuộn (CNN) được tự thiết kế nhằm tối ưu hóa sự cân bằng giữa độ chính xác và dung lượng tính toán trên các thiết bị di động hoặc môi trường nhúng:

- **Depthwise-Separable Convolutions:** Tách biệt quá trình lọc không gian (spatial) và phối hợp kênh (channel), giúp giảm số lượng tham số xuống chỉ còn **0.91M** (nhẹ hơn EfficientNet-B0 4.4 lần, ResNet50 26 lần).
- **Residual Skip Connections:** Kết nối tắt giữa các tầng block giúp dòng gradient truyền trực tiếp, tránh hiện tượng suy giảm gradient khi huấn luyện sâu.
- **Squeeze-and-Excitation (SE) Attention:** Cơ chế chú ý kênh giúp tự động tái trọng số các đặc trưng quan trọng, tập trung vào các chi tiết tổn thương đốm lá nhỏ.
- **Hiệu năng Thực nghiệm:** Đạt **94.00% OOF Accuracy** (CI 95%: 92.39-95.29) với chi phí
  **0.36 GFLOPs** - thấp hơn ResNet50 **67 lần** về FLOPs trong khi chỉ kém 4 điểm chính xác.
  Giá trị của kiến trúc này nằm ở tỉ lệ chính xác/chi phí, không phải ở độ chính xác tuyệt đối.
- **Lưu ý về huấn luyện:** Đây là mô hình duy nhất train **from-scratch** (không có trọng số
  pretrained). Nó cần learning rate 3e-3, cao hơn nhóm pretrained lớn (1e-3) - nhưng ngang bằng
  ShuffleNetV2, vốn cũng chọn 3e-3.

  Về độ ổn định, cần phân biệt hai con số: trên **tập test 133 ảnh** nó dao động ± 2.41%, lớn
  nhất trong 5 mô hình; nhưng trên **cross-validation 1034 ảnh** thì ± 1.23%, chỉ đứng **thứ 3**
  - ShuffleNetV2 (± 1.78%) và ResNet50 (± 1.49%) còn dao động hơn. Nói "BeanLeafLite kém ổn định
  nhất" là không chính xác; đúng hơn là kết quả của nó trên tập test nhỏ khó lặp lại hơn.

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

**Chỉ số chính là Cross-Validation Accuracy**: trung bình ± độ lệch chuẩn qua **5 fold**, đo
trên phần dữ liệu mà mỗi fold không nhìn thấy khi huấn luyện (tổng cộng phủ đủ 1034 ảnh train).
Không dùng Test Accuracy làm chỉ số chính vì tập test chuẩn chỉ có 133 ảnh và đã chạm trần:
ResNet50 đạt Test Acc cao nhất (99.85%) nhưng chỉ xếp thứ 3 theo Cross-Validation.

| Mô hình | LR | **CV Accuracy (5 fold)** | CI 95% | Test Accuracy | Params | GFLOPs | Đặc trưng Kiến trúc |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **EfficientNet-B0** | 1e-3 | **98.84 ± 0.73%** | 97.98-99.33 | 99.40 ± 0.63% | 4.01M | 2.26 | Compound Scaling cân bằng hiệu năng & tài nguyên |
| **MobileNetV3-Large** | 1e-3 | **98.74 ± 1.31%** | 97.86-99.26 | 99.70 ± 0.41% | 2.97M | 1.25 | Tối ưu suy luận thực địa cho thiết bị Edge |
| **ResNet50** | 1e-3 | **98.07 ± 1.49%** | 97.03-98.74 | 99.85 ± 0.34% | 23.51M | 24.02 | Mạng cuộn sâu Residual Skip Connections tiêu chuẩn |
| **ShuffleNetV2** (x1.0) | 3e-3 | **97.39 ± 2.38%** | 96.23-98.20 | 98.80 ± 0.67% | 1.26M | 0.85 | Channel Shuffle & Inverted Residual cho di động |
| **BeanLeafLite** (Custom CNN) | 3e-3 | **94.00 ± 1.23%** | 92.39-95.29 | 93.53 ± 2.41% | 0.91M | 0.36 | Depthwise-Separable + Residual + SE Attention |

Accuracy từng fold (dùng để tính cột CV Accuracy):

| Mô hình | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|:---:|:---:|:---:|:---:|:---:|
| EfficientNet-B0 | 99.03 | 99.52 | 97.58 | 99.03 | 99.03 |
| MobileNetV3-Large | 99.52 | 100.00 | 97.10 | 99.52 | 97.57 |
| ResNet50 | 99.03 | 99.03 | 95.65 | 97.58 | 99.03 |
| ShuffleNetV2 | 98.55 | 99.03 | 93.24 | 98.55 | 97.57 |
| BeanLeafLite | 94.69 | 95.17 | 93.24 | 94.69 | 92.23 |

### Giao thức Đánh giá (Benchmark Protocol)

5-fold StratifiedKFold trên tập train (1034 ảnh), seed cố định 42. **Mọi mô hình dùng chung
một quy trình huấn luyện**, không mô hình nào có ngoại lệ:

- AdamW (weight decay 1e-2), CosineAnnealingLR, **40 epoch cố định - không dừng sớm**
- Ảnh 384x384 (cùng phép nội suy Bilinear), batch 32, label smoothing 0.05, augmentation giống nhau
- Learning rate của từng mô hình chọn từ **cùng một lưới** {1e-4, 3e-4, 1e-3, 3e-3} bằng sweep
  15 epoch trên tập internal-validation. Mỗi mô hình được thử **đúng 4 lần như nhau**.
- Tập test (`val_dir`) không tham gia chọn siêu tham số, không tham gia chọn checkpoint, chỉ
  được đánh giá đúng 1 lần sau khi huấn luyện xong.
- **CV Accuracy** = trung bình ± độ lệch chuẩn accuracy của 5 fold. Vì các fold gần bằng nhau
  về kích thước, con số này trùng khít với accuracy tính bằng cách gộp toàn bộ dự đoán
  out-of-fold lại (1034 ảnh) - cột CI 95% được tính trên tập gộp đó, và hẹp hơn **khoảng 3 lần**
  (thực đo: 2.87-3.63 lần tuỳ mô hình) so với khoảng tin cậy khi chỉ đo trên 133 ảnh test.

Vì sao cần sweep LR thay vì ép chung một giá trị: `lr=3e-4` là learning rate kinh điển để
*fine-tune* mô hình pretrained, quá nhỏ với mô hình *train from-scratch*. Thực nghiệm xác nhận
điều này - **cả 5 mô hình đều chọn LR cao hơn 3e-4**. Ép chung một LR nghe có vẻ công bằng
nhưng thực chất thiên vị nhóm pretrained; công bằng đúng nghĩa là mọi mô hình được tune với
**cùng ngân sách tìm kiếm**, rồi so kết quả tốt nhất của từng mô hình.

### Nhận định & Giới hạn

- **Kiểm định McNemar** (so từng cặp trên cùng 1034 ảnh, hiệu chỉnh đa so sánh bằng
  Holm-Bonferroni với 10 phép so sánh). So `mean ± std` cạnh nhau là chưa đủ vì độ lệch chuẩn
  giữa các fold lớn hơn khoảng cách giữa các mô hình; McNemar chỉ đếm những ảnh mà **hai mô
  hình bất đồng**, nên nhạy hơn hẳn:

  | Cặp so sánh | Chỉ A đúng | Chỉ B đúng | p (thô) | Kết luận |
  |---|:---:|:---:|:---:|---|
  | BeanLeafLite vs EfficientNet-B0 | 8 | 58 | < 0.0001 | **Khác biệt** |
  | BeanLeafLite vs MobileNetV3 | 11 | 60 | < 0.0001 | **Khác biệt** |
  | BeanLeafLite vs ResNet50 | 14 | 56 | < 0.0001 | **Khác biệt** |
  | BeanLeafLite vs ShuffleNetV2 | 17 | 52 | 0.00003 | **Khác biệt** |
  | EfficientNet-B0 vs ShuffleNetV2 | 19 | 4 | 0.0026 | **Khác biệt** |
  | MobileNetV3 vs ShuffleNetV2 | 20 | 6 | 0.0094 | **Khác biệt** |
  | EfficientNet-B0 vs ResNet50 | 12 | 4 | 0.0768 | Không tách được |
  | ResNet50 vs ShuffleNetV2 | 12 | 5 | 0.1435 | Không tách được |
  | MobileNetV3 vs ResNet50 | 13 | 6 | 0.1671 | Không tách được |
  | EfficientNet-B0 vs MobileNetV3 | 8 | 7 | 1.0000 | Không tách được |

  Sáu kết luận đứng vững sau hiệu chỉnh, gộp lại thành thứ tự sau:

  1. **BeanLeafLite kém hơn cả 4 mô hình pretrained** - thua 52-60 ảnh, chỉ thắng 8-17.
  2. **EfficientNet-B0 và MobileNetV3 đều tốt hơn ShuffleNetV2.**
  3. **EfficientNet-B0, MobileNetV3 và ResNet50 không tách được khỏi nhau** - mọi cặp trong
     nhóm này đều có p > 0.07. Riêng EfficientNet-B0 vs MobileNetV3 bất đồng đúng 15 ảnh
     (8 vs 7, p = 1.0000): tương đương nhau ở mức không thể tương đương hơn.
  4. ResNet50 và ShuffleNetV2 cũng không tách được (p = 0.1435), dù ShuffleNetV2 thua rõ
     hai mô hình đầu bảng.

  Lưu ý mức tin cậy: cặp **MobileNetV3 vs ShuffleNetV2 có p = 0.0094 so với ngưỡng Holm 0.0100**
  - qua sát nút, không nên coi là kết luận chắc chắn.

  Kết luận: **không có mô hình "tốt nhất"**. ResNet50 dẫn đầu Test Accuracy (99.85%) nhưng
  không vượt trội ai theo kiểm định có cặp, trong khi tốn 24.02 GFLOPs so với 1.25 của
  MobileNetV3 - gấp **19 lần** cho cùng một mức chính xác không phân biệt được.
- **BeanLeafLite đạt 94.00% với 0.36 GFLOPs** - kém mô hình dẫn đầu 4.8 điểm nhưng rẻ hơn
  ResNet50 **67 lần** về FLOPs và **26 lần** về tham số. Đây mới là luận điểm của kiến trúc này,
  không phải độ chính xác. Lưu ý McNemar cho thấy khoảng cách này là **thật** (thua 52-60 ảnh
  trên 1034, p < 0.0001), khác hẳn với các khoảng cách trong nhóm pretrained vốn không phân
  biệt được.
- **Giới hạn (đã ghi nhận, không khảo sát thêm):** lưới LR dừng ở 3e-3, và **2/5 mô hình chọn
  đúng giá trị mép trên đó**. Kết quả sweep đầy đủ (internal-val accuracy sau 15 epoch):

  | Mô hình | 1e-4 | 3e-4 | 1e-3 | 3e-3 | Chọn |
  |---|:---:|:---:|:---:|:---:|:---:|
  | EfficientNet-B0 | 97.10 | 99.03 | **99.52** | 99.03 | 1e-3 |
  | MobileNetV3-Large | 97.58 | 98.07 | **99.52** | 98.55 | 1e-3 |
  | ResNet50 | 97.58 | 98.07 | **98.55** | 97.58 | 1e-3 |
  | ShuffleNetV2 | 97.10 | 99.03 | 99.03 | **99.52** | 3e-3 |
  | BeanLeafLite | 74.88 | 78.26 | 87.44 | **90.82** | 3e-3 |

  Ba mô hình đầu có cực đại nằm gọn trong lưới (giảm ở 3e-3) nên LR đã chọn là đáng tin.
  ShuffleNetV2 và BeanLeafLite thì chưa: BeanLeafLite còn **tăng đơn điệu** qua cả 4 mốc
  (74.88 → 78.26 → 87.44 → 90.82), tức điểm tối ưu của nó nhiều khả năng nằm **ngoài** lưới
  và con số 94.00% trong bảng là **cận dưới**, không phải hiệu năng tốt nhất mà kiến trúc này
  đạt được. Muốn biết chính xác thì phải nới lưới thêm (1e-2, 3e-2) cho **cả 5 mô hình** để
  giữ nguyên điều kiện ngân sách tìm kiếm bằng nhau.

  Một nhận xét đáng lưu ý từ bảng trên: **không mô hình nào chọn 3e-4** - giá trị vốn được
  dùng làm learning rate chung cho tất cả trước khi có sweep.
- **Giới hạn:** mọi mô hình đều chạy ở 384px, trong khi **cả 4 mô hình pretrained**
  (MobileNetV3, ShuffleNetV2, EfficientNet-B0, ResNet50) đều được thiết kế cho **224px** và
  trọng số ImageNet của chúng cũng học ở 224px. Ngược lại BeanLeafLite được thiết kế cho đúng
  384px (xem docstring của nó). Ảnh gốc là 500x500 nên 384px là thu nhỏ chứ không phóng to, và
  mọi mô hình đều chịu cùng điều kiện - nhưng đây là mốc "có kiểm soát" chứ không trung lập,
  vì nó trùng với điểm thiết kế của đúng một mô hình.

- **Đã sửa trong phiên benchmark này (ghi lại để tiện đối chiếu với kết quả cũ):** MobileNetV3
  từng có head riêng `Linear(960->256) + BatchNorm1d + SiLU + Dropout + Linear(256->3)` trong khi
  3 mô hình pretrained kia chỉ `Dropout + Linear`, và ShuffleNetV2 từng dùng `Dropout(0.2)` thay
  vì `0.3`. Sau khi đồng nhất, MobileNetV3 giảm từ 98.94% xuống 98.74% (params 3.22M -> 2.97M)
  và ShuffleNetV2 từ 97.58% xuống 97.39% - cả hai đều **nằm trong độ lệch chuẩn**, nên hai khác
  biệt cấu hình đó không phải nguyên nhân tạo ra thứ hạng. Tuy vậy nó đủ để **đảo ngôi đầu bảng**
  giữa MobileNetV3 và EfficientNet-B0 - minh hoạ trực tiếp cho việc thứ hạng trong nhóm dẫn đầu
  không mang thông tin.

Số liệu chi tiết từng fold, kết quả sweep và confusion matrix: [`outputs/kfold/`](outputs/kfold/).


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
