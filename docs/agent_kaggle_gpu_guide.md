# 🧠 Playbook: Kinh Nghiệm & Quy Trình Phát Triển Hybrid (Local PC + Kaggle GPU Cloud)

Tài liệu này tổng hợp các **bài học kinh nghiệm và quy chuẩn kỹ thuật (Best Practices)** được rút ra từ thực tế dự án. Đây là cẩm nang hướng dẫn cho các **AI Coding Agent** khi xây dựng và phát triển các hệ thống Deep Learning / Computer Vision trong môi trường kết hợp **Local PC** và **Kaggle GPU Cloud**.

---

## 🏛️ 1. Nguyên Tắc Mô-Đun Hóa & Tách Bạch Môi Trường

* **Luôn phát triển dạng Script `.py` chuẩn mực tại Local:**  
  Tránh phụ thuộc hoàn toàn vào các file Notebook tạm thời (`.ipynb`). Mọi logic huấn luyện, tiền xử lý và đánh giá phải được đóng gói sạch sẽ vào các file Python mô-đun (`src/` và `scripts/`).
* **Tuyệt đối không hardcode đường dẫn môi trường:**  
  Không dùng đường dẫn tuyệt đối phụ thuộc hệ điều hành (như `C:\Users\...`). Tất cả đường dẫn dữ liệu và weights mô hình phải sử dụng đường dẫn tương đối (`./data`, `./models`, `./outputs`) hoặc được tính toán tự động dựa trên thư mục gốc dự án (`BASE_DIR = os.path.dirname(...)`).

---

## ⚡ 2. Quy Trình Phối Hợp Hybrid (Hybrid Workflow Paradigm)

### A. Vai Trò Của Môi Trường Local PC
* **Phát triển & Debug:** Đóng gói giao diện người dùng (Streamlit/FastAPI), viết unit tests (`pytest`), kiểm thử chức năng với dữ liệu mẫu nhỏ.
* **Đánh giá Offline (Offline Evaluation):** Thực thi inference đánh giá định lượng độc lập và tải giao diện Web cho người dùng trực tiếp trên máy local.

### B. Vai Trò Của Kaggle GPU Cloud
* **Worker Huấn luyện Tốc độ cao:** Sử dụng GPU P100/T4 miễn phí trên Kaggle để chạy các job huấn luyện nhiều epoch (PyTorch, YOLOv8, Transformers) mà không làm quá tải tài nguyên máy local.
* **Đồng bộ Artifacts:** Sau khi job huấn luyện trên Cloud hoàn tất (`COMPLETE`), tiến hành tải weights mô hình đã tối ưu (`best_*.pth`, `best_*.pt`) về thư mục `models/` tại Local.

---

## 📊 3. Phương Pháp Đánh Giá Khách Quan & Bộ Đệm Metric

* **Tách biệt tập Test độc lập:**  
  Không dùng tập `validation` (tập dùng để chọn checkpoint/early-stopping trong lúc train) để làm báo cáo kết quả cuối cùng. Luôn đánh giá trên tập `test` độc lập để đảm bảo số liệu trung thực và không bị thiên vị.
* **Bộ đệm kết quả dạng JSON (Metric Caching):**  
  Sau khi huấn luyện xong, chạy script đánh giá offline 1 lần duy nhất và xuất toàn bộ kết quả (Accuracy, Precision, Recall, F1-score, Confusion Matrix) ra file JSON (ví dụ: `outputs/evaluation_metrics.json`). Điều này giúp Web App và báo cáo đọc kết quả tức thì mà không cần phải chạy lại inference tốn thời gian.

---

## 🔒 4. Tiêu Chuẩn Sản Phẩm & Tính An Danh

* **Single Source of Truth (`config.py`):**  
  Tập trung toàn bộ cấu hình siêu tham số (img_size, batch_size, learning_rate...) và đường dẫn trong 1 file cấu hình duy nhất để dễ quản lý và bảo trì.
* **Chuẩn hóa thông tin công khai:**  
  Đảm bảo file cấu hình và giao diện không chứa thông tin nhạy cảm, API key hay thông tin cá nhân cứng, giúp mã nguồn dễ dàng chia sẻ và triển khai công khai.
