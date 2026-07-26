# Model Checkpoints (`models/`)

Thư mục này chứa các file trọng số mô hình đã qua huấn luyện (`.pth`, `.pt`), dùng để phục vụ suy luận trực tiếp trên ứng dụng Web Streamlit (`app/streamlit_app.py`).

---

## 📋 Danh sách File Checkpoints Standardized

| Mô hình AI | Kiến trúc | Tên file Checkpoint trong `models/` |
|---|---|---|
| **MobileNetV3-Large** | PyTorch MobileNetV3 | `best_mobilenetv3.pth` |
| **BeanLeafLite** | Custom CNN (Depthwise + Residual + SE) | `best_beanleaflite.pth` |
| **DeiT-Small** | Vision Transformer (PyTorch / Timm) | `best_deit.pth` |
| **EfficientNet-B3** | PyTorch EfficientNet | `best_efficientnet.pth` |
| **YOLOv8-seg** | Ultralytics Instance Segmentation | `best_yolov8_segmentation.pt` |

> 📌 **Lưu ý:** Tên các file trọng số trên phải khớp 100% với định nghĩa cấu hình tại [`app/config.py`](../app/config.py) (`MODELS[...]['file']`).
