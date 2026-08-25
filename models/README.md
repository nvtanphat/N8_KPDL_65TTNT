# Model Checkpoints (`models/`)

Thư mục này chứa các file trọng số mô hình đã qua huấn luyện (`.pth`, `.pt`), dùng để phục vụ suy luận trực tiếp trên ứng dụng Web Streamlit (`app/streamlit_app.py`).

---

## 📋 Danh sách File Checkpoints Standardized

| Mô hình AI | Kiến trúc | Tên file Checkpoint trong `models/` |
|---|---|---|
| **BeanLeafLite** | Custom CNN (Depthwise + Residual + SE) | `best_beanleaflite.pth` |
| **ShuffleNetV2** | PyTorch ShuffleNetV2 (x1.0) | `best_shufflenetv2.pth` |
| **MobileNetV3-Large** | PyTorch MobileNetV3 | `best_mobilenetv3.pth` |
| **EfficientNet-B0** | PyTorch EfficientNet-B0 | `best_efficientnet_b0.pth` |
| **ResNet50** | PyTorch ResNet50 | `best_resnet50.pth` |

> 📌 **Lưu ý:** Tên các file trọng số trên phải khớp 100% với định nghĩa cấu hình tại [`app/config.py`](../app/config.py) (`MODELS[...]['file']`).
