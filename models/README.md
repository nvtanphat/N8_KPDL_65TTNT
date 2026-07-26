# Models

Thư mục này chứa checkpoint **đã sẵn sàng để web app đọc** (`.pth`, `.pt`), **không commit vào git**
(đã khai báo trong `.gitignore`) vì các file này quá nặng cho một git repo thông thường — dùng Git LFS /
cloud storage nếu cần chia sẻ.

`scripts/train.py` lưu checkpoint vào `outputs/<model>/best_<model>_model.pth` (thư mục làm việc khi train,
xem `--output_dir`), **khác** với thư mục `models/` mà `app/streamlit_app.py` đọc để inference. Sau khi
train xong, copy/đổi tên file theo bảng dưới đây:

| Model | Output của `scripts/train.py` | Copy vào `models/` với tên |
|---|---|---|
| BeanLeafLite (custom CNN) | `outputs/vgg/best_vgg_model.pth` | `model_cratch_hoangloc.pth` |
| EfficientNet-B3 | `outputs/efficientnet/best_efficientnet_model.pth` | `best_efficientnet_model.pth` (giữ nguyên tên) |
| MobileNetV3 | `outputs/mobilenet/best_mobilenet_model.pth` | `best_mobilenetv3.pth` |
| DeiT | `outputs/deit/best_deit_model.pth` | `model_deit_tanphat.pth` |
| YOLOv8-seg | `runs/segment/train/weights/best.pt` (train bằng `scripts/train_yolo.py --data_yaml ...`, hoặc `notebooks/06_yolo_segmentation.ipynb`) | `model_segemnt_yolo.pt` |

Tên file đích khớp với `app/config.py::MODELS[...]['file']` - đổi 1 trong 2 chỗ thì phải đổi luôn chỗ kia.

> **Lưu ý:** MobileNetV3 đã được viết lại bằng PyTorch (`src/bean_leaf/models/mobilenetv3.py`),
> thay cho bản TensorFlow/Keras cũ. Checkpoint `.keras` cũ **không tương thích** — cần train lại
> bằng `scripts/train.py --model mobilenet` để có `best_mobilenetv3.pth`. Cho đến khi đó, mục
> MobileNetV3 trên web app sẽ báo "không tìm thấy model".
>
> **Lưu ý:** kiến trúc custom CNN đã đổi từ BeanLeafVGG (conv3x3 thường) sang BeanLeafLite
> (Depthwise-Separable + Residual + SE, ~1M tham số). Checkpoint `model_cratch_hoangloc.pth`
> cũ (nếu có, train trên kiến trúc BeanLeafVGG) **không tương thích** — cần train lại bằng
> `scripts/train.py --model vgg` để có checkpoint đúng kiến trúc mới.
