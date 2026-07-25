# Kaggle Job - YOLOv8 Segmentation

Chạy `scripts/train_yolo.py` trên GPU miễn phí của Kaggle Kernels. Tách riêng khỏi
`kaggle_job/` (job classification) vì dataset khác nguồn: YOLO tải trực tiếp từ Roboflow
(`final_instance_segmentation`), không phải Kaggle dataset qua `dataset_sources`.

## Tinh chỉnh so với baseline notebook

`notebooks/06_yolo_segmentation.ipynb` dùng `MODEL_SIZE="n"` (nano - nhẹ nhất, mAP thấp
nhất). Job này mặc định đổi sang `"s"` (small) để tăng mAP đáng kể với chi phí thêm không
lớn trên T4. Override bằng biến môi trường của kernel (Settings → Environment Variables trên
Kaggle) nếu muốn thử size khác:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `YOLO_MODEL_SIZE` | `s` | `n`/`s`/`m`/`l`/`x` |
| `YOLO_EPOCHS` | `100` | Số epoch tối đa |
| `YOLO_PATIENCE` | `20` | Early stopping patience |
| `ROBOFLOW_API_KEY` | key public đi kèm dataset | Đổi nếu key cũ hết hạn |

## Chạy

> ⚠️ Trên Windows nhớ set `PYTHONUTF8=1` trước khi gọi `kaggle` (xem giải thích trong
> [../kaggle_job/README.md](../kaggle_job/README.md)).

```bash
# 1. Đẩy job lên Kaggle
kaggle kernels push -p ./kaggle_job_yolo

# 2. Theo dõi trạng thái
kaggle kernels status nguynvntnpht/bean-leaf-yolo-training

# 3. Khi Complete: tải checkpoint (best.pt) + log về máy local
kaggle kernels output nguynvntnpht/bean-leaf-yolo-training -p ./outputs_yolo
```

Checkpoint tốt nhất nằm ở `outputs_yolo/bean-leaf-disease/runs/segment/train/weights/best.pt`
(đường dẫn do Ultralytics tự tạo, `train_yolo.py` in ra đường dẫn chính xác ở cuối log) - copy
file này vào `models/model_segemnt_yolo.pt` để web app dùng được (xem
[../models/README.md](../models/README.md)).
