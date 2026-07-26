"""
YOLOv8 Segmentation Model for Bean Leaf Disease Detection

Khác với 4 model classification (VGG/EfficientNet/MobileNetV3/DeiT) dùng chung
DataLoader + vòng train thủ công trong scripts/train.py, YOLO dùng thẳng API của
Ultralytics: kiến trúc + augmentation + vòng train/eval đều nằm trong model.train()/
model.val(), đọc dữ liệu qua 1 file data.yaml (định dạng segmentation, tải từ
Roboflow) chứ không phải cấu trúc train/val ImageFolder. Vì vậy module này không có
train_one_epoch/validate/get_optimizer_scheduler như các model khác - xem
scripts/train_yolo.py cho entrypoint train riêng.
"""
from ultralytics import YOLO

MODEL_SIZE = "n"  # n, s, m, l, x (nano -> xlarge)
IMG_SIZE = 640
BATCH_SIZE = 16
NUM_EPOCHS = 100
PATIENCE = 20


def create_yolo_model(model_size=MODEL_SIZE):
    """Load kiến trúc YOLOv8-seg pretrained (COCO) - Ultralytics gộp chung kiến trúc + weight."""
    return YOLO(f"yolov8{model_size}-seg.pt")


def load_yolo_checkpoint(model_path):
    """Load lại 1 checkpoint đã train (vd: best.pt) để đánh giá/inference."""
    return YOLO(model_path)


def train_yolo_model(model, data_yaml_path, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                      img_size=IMG_SIZE, patience=PATIENCE, device=None,
                      weight_decay=1e-3, mixup=0.0):
    """
    Train qua Ultralytics API, augmentation config khớp notebook 06_yolo_segmentation,
    trừ 2 chỗ đã tune riêng cho model size "s" (nhiều tham số hơn "n" ~3.6x, dataset chỉ
    ~3100 ảnh train nên dễ overfit hơn nếu giữ nguyên regularization của "n"):
      - weight_decay: 5e-4 (mặc định Ultralytics, tối ưu cho "n") -> 1e-3, phạt trọng số
        mạnh hơn tương ứng với model lớn hơn.
      - mixup: 0.1 -> 0.0, vì trộn ảnh (mixup) làm mờ ranh giới vết bệnh nhỏ/rải rác
        (đặc biệt bean_rust) - hại nhiều hơn lợi cho segmentation mask ở dataset này.
    """
    return model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=patience,
        save=True,
        save_period=10,
        device=device if device is not None else 0,
        workers=4,
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        weight_decay=weight_decay,
        verbose=True,
        seed=42,
        deterministic=True,
        # Data augmentation
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=5.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=mixup,
    )


def evaluate_yolo_model(model, data_yaml_path, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
                         device=None, split='test'):
    """
    Đánh giá mAP cuối cùng - mặc định split='test' (data.yaml của Roboflow có sẵn 3-way
    split train/val/test). Ultralytics tự dùng 'val' nội bộ trong model.train() để chọn
    best.pt/early-stopping (patience) - nếu evaluate_yolo_model cũng chấm trên 'val' thì
    con số báo cáo sẽ thiên vị lạc quan (chấm đúng tập đã chọn best.pt theo nó). 'test'
    chưa từng được nhìn thấy trong lúc train nên là con số đáng tin cậy để báo cáo.
    """
    return model.val(
        data=data_yaml_path,
        split=split,
        imgsz=img_size,
        batch=batch_size,
        conf=0.25,
        iou=0.6,
        device=device if device is not None else 0,
    )
