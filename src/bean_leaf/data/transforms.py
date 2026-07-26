"""
Augmentation dùng chung cho các model, tham số khớp với augmentation "bảo thủ"
đã dùng trong các notebook gốc (04, 05 - torchvision; 02 - Keras ImageDataGenerator
với rotation_range=10, shift=0.05, zoom_range=0.05, brightness=[0.9,1.1] tương đương):
flip cả 2 chiều + affine nhẹ (rotate ±10°, shift 5%, zoom 95-105%) + brightness jitter 10%.
"""
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(img_size=224, interpolation=InterpolationMode.BILINEAR):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=interpolation),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=0,
            interpolation=interpolation,
            fill=0,
        ),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_val_transform(img_size=224, interpolation=InterpolationMode.BILINEAR):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=interpolation),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
