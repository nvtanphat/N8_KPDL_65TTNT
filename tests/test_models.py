"""Smoke tests: mỗi kiến trúc phải khởi tạo được và forward pass đúng shape output."""
import torch

from bean_leaf.models import bean_leaf_lite, efficientnet, mobilenetv3, resnet50, shufflenetv2

NUM_CLASSES = 3


def _assert_forward_shape(model, img_size):
    model.eval()
    x = torch.randn(2, 3, img_size, img_size)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, NUM_CLASSES)


def test_bean_leaf_lite_forward():
    model = bean_leaf_lite.create_lite_model(NUM_CLASSES)
    _assert_forward_shape(model, img_size=bean_leaf_lite.IMG_SIZE)


def test_efficientnet_forward():
    model = efficientnet.create_efficientnet_model(NUM_CLASSES, pretrained=False)
    _assert_forward_shape(model, img_size=efficientnet.IMG_SIZE)


def test_mobilenetv3_forward():
    model = mobilenetv3.create_mobilenetv3_model(NUM_CLASSES, pretrained=False)
    _assert_forward_shape(model, img_size=mobilenetv3.IMG_SIZE)


def test_resnet50_forward():
    model = resnet50.create_resnet50_model(NUM_CLASSES, pretrained=False)
    _assert_forward_shape(model, img_size=resnet50.IMG_SIZE)


def test_shufflenetv2_forward():
    model = shufflenetv2.create_shufflenetv2_model(NUM_CLASSES, pretrained=False)
    _assert_forward_shape(model, img_size=shufflenetv2.IMG_SIZE)


def test_4_model_pretrained_dung_chung_1_head():
    """
    Chốt bằng test: 4 model pretrained phải có head phân loại cùng cấu trúc.

    Trước đây MobileNetV3 có head riêng (Linear(960->256) + BatchNorm1d + SiLU + Dropout +
    Linear(256->3)) trong khi 3 model kia chỉ Dropout + Linear. Nó thêm ~0.24M params và
    MobileNetV3 lại đứng hạng 1 benchmark - không tách được đóng góp của head khỏi backbone.
    """
    import torch.nn as nn
    from bean_leaf.models import efficientnet, mobilenetv3, resnet50, shufflenetv2

    builders = {
        'mobilenetv3': mobilenetv3.create_mobilenetv3_model(NUM_CLASSES, pretrained=False),
        'efficientnet': efficientnet.create_efficientnet_model(NUM_CLASSES, pretrained=False),
        'resnet50': resnet50.create_resnet50_model(NUM_CLASSES, pretrained=False),
        'shufflenetv2': shufflenetv2.create_shufflenetv2_model(NUM_CLASSES, pretrained=False),
    }

    shapes = {}
    for name, model in builders.items():
        head = model.classifier if hasattr(model, 'classifier') else model.fc
        shapes[name] = tuple(type(layer).__name__ for layer in head)

    assert len(set(shapes.values())) == 1, f"head lệch nhau: {shapes}"

    # Lớp cuối phải map thẳng từ backbone ra num_classes, không qua tầng ẩn nào
    for name, model in builders.items():
        head = model.classifier if hasattr(model, 'classifier') else model.fc
        last = [layer for layer in head if isinstance(layer, nn.Linear)]
        assert len(last) == 1, f"{name} có nhiều hơn 1 Linear trong head"
        assert last[0].out_features == NUM_CLASSES

    # Dropout phải bằng nhau: ShuffleNetV2 từng để 0.2 trong khi 3 model kia dùng 0.3, tức
    # regularization nhẹ hơn - một hyperparameter không đồng nhất giữa các model được so sánh.
    drops = {n: [l.p for l in (m.classifier if hasattr(m, 'classifier') else m.fc)
                 if isinstance(l, nn.Dropout)][0] for n, m in builders.items()}
    assert set(drops.values()) == {0.3}, f"dropout lệch nhau: {drops}"
