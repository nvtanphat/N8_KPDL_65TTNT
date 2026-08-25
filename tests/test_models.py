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
