"""Smoke tests: mỗi kiến trúc phải khởi tạo được và forward pass đúng shape output."""
import torch

from bean_leaf.models import vgg_custom, efficientnet, mobilenetv3, deit

NUM_CLASSES = 3


def _assert_forward_shape(model, img_size):
    model.eval()
    x = torch.randn(2, 3, img_size, img_size)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, NUM_CLASSES)


def test_vgg_custom_forward():
    model = vgg_custom.create_vgg_model(NUM_CLASSES)
    _assert_forward_shape(model, img_size=vgg_custom.IMG_SIZE)


def test_efficientnet_forward():
    model = efficientnet.create_efficientnet_model(NUM_CLASSES, pretrained=False)
    _assert_forward_shape(model, img_size=efficientnet.IMG_SIZE)


def test_mobilenetv3_forward():
    model = mobilenetv3.create_mobilenetv3_model(NUM_CLASSES, pretrained=False)
    _assert_forward_shape(model, img_size=mobilenetv3.IMG_SIZE)


def test_deit_forward():
    model = deit.create_deit_model(NUM_CLASSES, pretrained=False)
    _assert_forward_shape(model, img_size=deit.IMG_SIZE)
