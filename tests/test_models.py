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


def test_mobilenetv3_freeze_unfreeze():
    model = mobilenetv3.create_mobilenetv3_model(NUM_CLASSES, pretrained=False)

    mobilenetv3.freeze_backbone(model, freeze=True)
    assert all(not p.requires_grad for p in model.features.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())

    mobilenetv3.unfreeze_backbone_for_finetune(model, freeze_ratio=0.7)
    trainable_blocks = [any(p.requires_grad for p in block.parameters()) for block in model.features.children()]
    assert any(trainable_blocks), "phải có ít nhất 1 block cuối được mở khóa ở phase 2"
    bn_trainable = [p.requires_grad for m in model.features.modules()
                    if isinstance(m, torch.nn.BatchNorm2d) for p in m.parameters()]
    assert not any(bn_trainable), "BatchNorm2d phải luôn bị đóng băng ở phase 2"


def test_deit_forward():
    model = deit.create_deit_model(NUM_CLASSES, pretrained=False)
    _assert_forward_shape(model, img_size=deit.IMG_SIZE)
