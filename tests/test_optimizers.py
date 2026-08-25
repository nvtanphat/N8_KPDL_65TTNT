"""Đảm bảo get_optimizer_scheduler() của từng model khởi tạo được."""
import torch
from torch.utils.data import DataLoader, TensorDataset

from bean_leaf.models import bean_leaf_lite, efficientnet, mobilenetv3, resnet50, shufflenetv2

NUM_CLASSES = 3


def _dummy_loader(batch_size=2, n_batches=3):
    x = torch.randn(batch_size * n_batches, 3, 8, 8)
    y = torch.randint(0, NUM_CLASSES, (batch_size * n_batches,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def test_vgg_optimizer_scheduler():
    model = bean_leaf_lite.create_lite_model(NUM_CLASSES)
    loader = _dummy_loader()
    criterion, optimizer, scheduler = bean_leaf_lite.get_optimizer_scheduler(model, loader, num_epochs=2)
    assert criterion is not None and optimizer is not None and scheduler is not None


def test_efficientnet_optimizer_scheduler():
    model = efficientnet.create_efficientnet_model(NUM_CLASSES, pretrained=False)
    criterion, optimizer, scheduler = efficientnet.get_optimizer_scheduler(model, num_epochs=2)
    assert criterion is not None and optimizer is not None and scheduler is not None


def test_mobilenetv3_optimizer_scheduler():
    model = mobilenetv3.create_mobilenetv3_model(NUM_CLASSES, pretrained=False)
    criterion, optimizer, scheduler = mobilenetv3.get_optimizer_scheduler(model, num_epochs=2)
    assert criterion is not None and optimizer is not None and scheduler is not None


def test_resnet50_optimizer_scheduler():
    model = resnet50.create_resnet50_model(NUM_CLASSES, pretrained=False)
    criterion, optimizer, scheduler = resnet50.get_optimizer_scheduler(model, num_epochs=2)
    assert criterion is not None and optimizer is not None and scheduler is not None


def test_shufflenetv2_optimizer_scheduler():
    model = shufflenetv2.create_shufflenetv2_model(NUM_CLASSES, pretrained=False)
    criterion, optimizer, scheduler = shufflenetv2.get_optimizer_scheduler(model, num_epochs=2)
    assert criterion is not None and optimizer is not None and scheduler is not None
