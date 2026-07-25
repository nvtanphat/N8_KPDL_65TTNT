"""Đảm bảo get_optimizer_scheduler() của từng model khởi tạo được (bắt regression như
DeiT từng thiếu scheduler và bị crash khi train)."""
import torch
from torch.utils.data import DataLoader, TensorDataset

from bean_leaf.models import vgg_custom, efficientnet, deit

NUM_CLASSES = 3


def _dummy_loader(batch_size=2, n_batches=3):
    x = torch.randn(batch_size * n_batches, 3, 8, 8)
    y = torch.randint(0, NUM_CLASSES, (batch_size * n_batches,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def test_vgg_optimizer_scheduler():
    model = vgg_custom.create_vgg_model(NUM_CLASSES)
    loader = _dummy_loader()
    criterion, optimizer, scheduler = vgg_custom.get_optimizer_scheduler(model, loader, num_epochs=2)
    assert criterion is not None and optimizer is not None and scheduler is not None


def test_efficientnet_optimizer_scheduler():
    model = efficientnet.create_efficientnet_model(NUM_CLASSES, pretrained=False)
    criterion, optimizer, scheduler = efficientnet.get_optimizer_scheduler(model, num_epochs=2)
    assert criterion is not None and optimizer is not None and scheduler is not None


def test_deit_optimizer_scheduler_and_ema():
    model = deit.create_deit_model(NUM_CLASSES, pretrained=False)
    loader = _dummy_loader()
    criterion, optimizer, scheduler, model_ema = deit.get_optimizer_scheduler(
        model, loader, num_epochs=2, warmup_epochs=1
    )
    assert criterion is not None and optimizer is not None
    assert scheduler is not None  # trước đây bị bỏ trống -> crash khi train_one_epoch gọi scheduler.step()
    assert model_ema is not None
