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
    criterion, optimizer, scheduler = bean_leaf_lite.get_optimizer_scheduler(model, num_epochs=2)
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


def test_moi_model_dung_chung_1_recipe():
    """
    Chốt bằng test: không model nào được có optimizer/scheduler riêng.
    Đây là điều kiện để bảng benchmark là so sánh kiến trúc chứ không phải so sánh recipe -
    trước đây BeanLeafLite dùng OneCycleLR(max_lr=2e-3) trong khi 4 model kia dùng
    CosineAnnealingLR(lr=3e-4), tức peak LR gấp 6.7 lần.
    """
    from bean_leaf.config import DEFAULT_CONFIG

    builders = [
        (bean_leaf_lite, bean_leaf_lite.create_lite_model(NUM_CLASSES)),
        (efficientnet, efficientnet.create_efficientnet_model(NUM_CLASSES, pretrained=False)),
        (mobilenetv3, mobilenetv3.create_mobilenetv3_model(NUM_CLASSES, pretrained=False)),
        (resnet50, resnet50.create_resnet50_model(NUM_CLASSES, pretrained=False)),
        (shufflenetv2, shufflenetv2.create_shufflenetv2_model(NUM_CLASSES, pretrained=False)),
    ]

    for module, model in builders:
        name = module.__name__.rsplit('.', 1)[-1]
        criterion, optimizer, scheduler = module.get_optimizer_scheduler(model, num_epochs=7)

        assert isinstance(optimizer, torch.optim.AdamW), f"{name} không dùng AdamW"
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR),             f"{name} không dùng CosineAnnealingLR"
        assert scheduler.T_max == 7, f"{name} có T_max khác num_epochs"
        assert optimizer.param_groups[0]['lr'] == DEFAULT_CONFIG.learning_rate,             f"{name} có learning rate riêng"
        assert optimizer.param_groups[0]['weight_decay'] == DEFAULT_CONFIG.weight_decay,             f"{name} có weight decay riêng"
        assert criterion.label_smoothing == DEFAULT_CONFIG.label_smoothing,             f"{name} có label smoothing riêng"


def test_moi_model_train_one_epoch_cung_chu_ky():
    """train_one_epoch của mọi model phải cùng signature thì train.py mới gọi chung 1 nhánh."""
    import inspect

    modules = [bean_leaf_lite, efficientnet, mobilenetv3, resnet50, shufflenetv2]
    sigs = {m.__name__.rsplit('.', 1)[-1]: list(inspect.signature(m.train_one_epoch).parameters)
            for m in modules}
    assert len(set(map(tuple, sigs.values()))) == 1, f"signature lệch nhau: {sigs}"
