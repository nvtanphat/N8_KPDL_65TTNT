import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


@torch.no_grad()
def _run_inference(model, val_loader, device):
    model.eval()
    all_labels = []
    all_scores = []

    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        scores = F.softmax(outputs, dim=1)
        all_scores.append(scores.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_scores)


def evaluate_model(model, val_loader, device, class_names):
    """
    Đánh giá model PyTorch trên tập validation: classification report + confusion matrix.
    """
    y_true, y_score = _run_inference(model, val_loader, device)
    y_pred = np.argmax(y_score, axis=1)

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def plot_gradcam(model, image_tensor, device, target_layer=None):
    """
    Grad-CAM cho model CNN dạng PyTorch (VGG/EfficientNet/MobileNetV3).
    image_tensor: tensor đã preprocess, shape (1, C, H, W).
    target_layer: nn.Conv2d cụ thể; nếu None sẽ tự tìm Conv2d cuối cùng trong model.

    Lưu ý: chỉ hỗ trợ kiến trúc CNN (có Conv2d). Với Vision Transformer (DeiT)
    không có khái niệm feature-map dạng lưới nên bỏ qua (trả về None).
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    if target_layer is None:
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
        if target_layer is None:
            print("Không tìm thấy Conv2d layer nào — model này không hỗ trợ Grad-CAM kiểu CNN "
                  "(vd: Vision Transformer như DeiT).")
            return None

    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations['value'] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    try:
        output = model(image_tensor)
        pred_index = output.argmax(dim=1)
        model.zero_grad()
        output[0, pred_index].backward()

        grads = gradients['value'][0]
        acts = activations['value'][0]
        weights = grads.mean(dim=(1, 2))

        heatmap = torch.zeros(acts.shape[1:], device=acts.device)
        for i, w in enumerate(weights):
            heatmap += w * acts[i]

        heatmap = torch.relu(heatmap)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        heatmap = heatmap.cpu().numpy()
    finally:
        handle_f.remove()
        handle_b.remove()

    plt.imshow(heatmap, cmap='jet')
    plt.title('Grad-CAM')
    plt.axis('off')
    plt.show()
    return heatmap


def plot_roc_auc(model, val_loader, device, class_names):
    """
    Vẽ ROC curve và tính AUC cho multi-class classification (PyTorch model).
    """
    y_true, y_score = _run_inference(model, val_loader, device)
    n_classes = y_score.shape[1]

    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    roc_auc["macro"] = np.mean([roc_auc[i] for i in range(n_classes)])

    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')

    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', lw=3,
             label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves (Macro AUC = {roc_auc["macro"]:.3f})', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\n=== AUC Summary ===")
    for i in range(n_classes):
        print(f"{class_names[i]}: {roc_auc[i]:.4f}")
    print(f"Macro-average: {roc_auc['macro']:.4f}")
    print(f"Micro-average: {roc_auc['micro']:.4f}")

    return roc_auc
