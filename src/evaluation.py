import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf
import cv2

def evaluate_model(model, val_generator):
    """
    Đánh giá model trên tập validation.
    """
    val_generator.reset()
    preds = model.predict(val_generator, verbose=1)
    
    if hasattr(preds, 'logits'): # ViT
        y_pred = np.argmax(preds.logits, axis=1)
    else:
        y_pred = np.argmax(preds, axis=1)
        
    y_true = val_generator.classes
    class_names = list(val_generator.class_indices.keys())
    
    # Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0) 
    return array

def plot_gradcam(model, img_path, img_size, alpha=0.4):
    """
    Vẽ Grad-CAM hoặc Attention Map.
    """
    img_array = get_img_array(img_path, (img_size, img_size)) / 255.0
    
    is_vit = 'ViT' in model.__class__.__name__ or 'DeiT' in model.__class__.__name__
    
    if is_vit:
        try:
            outputs = model(img_array, output_attentions=True)
            if outputs.attentions is None: return
            attentions = outputs.attentions[-1]
            att_map = tf.reduce_mean(attentions, axis=1)
            cls_attention = att_map[0, 0, 1:]
            grid_size = int(np.sqrt(cls_attention.shape[0]))
            heatmap = tf.reshape(cls_attention, (grid_size, grid_size)).numpy()
            heatmap = cv2.resize(heatmap, (img_size, img_size))
            heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        except:
            return
    else:
        last_conv_layer = None
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4 and 'conv' in layer.name:
                last_conv_layer = layer.name
                break
        
        if not last_conv_layer: return
        
        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer).output, model.output])
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()

    # Plotting code simplified...
    plt.imshow(heatmap)
    plt.show()


def plot_roc_auc(model, val_generator, class_names=None):
    """
    Vẽ ROC curve và tính AUC cho multi-class classification.
    
    Args:
        model: trained model
        val_generator: validation data generator
        class_names: list of class names
    """
    val_generator.reset()
    preds = model.predict(val_generator, verbose=1)
    
    # Handle ViT/DeiT models
    if hasattr(preds, 'logits'):
        y_score = tf.nn.softmax(preds.logits).numpy()
    else:
        y_score = preds
    
    y_true = val_generator.classes
    n_classes = y_score.shape[1]
    
    if class_names is None:
        class_names = list(val_generator.class_indices.keys())
    
    # Binarize labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average AUC
    roc_auc["macro"] = np.mean([roc_auc[i] for i in range(n_classes)])
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Plot ROC for each class
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    # Plot micro-average
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', lw=3,
             label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})')
    
    # Plot diagonal
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
    
    # Print AUC summary
    print("\n=== AUC Summary ===")
    for i in range(n_classes):
        print(f"{class_names[i]}: {roc_auc[i]:.4f}")
    print(f"Macro-average: {roc_auc['macro']:.4f}")
    print(f"Micro-average: {roc_auc['micro']:.4f}")
    
    return roc_auc
