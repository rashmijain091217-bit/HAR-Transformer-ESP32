import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

os.makedirs("figures", exist_ok=True)

# -------------------------
# 1. TRAINING CURVES
# -------------------------
def save_training_curves(history):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title("Accuracy Curve")
    plt.legend()
    plt.savefig("figures/accuracy.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig("figures/loss.png", dpi=300)
    plt.close()


# -------------------------
# 2. CONFUSION MATRIX
# -------------------------
def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("figures/confusion_matrix.png", dpi=300)
    plt.close()


# -------------------------
# 3. ROC CURVE
# -------------------------
def save_roc_curve(y_true, y_score, num_classes):
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    plt.figure()

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")

    plt.plot([0,1],[0,1],'k--')
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("figures/roc_curve.png", dpi=300)
    plt.close()
