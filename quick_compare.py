import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from dataset import load_dataset
from model import build_model

# -----------------------------
# CONFIG
# -----------------------------
CLASS_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


# -----------------------------
# SAFE DATA LOADER FIX
# -----------------------------
def get_data():
    data = load_dataset()

    # handle both dict or tuple formats safely
    if isinstance(data, dict):
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]
    else:
        X_train, y_train, X_test, y_test = data

    print("X_test shape:", X_test.shape)
    return X_train, y_train, X_test, y_test


# -----------------------------
# ROC FUNCTION
# -----------------------------
def plot_roc(model, X_test, y_test, name):
    n_classes = len(CLASS_NAMES)

    y_bin = label_binarize(y_test, classes=list(range(n_classes)))

    y_proba = model.predict(X_test, verbose=0)

    plt.figure(figsize=(7, 6))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=1.5,
                 label=f"{CLASS_NAMES[i]} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"ROC Curve - {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)

    path = os.path.join(FIG_DIR, f"roc_{name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved {path}")


# -----------------------------
# SAFE MODEL LOADER
# -----------------------------
def load_model_safe(name):
    print(f"\nLoading model: {name}")

    model = build_model(name)

    path = f"saved_models/{name}_best.keras"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")

    model.load_weights(path)
    return model


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("\nLoading dataset...")
    X_train, y_train, X_test, y_test = get_data()

    models = ["transformer", "patchtst"]

    results = {}

    for name in models:

        model = load_model_safe(name)

        print(f"Predicting {name}...")
        y_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_proba, axis=1)

        acc = np.mean(y_pred == y_test)
        results[name] = acc

        print(f"{name} Accuracy: {acc:.4f}")

        plot_roc(model, X_test, y_test, name)

    # -----------------------------
    # COMPARISON PLOT
    # -----------------------------
    plt.figure(figsize=(5, 4))
    plt.bar(results.keys(), results.values())

    plt.title("Transformer vs PatchTST Accuracy")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    for i, v in enumerate(results.values()):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

    out_path = os.path.join(FIG_DIR, "model_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nFINAL RESULTS:", results)
    print(f"[OK] Saved {out_path}")


if __name__ == "__main__":
    main()
