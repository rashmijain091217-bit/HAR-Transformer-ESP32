import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

from config import (
    SEED,
    BATCH_SIZE,
    EPOCHS,
    N_FOLDS,
    N_CLASSES,
    CLASS_NAMES,
    MODELS_DIR,
    TABLES_DIR,
    FIGURES_DIR
)

from dataset import load_dataset, normalize
from model import build_model


# ==================================================
# Compile Model
# ==================================================
def compile_model(model):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==================================================
# Save Accuracy & Loss Curves
# ==================================================
def save_training_curves(history, model_name):

    plt.figure(figsize=(8,5))

    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")

    plt.title(f"{model_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            f"{model_name}_accuracy.png"
        ),
        dpi=300
    )

    plt.close()


    plt.figure(figsize=(8,5))

    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")

    plt.title(f"{model_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            f"{model_name}_loss.png"
        ),
        dpi=300
    )

    plt.close()


# ==================================================
# Save Confusion Matrix
# ==================================================
def save_confusion(y_true, y_pred, model_name):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.title(f"{model_name} Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            f"{model_name}_confusion.png"
        ),
        dpi=300
    )

    plt.close()


# ==================================================
# Save ROC Curve
# ==================================================
def save_roc(y_true, y_prob, model_name):

    y_true_bin = label_binarize(
        y_true,
        classes=np.arange(N_CLASSES)
    )

    plt.figure(figsize=(8,6))

    aucs = []

    for i in range(N_CLASSES):

        fpr, tpr, _ = roc_curve(
            y_true_bin[:, i],
            y_prob[:, i]
        )

        roc_auc = auc(fpr, tpr)

        aucs.append(roc_auc)

        plt.plot(
            fpr,
            tpr,
            label=f"{CLASS_NAMES[i]} (AUC={roc_auc:.2f})"
        )

    plt.plot([0,1], [0,1], "k--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(f"{model_name} ROC Curve")

    plt.legend(fontsize=8)

    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            f"{model_name}_roc.png"
        ),
        dpi=300
    )

    plt.close()

    return np.mean(aucs)


# ==================================================
# Save Test Results CSV
# ==================================================
def save_test_results(results):

    df = pd.DataFrame(results)

    df.to_csv(
        os.path.join(
            TABLES_DIR,
            "test_results.csv"
        ),
        index=False
    )

    print("\nSaved tables/test_results.csv")


# ==================================================
# Save Cross Validation CSV
# ==================================================
def save_cv_results(results):

    df = pd.DataFrame(results)

    df.to_csv(
        os.path.join(
            TABLES_DIR,
            "cv_results.csv"
        ),
        index=False
    )

    print("Saved tables/cv_results.csv")

    # ==================================================
# Train Single Model
# ==================================================
def train_single_model(
    model_name,
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n" + "=" * 50)
    print("TRAINING:", model_name.upper())
    print("=" * 50)

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)

    model = compile_model(build_model(model_name))

    checkpoint_path = os.path.join(
        MODELS_DIR,
        f"{model_name}_best.keras"
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1
        )
    ]

    history = model.fit(

        X_train,
        y_train,

        validation_data=(X_test, y_test),

        epochs=EPOCHS,
        batch_size=BATCH_SIZE,

        callbacks=callbacks,

        verbose=1
    )

    # ----------------------------------------
    # Predictions
    # ----------------------------------------

    y_prob = model.predict(X_test)

    y_pred = np.argmax(y_prob, axis=1)

    # ----------------------------------------
    # Metrics
    # ----------------------------------------

    acc = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro"
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    auc_score = save_roc(
        y_test,
        y_prob,
        model_name
    )

    print("\nClassification Report\n")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=CLASS_NAMES
        )
    )

    print(
        f"{model_name} "
        f"Accuracy={acc:.4f} "
        f"F1={f1:.4f} "
        f"AUC={auc_score:.4f}"
    )

    # ----------------------------------------
    # Save Plots
    # ----------------------------------------

    save_training_curves(
        history,
        model_name
    )

    save_confusion(
        y_test,
        y_pred,
        model_name
    )

    return (

        model,
        history,

        {
            "Model": model_name,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "Macro_F1": f1,
            "AUC": auc_score
        }
    )


# ==================================================
# 5-Fold Cross Validation part -2
# ==================================================
def run_cross_validation(
    model_name,
    X,
    y
):

    print("\n")
    print("=" * 50)
    print("5-FOLD CV:", model_name.upper())
    print("=" * 50)

    skf = StratifiedKFold(

        n_splits=N_FOLDS,

        shuffle=True,

        random_state=SEED
    )

    fold_results = []

    fold_no = 1

    for train_idx, val_idx in skf.split(X, y):

        print(f"\nFold {fold_no}/{N_FOLDS}")

        X_tr = X[train_idx]
        y_tr = y[train_idx]

        X_val = X[val_idx]
        y_val = y[val_idx]

        tf.keras.backend.clear_session()

        model = compile_model(
            build_model(model_name)
        )

        history = model.fit(

            X_tr,
            y_tr,

            validation_data=(X_val, y_val),

            epochs=EPOCHS,
            batch_size=BATCH_SIZE,

            verbose=0
        )

        _, val_acc = model.evaluate(

            X_val,
            y_val,

            verbose=0
        )

        print(
            f"Fold {fold_no} Accuracy: "
            f"{val_acc:.4f}"
        )

        fold_results.append(

            {
                "Model": model_name,
                "Fold": fold_no,
                "Accuracy": val_acc
            }
        )

        fold_no += 1

        gc.collect()

    return fold_results

# ==================================================
# Comparison Plots part - 3
# ==================================================
def plot_comparison_accuracy(histories):

    plt.figure(figsize=(8,5))

    for name, history in histories.items():

        plt.plot(
            history.history["val_accuracy"],
            label=name
        )

    plt.title("Validation Accuracy Comparison")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")

    plt.legend()

    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            "comparison_accuracy.png"
        ),
        dpi=300
    )

    plt.close()


def plot_5fold_comparison(cv_results):

    plt.figure(figsize=(8,5))

    models = sorted(
        list(
            set(
                [r["Model"] for r in cv_results]
            )
        )
    )

    for model in models:

        values = [
            r["Accuracy"]
            for r in cv_results
            if r["Model"] == model
        ]

        plt.plot(
            range(1, len(values)+1),
            values,
            marker="o",
            label=model
        )

    plt.title("5-Fold Cross Validation Comparison")

    plt.xlabel("Fold")
    plt.ylabel("Accuracy")

    plt.xticks(range(1, N_FOLDS + 1))

    plt.legend()

    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            "comparison_5fold.png"
        ),
        dpi=300
    )

    plt.close()


# ==================================================
# MAIN
# ==================================================
def main():

    print("\nLoading Dataset...\n")

    X_train, y_train, X_test, y_test = load_dataset(
        "UCI HAR Dataset"
    )

    X_train, X_test = normalize(
        X_train,
        X_test
    )

    models_to_run = [

        "transformer",
        "patchtst"

        # Uncomment if needed:
        # "lstm"
    ]

    histories = {}

    test_results = []

    cv_results = []

    for model_name in models_to_run:

        # ----------------------------------
        # Train/Test Evaluation
        # ----------------------------------

        model, history, metrics = train_single_model(

            model_name,

            X_train,
            y_train,

            X_test,
            y_test
        )

        histories[model_name] = history

        test_results.append(metrics)

        # ----------------------------------
        # Cross Validation
        # ----------------------------------

        cv_fold_results = run_cross_validation(

            model_name,

            X_train,
            y_train
        )

        cv_results.extend(
            cv_fold_results
        )

        gc.collect()

    # --------------------------------------
    # Save CSV Files
    # --------------------------------------

    save_test_results(
        test_results
    )

    save_cv_results(
        cv_results
    )

    # --------------------------------------
    # Comparison Plots
    # --------------------------------------

    plot_comparison_accuracy(
        histories
    )

    plot_5fold_comparison(
        cv_results
    )

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)

    print("\nSaved Figures:")

    print("figures/")
    print(" ├─ transformer_accuracy.png")
    print(" ├─ transformer_loss.png")
    print(" ├─ transformer_roc.png")
    print(" ├─ transformer_confusion.png")
    print(" ├─ patchtst_accuracy.png")
    print(" ├─ patchtst_loss.png")
    print(" ├─ patchtst_roc.png")
    print(" ├─ patchtst_confusion.png")
    print(" ├─ comparison_accuracy.png")
    print(" └─ comparison_5fold.png")

    print("\nSaved Tables:")

    print("tables/")
    print(" ├─ test_results.csv")
    print(" └─ cv_results.csv")

    print("\nSaved Models:")

    print("saved_models/")
    print(" ├─ transformer_best.keras")
    print(" └─ patchtst_best.keras")

    print("\nDone.\n")


# ==================================================
# RUN
# ==================================================
if __name__ == "__main__":

    main()
