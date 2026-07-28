"""
evaluate.py — Evaluation pipeline.

For each saved model computes:
  • Accuracy, Precision, Recall, Macro-F1, Weighted-F1
  • Full sklearn classification report
  • Confusion matrix (raw counts)

Saves:
  tables/overall_metrics.csv / .tex
  tables/per_class_metrics.csv / .tex
  tables/model_comparison.csv / .tex
  tables/hyperparameters.csv / .tex
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)

from config import (
    CLASS_NAMES, MODELS_DIR, TABLES_DIR,
    TRANSFORMER_CFG, PATCHTST_CFG, LSTM_CFG,
)
from dataset import load_dataset, normalize
from model  import MODEL_REGISTRY


# ══════════════════════════════════════════════════════════════════════
# CORE EVALUATION FUNCTION
# ══════════════════════════════════════════════════════════════════════

def evaluate_model(model: tf.keras.Model,
                   X_test: np.ndarray,
                   y_test:  np.ndarray,
                   model_name: str = "model") -> dict:
    """
    Compute all classification metrics for a trained model.

    Returns a dict with keys:
        accuracy, precision_macro, recall_macro,
        f1_macro, f1_weighted,
        report (dict), confusion_matrix (ndarray),
        y_pred (ndarray), y_proba (ndarray)
    """
    # Predicted probabilities and hard labels
    y_proba = model.predict(X_test, verbose=0)          # (N, 6)
    y_pred  = np.argmax(y_proba, axis=1)                # (N,)

    acc      = accuracy_score(y_test, y_pred)
    prec     = precision_score(y_test, y_pred, average="macro",    zero_division=0)
    rec      = recall_score(y_test,   y_pred, average="macro",    zero_division=0)
    f1_mac   = f1_score(y_test,       y_pred, average="macro",    zero_division=0)
    f1_wt    = f1_score(y_test,       y_pred, average="weighted", zero_division=0)

    report   = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0
    )
    cm       = confusion_matrix(y_test, y_pred)

    print(f"\n[evaluate] {model_name.upper()}")
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  Precision (macro): {prec:.4f}")
    print(f"  Recall (macro)   : {rec:.4f}")
    print(f"  F1 (macro)       : {f1_mac:.4f}")
    print(f"  F1 (weighted)    : {f1_wt:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES,
                                zero_division=0))

    return dict(
        accuracy       = acc,
        precision_macro= prec,
        recall_macro   = rec,
        f1_macro       = f1_mac,
        f1_weighted    = f1_wt,
        report         = report,
        confusion_matrix= cm,
        y_pred         = y_pred,
        y_proba        = y_proba,
    )


# ══════════════════════════════════════════════════════════════════════
# TABLE SAVERS
# ══════════════════════════════════════════════════════════════════════

def _save_csv_latex(df: pd.DataFrame, stem: str, caption: str, label: str):
    """Write a DataFrame as both .csv and a booktabs LaTeX table."""
    csv_path   = os.path.join(TABLES_DIR, f"{stem}.csv")
    latex_path = os.path.join(TABLES_DIR, f"{stem}.tex")

    df.to_csv(csv_path, index=False)
    with open(latex_path, "w") as f:
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write(df.to_latex(index=False, escape=True, float_format="%.4f"))
        f.write("\\end{table}\n")

    print(f"  Saved {csv_path}  &  {latex_path}")


def save_overall_metrics(all_results: dict):
    """
    Table 2 — Overall metrics for every model.
    Columns: Model | Accuracy | Precision | Recall | F1-Macro | F1-Weighted
    """
    rows = []
    for name, res in all_results.items():
        rows.append({
            "Model":       name,
            "Accuracy":    res["accuracy"],
            "Precision":   res["precision_macro"],
            "Recall":      res["recall_macro"],
            "F1-Macro":    res["f1_macro"],
            "F1-Weighted": res["f1_weighted"],
        })
    df = pd.DataFrame(rows)
    _save_csv_latex(df, "overall_metrics",
                    "Overall Classification Metrics", "tab:overall")
    return df


def save_per_class_metrics(all_results: dict):
    """
    Table 3 — Per-class Precision / Recall / F1 for every model.
    """
    rows = []
    for model_name, res in all_results.items():
        rpt = res["report"]
        for cls in CLASS_NAMES:
            rows.append({
                "Model":     model_name,
                "Class":     cls,
                "Precision": rpt[cls]["precision"],
                "Recall":    rpt[cls]["recall"],
                "F1":        rpt[cls]["f1-score"],
                "Support":   int(rpt[cls]["support"]),
            })
    df = pd.DataFrame(rows)
    _save_csv_latex(df, "per_class_metrics",
                    "Per-Class Classification Metrics", "tab:per_class")
    return df


def save_model_comparison(all_results: dict):
    """
    Table 5 — Side-by-side model comparison (LSTM vs Transformer vs PatchTST).
    """
    rows = []
    for name, res in all_results.items():
        rows.append({
            "Architecture": name,
            "Accuracy":     f"{res['accuracy']:.4f}",
            "F1-Macro":     f"{res['f1_macro']:.4f}",
            "F1-Weighted":  f"{res['f1_weighted']:.4f}",
        })
    df = pd.DataFrame(rows)
    _save_csv_latex(df, "model_comparison",
                    "Model Comparison: LSTM vs Transformer vs PatchTST",
                    "tab:comparison")
    return df


def save_hyperparameters():
    """
    Table 1 — Hyperparameters for all architectures.
    """
    rows = [
        # Transformer
        {"Architecture": "Transformer", "d_model": TRANSFORMER_CFG["d_model"],
         "num_heads": TRANSFORMER_CFG["num_heads"], "ff_dim": TRANSFORMER_CFG["ff_dim"],
         "num_layers": TRANSFORMER_CFG["num_layers"], "dropout": TRANSFORMER_CFG["dropout_rate"],
         "patch_size": "—"},
        # PatchTST
        {"Architecture": "PatchTST", "d_model": PATCHTST_CFG["d_model"],
         "num_heads": PATCHTST_CFG["num_heads"], "ff_dim": PATCHTST_CFG["ff_dim"],
         "num_layers": PATCHTST_CFG["num_layers"], "dropout": PATCHTST_CFG["dropout_rate"],
         "patch_size": PATCHTST_CFG["patch_size"]},
        # LSTM
        {"Architecture": "LSTM", "d_model": "—",
         "num_heads": "—", "ff_dim": "—",
         "num_layers": len(LSTM_CFG["units"]), "dropout": LSTM_CFG["dropout_rate"],
         "patch_size": "—"},
    ]
    df = pd.DataFrame(rows)
    _save_csv_latex(df, "hyperparameters",
                    "Model Hyperparameters", "tab:hyperparams")
    return df


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    # ── Load & normalise ───────────────────────────────────────────────
    X_train, y_train, X_test, y_test = load_dataset("UCI HAR Dataset")
    _, X_test = normalize(X_train, X_test)

    all_results = {}

    # ── Evaluate each saved model ──────────────────────────────────────
    for model_name in MODEL_REGISTRY:
        model_path = os.path.join(MODELS_DIR, f"{model_name}_best.keras")

        if not os.path.exists(model_path):
            print(f"[evaluate] WARNING: {model_path} not found — skipping.")
            continue

        model = tf.keras.models.load_model(model_path)
        all_results[model_name] = evaluate_model(model, X_test, y_test, model_name)

    if not all_results:
        print("[evaluate] No saved models found. Run train.py first.")
        return all_results

    # ── Save all tables ────────────────────────────────────────────────
    print("\n[evaluate] Saving tables …")
    save_hyperparameters()
    save_overall_metrics(all_results)
    save_per_class_metrics(all_results)
    save_model_comparison(all_results)

    print("\n[evaluate] Done.")
    return all_results


if __name__ == "__main__":
    results = main()
