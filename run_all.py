"""
run_all.py — Master orchestration script.

Executes the full pipeline in order:
  1. Load & normalise dataset
  2. Train all models (full split + 5-fold CV)
  3. Evaluate all saved models
  4. Generate all figures and tables

Run from the har_transformer/ directory:
    python run_all.py

On Google Colab, run from a cell after mounting Drive:
    %run run_all.py
"""

import os
import numpy as np
import tensorflow as tf

from config        import MODELS_DIR, SEED
from dataset       import load_dataset, normalize
from model         import MODEL_REGISTRY
from train         import train_single, cross_validate, save_cv_table
from evaluate      import (evaluate_model, save_hyperparameters,
                            save_overall_metrics, save_per_class_metrics,
                            save_model_comparison)
from visualization import main as run_viz


def main():
    print("=" * 70)
    print("  HAR TRANSFORMER — Full Research Pipeline")
    print(f"  TensorFlow {tf.__version__}  |  GPU: {bool(tf.config.list_physical_devices('GPU'))}")
    print("=" * 70)

    # ── 1. Data ────────────────────────────────────────────────────────
    X_train, y_train, X_test, y_test = load_dataset("UCI HAR Dataset")
    X_train, X_test = normalize(X_train, X_test)

    # ── 2. Training ────────────────────────────────────────────────────
    histories   = {}   # { name: keras.History }
    trained_models = {}
    cv_results  = {}

    for model_name in MODEL_REGISTRY:
        result = train_single(model_name, X_train, y_train, X_test, y_test)
        histories[model_name]      = result["history"]
        trained_models[model_name] = result["model"]

        cv_res = cross_validate(model_name, X_train, y_train)
        cv_results[model_name] = cv_res

    save_cv_table(cv_results)

    # ── 3. Evaluation ──────────────────────────────────────────────────
    all_results = {}
    for model_name in MODEL_REGISTRY:
        path = os.path.join(MODELS_DIR, f"{model_name}_best.keras")
        if os.path.exists(path):
            model = tf.keras.models.load_model(path)
        else:
            model = trained_models[model_name]

        all_results[model_name] = evaluate_model(model, X_test, y_test, model_name)
        trained_models[model_name] = model   # keep reference for attention viz

    save_hyperparameters()
    save_overall_metrics(all_results)
    save_per_class_metrics(all_results)
    save_model_comparison(all_results)

    # ── 4. Visualisation ───────────────────────────────────────────────
    run_viz(
        histories   = histories,
        all_results = all_results,
        cv_results  = cv_results,
        models      = trained_models,
        X_test      = X_test,
        y_test      = y_test,
    )

    print("\n" + "=" * 70)
    print("  Pipeline complete.")
    print("  Figures  → har_transformer/figures/")
    print("  Tables   → har_transformer/tables/")
    print("  Models   → har_transformer/saved_models/")
    print("  TensorBoard logs → har_transformer/logs/")
    print("=" * 70)


if __name__ == "__main__":
    main()
