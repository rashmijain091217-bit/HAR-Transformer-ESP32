"""
config.py — Central configuration for the HAR Transformer project.
All hyperparameters, paths, and reproducibility seeds live here.
Editing this single file propagates changes to every other module.
"""

import os
import random
import numpy as np
import tensorflow as tf

# ─────────────────────────────────────────────
# 1. REPRODUCIBILITY
# ─────────────────────────────────────────────
SEED = 42

def set_global_seed(seed: int = SEED):
    """Fix every source of randomness for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_global_seed()

# ─────────────────────────────────────────────
# 2. DATASET PARAMETERS
# ─────────────────────────────────────────────
SEQ_LEN       = 128          # Fixed window length (UCI HAR is pre-windowed)
N_CHANNELS    = 9            # body_acc xyz, body_gyro xyz, total_acc xyz
N_CLASSES     = 6            # WALKING … LAYING
CLASS_NAMES   = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING",  "STANDING",        "LAYING"
]

# ─────────────────────────────────────────────
# 3. MODEL HYPERPARAMETERS
# ─────────────────────────────────────────────

# --- Shared ---
DROPOUT_RATE   = 0.1

# --- Vanilla Transformer ---
TRANSFORMER_CFG = dict(
    d_model       = 64,     # Embedding / model dimension
    num_heads     = 4,      # Multi-head attention heads
    ff_dim        = 128,    # Feed-forward hidden size
    num_layers    = 2,      # Number of stacked transformer blocks
    dropout_rate  = DROPOUT_RATE,
)

# --- PatchTST ---
PATCHTST_CFG = dict(
    patch_size    = 16,     # Each patch covers 16 timesteps
    d_model       = 64,
    num_heads     = 4,
    ff_dim        = 128,
    num_layers    = 2,
    dropout_rate  = DROPOUT_RATE,
)

# --- LSTM Baseline ---
LSTM_CFG = dict(
    units         = [128, 64],   # Two stacked LSTM layers
    dropout_rate  = DROPOUT_RATE,
)

# ─────────────────────────────────────────────
# 4. TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────
BATCH_SIZE      = 64
EPOCHS          = 20
LEARNING_RATE   = 1e-3
PATIENCE_ES     = 10        # EarlyStopping patience
PATIENCE_LR     = 5         # ReduceLROnPlateau patience
LR_FACTOR       = 0.5       # Multiplicative LR reduction factor
MIN_LR          = 1e-6

# ─────────────────────────────────────────────
# 5. CROSS-VALIDATION
# ─────────────────────────────────────────────
N_FOLDS = 5

# ─────────────────────────────────────────────
# 6. PROJECT PATHS
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR     = os.path.join(BASE_DIR, "figures")
TABLES_DIR      = os.path.join(BASE_DIR, "tables")
MODELS_DIR      = os.path.join(BASE_DIR, "saved_models")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")

for _dir in [FIGURES_DIR, TABLES_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ─────────────────────────────────────────────
# 7. FIGURE QUALITY
# ─────────────────────────────────────────────
FIG_DPI = 300
