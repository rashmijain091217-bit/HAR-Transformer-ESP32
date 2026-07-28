"""
model.py — Three HAR model architectures.

  1. build_transformer()  — Vanilla Transformer with learnable positional embeddings
  2. build_patchtst()     — PatchTST: channel-independent patch-based transformer
  3. build_lstm()         — Stacked Bidirectional LSTM baseline

All models accept input shape (batch, SEQ_LEN=128, N_CHANNELS=9)
and output a softmax distribution over N_CLASSES=6 activities.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from config import SEQ_LEN, N_CHANNELS, N_CLASSES, TRANSFORMER_CFG, PATCHTST_CFG, LSTM_CFG


# ══════════════════════════════════════════════════════════════════════
# SHARED BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════

class TransformerBlock(layers.Layer):
    """
    Single Transformer encoder block:
        MultiHeadAttention → Add & Norm → FFN → Add & Norm

    Implements Pre-LN (LayerNorm before sub-layer) which is more stable
    during training than the original Post-LN variant.
    """

    def __init__(self, d_model: int, num_heads: int, ff_dim: int,
                 dropout_rate: float, **kwargs):
        super().__init__(**kwargs)

        # ── Multi-Head Self-Attention ──────────────────────────────────
        self.attn    = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads,
            dropout=dropout_rate
        )

        # ── Feed-Forward Network (position-wise MLP) ───────────────────
        self.ffn     = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),   # Expand
            layers.Dropout(dropout_rate),
            layers.Dense(d_model),                     # Project back
        ])

        # ── Normalisation & Dropout ────────────────────────────────────
        self.norm1   = layers.LayerNormalization(epsilon=1e-6)
        self.norm2   = layers.LayerNormalization(epsilon=1e-6)
        self.drop1   = layers.Dropout(dropout_rate)
        self.drop2   = layers.Dropout(dropout_rate)

    def call(self, x, training=False, return_attention=False):
        # Pre-LN Multi-Head Attention + residual
        x_norm  = self.norm1(x)
        attn_out, attn_scores = self.attn(
            x_norm, x_norm, return_attention_scores=True, training=training
        )
        x = x + self.drop1(attn_out, training=training)

        # Pre-LN FFN + residual
        x_norm  = self.norm2(x)
        ffn_out = self.ffn(x_norm, training=training)
        x = x + self.drop2(ffn_out, training=training)

        if return_attention:
            return x, attn_scores
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.attn.key_dim, "num_heads": self.attn.num_heads})
        return cfg


class LearnablePositionalEmbedding(layers.Layer):
    """
    Learnable (not sinusoidal) positional embedding.
    An embedding table of shape (max_len, d_model) is trained end-to-end,
    giving the model flexibility to encode temporal position.
    """

    def __init__(self, max_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=d_model)
        self.max_len = max_len

    def call(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)  # (seq_len,)
        return x + self.pos_emb(positions)                      # broadcast

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_len": self.max_len, "d_model": self.pos_emb.output_dim})
        return cfg


# ══════════════════════════════════════════════════════════════════════
# 1. VANILLA TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

def build_transformer(cfg: dict = TRANSFORMER_CFG,
                      seq_len: int = SEQ_LEN,
                      n_channels: int = N_CHANNELS,
                      n_classes: int = N_CLASSES) -> Model:
    """
    Vanilla Transformer Encoder for time-series classification.

    Architecture:
        Input (batch, 128, 9)
        → Dense projection to d_model                  # Embed each timestep
        → Learnable Positional Embedding                # Add temporal position
        → N × TransformerBlock                         # Self-attention + FFN
        → GlobalAveragePooling1D                       # Aggregate over time
        → Dropout → Dense(n_classes, softmax)

    Args:
        cfg        : Hyperparameter dict from config.py
        seq_len    : Number of timesteps (128)
        n_channels : Number of input channels (9)
        n_classes  : Number of output classes (6)

    Returns:
        Compiled Keras Model.
    """
    d_model      = cfg["d_model"]
    num_heads    = cfg["num_heads"]
    ff_dim       = cfg["ff_dim"]
    num_layers   = cfg["num_layers"]
    dropout_rate = cfg["dropout_rate"]

    inputs = layers.Input(shape=(seq_len, n_channels), name="input")

    # ── Linear projection: each timestep vector → d_model ─────────────
    x = layers.Dense(d_model, name="token_embedding")(inputs)

    # ── Add learnable positional information ───────────────────────────
    x = LearnablePositionalEmbedding(max_len=seq_len, d_model=d_model,
                                     name="pos_embedding")(x)
    x = layers.Dropout(dropout_rate, name="input_dropout")(x)

    # ── Stack TransformerBlocks ────────────────────────────────────────
    for i in range(num_layers):
        x = TransformerBlock(
            d_model=d_model, num_heads=num_heads,
            ff_dim=ff_dim, dropout_rate=dropout_rate,
            name=f"transformer_block_{i}"
        )(x)

    # ── Aggregate: mean-pool over the time axis ────────────────────────
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)

    # ── Classification head ────────────────────────────────────────────
    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = Model(inputs, outputs, name="VanillaTransformer")
    return model


# ══════════════════════════════════════════════════════════════════════
# 2. PatchTST
# ══════════════════════════════════════════════════════════════════════

class PatchEmbedding(layers.Layer):
    """
    Splits the time-series into non-overlapping patches and linearly
    projects each patch to d_model.

    For PatchTST (Nie et al., 2022) on HAR, we treat each channel
    independently (channel-independent strategy) and handle them
    together via Dense projection over all channels simultaneously.

    Input : (batch, seq_len, n_channels)
    Output: (batch, num_patches, d_model)

    num_patches = seq_len // patch_size
    """

    def __init__(self, patch_size: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.proj = layers.Dense(d_model)

    def call(self, x):
        # x: (batch, seq_len, n_channels)
        batch     = tf.shape(x)[0]
        seq_len   = x.shape[1]
        n_patches = seq_len // self.patch_size

        # Reshape into patches: (batch, n_patches, patch_size * n_channels)
        x = tf.reshape(x, (batch, n_patches,
                            self.patch_size * x.shape[2]))
        # Linear projection → (batch, n_patches, d_model)
        return self.proj(x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"patch_size": self.patch_size})
        return cfg


def build_patchtst(cfg: dict = PATCHTST_CFG,
                   seq_len: int = SEQ_LEN,
                   n_channels: int = N_CHANNELS,
                   n_classes: int = N_CLASSES) -> Model:
    """
    PatchTST-inspired Transformer for HAR classification.

    Key idea: divide the 128-timestep sequence into non-overlapping
    patches (default size 16 → 8 patches).  Each patch is a local
    temporal segment, reducing sequence length and improving locality.

    Architecture:
        Input (batch, 128, 9)
        → PatchEmbedding → (batch, 8, d_model)
        → Learnable Positional Embedding
        → N × TransformerBlock
        → GlobalAveragePooling1D
        → Dense(n_classes, softmax)
    """
    patch_size   = cfg["patch_size"]
    d_model      = cfg["d_model"]
    num_heads    = cfg["num_heads"]
    ff_dim       = cfg["ff_dim"]
    num_layers   = cfg["num_layers"]
    dropout_rate = cfg["dropout_rate"]

    n_patches = seq_len // patch_size

    inputs = layers.Input(shape=(seq_len, n_channels), name="input")

    # ── Patch creation + projection ────────────────────────────────────
    x = PatchEmbedding(patch_size=patch_size, d_model=d_model,
                       name="patch_embedding")(inputs)

    # ── Learnable positional embedding over patch tokens ───────────────
    x = LearnablePositionalEmbedding(max_len=n_patches, d_model=d_model,
                                     name="pos_embedding")(x)
    x = layers.Dropout(dropout_rate, name="input_dropout")(x)

    # ── Stack TransformerBlocks ────────────────────────────────────────
    for i in range(num_layers):
        x = TransformerBlock(
            d_model=d_model, num_heads=num_heads,
            ff_dim=ff_dim, dropout_rate=dropout_rate,
            name=f"transformer_block_{i}"
        )(x)

    # ── Aggregate + classify ───────────────────────────────────────────
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = Model(inputs, outputs, name="PatchTST")
    return model


# ══════════════════════════════════════════════════════════════════════
# 3. LSTM BASELINE
# ══════════════════════════════════════════════════════════════════════

def build_lstm(cfg: dict = LSTM_CFG,
               seq_len: int = SEQ_LEN,
               n_channels: int = N_CHANNELS,
               n_classes: int = N_CLASSES) -> Model:
    """
    Stacked Bidirectional LSTM baseline.

    Architecture:
        Input (batch, 128, 9)
        → BiLSTM(128, return_sequences=True)
        → Dropout
        → BiLSTM(64)
        → Dropout
        → Dense(n_classes, softmax)

    Bidirectional LSTMs capture both past and future context at each
    timestep, consistently outperforming uni-directional baselines on HAR.
    """
    units        = cfg["units"]
    dropout_rate = cfg["dropout_rate"]

    inputs = layers.Input(shape=(seq_len, n_channels), name="input")
    x = inputs

    # ── Stacked Bidirectional LSTMs ────────────────────────────────────
    for i, u in enumerate(units):
        return_seq = (i < len(units) - 1)   # All but last return sequences
        x = layers.Bidirectional(
            layers.LSTM(u, return_sequences=return_seq),
            name=f"bilstm_{i}"
        )(x)
        x = layers.Dropout(dropout_rate, name=f"drop_{i}")(x)

    # ── Classification head ────────────────────────────────────────────
    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = Model(inputs, outputs, name="LSTM_Baseline")
    return model


# ══════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ══════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "transformer": build_transformer,
    "patchtst":    build_patchtst,
    "lstm":        build_lstm,
}

def build_model(name: str) -> Model:
    """
    Return a compiled Keras model by name.
    name ∈ {"transformer", "patchtst", "lstm"}
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]()


# ── Quick architecture summary (run file directly) ────────────────────
if __name__ == "__main__":
    for name in MODEL_REGISTRY:
        m = build_model(name)
        m.summary()
        print()
