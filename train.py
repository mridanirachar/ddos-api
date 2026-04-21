"""
train.py — AI-Powered DDoS Detection
=====================================
Trains Random Forest (baseline) + Residual DNN on CIC-IDS2017.
Saves model, scaler, and labels into model/ directory for cloud deployment.

Usage:
    python train.py
    python train.py --data path/to/cicids2017_cleaned.csv
    python train.py --data cicids2017_cleaned.csv --epochs 50 --sample 300000
"""

import argparse
import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train DDoS detection model")
    p.add_argument("--data",    default="cicids2017_cleaned.csv")
    p.add_argument("--out",     default="model",     help="Output directory for model artifacts")
    p.add_argument("--plots",   default="outputs",   help="Directory for plots")
    p.add_argument("--epochs",  type=int, default=30)
    p.add_argument("--batch",   type=int, default=256)
    p.add_argument("--sample",  type=int, default=500_000,
                   help="Max rows to use (0 = use all)")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.7,
                   help="Confidence threshold for unknown detection")
    return p.parse_args()


# ─────────────────────────────────────────────────────────
# 1. LOAD & INSPECT
# ─────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    print(f"\n[1] Loading dataset: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    print(f"    Shape : {df.shape}")
    print(f"    Cols  : {list(df.columns[:5])} ... [{len(df.columns)} total]")
    return df


def detect_label_col(df: pd.DataFrame) -> str:
    candidates = ["label", "class", "attack_type", "category"]
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[-1]   # fallback


# ─────────────────────────────────────────────────────────
# 2. PREPROCESS
# ─────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame, label_col: str, sample: int, seed: int):
    print(f"\n[2] Preprocessing  (label='{label_col}')")

    # Remove inf / NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    print(f"    Dropped {before - len(df)} rows with NaN/inf")

    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
    y = df[label_col]
    print(f"    Features (numeric) : {X.shape[1]}")

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    print(f"    Classes ({n_classes}): {list(le.classes_)}")
    print(f"\n    Class distribution:\n{pd.Series(y_enc).value_counts().rename(dict(enumerate(le.classes_)))}")

    # Optional downsample
    if sample > 0 and len(X) > sample:
        print(f"    Sampling {sample:,} rows from {len(X):,}")
        idx = np.random.choice(len(X), sample, replace=False)
        X    = X.iloc[idx].reset_index(drop=True)
        y_enc = y_enc[idx]

    return X, y_enc, le


# ─────────────────────────────────────────────────────────
# 3. SPLIT & SCALE
# ─────────────────────────────────────────────────────────
def split_and_scale(X, y_enc, seed: int):
    print("\n[3] Splitting & scaling...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=seed, stratify=y_enc
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.1 / 0.8,   # 10% of original
        random_state=seed, stratify=y_train
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    print(f"    Train : {X_train_sc.shape}")
    print(f"    Val   : {X_val_sc.shape}")
    print(f"    Test  : {X_test_sc.shape}")

    return X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test, scaler


# ─────────────────────────────────────────────────────────
# 4. RANDOM FOREST BASELINE
# ─────────────────────────────────────────────────────────
def train_random_forest(X_train, y_train, X_test, y_test, le, seed):
    print("\n[4] Training Random Forest baseline...")

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1  = f1_score(y_test, rf_pred, average="weighted", zero_division=0)

    print(f"    Accuracy : {rf_acc * 100:.2f}%")
    print(f"    F1-Score : {rf_f1  * 100:.2f}%")
    print("\n    Classification Report (RF):")
    print(classification_report(y_test, rf_pred, target_names=le.classes_, zero_division=0))

    return rf, rf_pred, rf_acc, rf_f1


# ─────────────────────────────────────────────────────────
# 5. RESIDUAL DNN
# ─────────────────────────────────────────────────────────
def build_dnn(n_features: int, n_classes: int) -> Model:
    """
    Residual DNN for tabular network traffic features.

    Why NOT CNN-LSTM:
      CIC-IDS2017 features are flow-level statistics (mean, std, packet counts).
      They are NOT a time-series — treating each feature as a timestep and running
      LSTM/CNN over them creates spurious temporal patterns and causes overfitting.
      A deep feedforward network is the correct inductive bias for tabular data.

    Why residual connections:
      Skip connections allow gradients to flow directly to earlier layers,
      preventing vanishing gradients in deeper tabular nets.
    """
    def dense_block(x, units, dropout_rate=0.3):
        x = Dense(units)(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = Dropout(dropout_rate)(x)
        return x

    inp = Input(shape=(n_features,), name="input")

    # Block 1
    x = dense_block(inp, 256, dropout_rate=0.3)

    # Block 2 + residual skip
    skip = Dense(256)(inp)          # project input to same dim
    x2   = dense_block(x, 256, dropout_rate=0.3)
    x    = Add()([x2, skip])

    # Block 3
    x = dense_block(x, 128, dropout_rate=0.3)

    # Block 4
    x = dense_block(x, 64, dropout_rate=0.2)

    # Embedding (used for open-set confidence scoring)
    embedding = Dense(32, activation="relu", name="embedding")(x)

    # Output
    out = Dense(n_classes, activation="softmax", name="output")(embedding)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_dnn(X_train, X_val, X_test, y_train, y_val, y_test,
              n_classes, le, epochs, batch_size, seed):
    print("\n[5] Training Residual DNN...")

    # One-hot encode
    y_train_oh = to_categorical(y_train, n_classes)
    y_val_oh   = to_categorical(y_val,   n_classes)

    # Class weights for imbalanced data
    cw_vals = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(enumerate(cw_vals))

    n_features = X_train.shape[1]
    model = build_dnn(n_features, n_classes)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=7,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    dl_pred_prob = model.predict(X_test, verbose=0)
    dl_pred      = np.argmax(dl_pred_prob, axis=1)

    dl_acc = accuracy_score(y_test, dl_pred)
    dl_f1  = f1_score(y_test, dl_pred, average="weighted", zero_division=0)
    dl_pre = precision_score(y_test, dl_pred, average="weighted", zero_division=0)
    dl_rec = recall_score(y_test, dl_pred, average="weighted", zero_division=0)

    print(f"\n    DNN Accuracy  : {dl_acc * 100:.2f}%")
    print(f"    DNN Precision : {dl_pre * 100:.2f}%")
    print(f"    DNN Recall    : {dl_rec * 100:.2f}%")
    print(f"    DNN F1-Score  : {dl_f1  * 100:.2f}%")
    print("\n    Classification Report (DNN):")
    print(classification_report(y_test, dl_pred, target_names=le.classes_, zero_division=0))

    return model, history, dl_pred, dl_pred_prob, dl_acc, dl_f1


# ─────────────────────────────────────────────────────────
# 6. OPEN-SET / UNKNOWN DETECTION
# ─────────────────────────────────────────────────────────
def open_set_eval(dl_pred_prob, dl_pred, y_test, threshold):
    print(f"\n[6] Open-Set Evaluation (threshold={threshold})")

    max_conf     = dl_pred_prob.max(axis=1)
    unknown_mask = max_conf < threshold

    known_acc = accuracy_score(
        y_test[~unknown_mask], dl_pred[~unknown_mask]
    ) if (~unknown_mask).sum() > 0 else 0.0

    print(f"    Flagged as unknown/suspicious : {unknown_mask.sum()} "
          f"({unknown_mask.mean() * 100:.1f}%)")
    print(f"    Accuracy on confident preds   : {known_acc * 100:.2f}%")

    return unknown_mask, known_acc


# ─────────────────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────────────────
def save_plots(history, dl_pred, y_test, le,
               rf_acc, rf_f1, dl_acc, dl_f1,
               y_enc, plots_dir):
    print(f"\n[7] Saving plots → {plots_dir}/")
    os.makedirs(plots_dir, exist_ok=True)

    # — Training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"],     label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title("DNN Accuracy"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy");   axes[0].legend()

    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("DNN Loss"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss");    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{plots_dir}/training_curves.png", dpi=150)
    plt.close()

    # — Confusion Matrix
    cm  = confusion_matrix(y_test, dl_pred)
    n   = len(le.classes_)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title("DNN Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/confusion_matrix.png", dpi=150)
    plt.close()

    # — Class Distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    pd.Series(y_enc).map(dict(enumerate(le.classes_))) \
      .value_counts().plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Class Distribution"); ax.set_xlabel("Attack Type")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/class_distribution.png", dpi=150)
    plt.close()

    # — Model Comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    models = ["Random Forest", "DNN (Residual)"]
    accs   = [rf_acc * 100, dl_acc * 100]
    f1s    = [rf_f1  * 100, dl_f1  * 100]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, accs, w, label="Accuracy", color="#2196F3")
    ax.bar(x + w / 2, f1s,  w, label="F1-Score",  color="#4CAF50")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylim(0, 110); ax.set_ylabel("Score (%)")
    ax.set_title("Model Performance Comparison"); ax.legend()
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - w / 2, a + 0.5, f"{a:.1f}%", ha="center", fontsize=8)
        ax.text(i + w / 2, f + 0.5, f"{f:.1f}%", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/model_comparison.png", dpi=150)
    plt.close()

    print(f"    Saved 4 plots to {plots_dir}/")


# ─────────────────────────────────────────────────────────
# 8. SAVE ARTIFACTS  ← THIS IS THE CRITICAL PART
# ─────────────────────────────────────────────────────────
def save_artifacts(model, scaler, le, out_dir: str):
    """
    Save everything the cloud API needs:
      model/dnn_ddos_model.h5  — Keras model weights + architecture
      model/scaler.pkl         — SAME StandardScaler fitted on training data
      model/labels.txt         — class names in label-encoded order

    WHY scaler must be saved:
      StandardScaler computes mean and std from the TRAINING data.
      If you fit a new scaler at inference time (on the live input),
      the numbers will be completely different → wrong predictions.
      The cloud API MUST use the exact same scaler object.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. Keras model
    model_path = os.path.join(out_dir, "dnn_ddos_model.h5")
    model.save(model_path)
    print(f"    Model  → {model_path}  ({os.path.getsize(model_path) / 1e6:.1f} MB)")

    # 2. Scaler (pickle)
    scaler_path = os.path.join(out_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"    Scaler → {scaler_path}")

    # 3. Label names (one per line, in class index order)
    labels_path = os.path.join(out_dir, "labels.txt")
    with open(labels_path, "w") as f:
        f.write("\n".join(le.classes_) + "\n")
    print(f"    Labels → {labels_path}")
    for i, name in enumerate(le.classes_):
        print(f"             {i}: {name}")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    args = parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    os.makedirs(args.plots, exist_ok=True)
    os.makedirs(args.out,   exist_ok=True)

    print("=" * 60)
    print("  AI-Powered DDoS Detection — CIC-IDS2017")
    print("=" * 60)

    # Load
    df        = load_data(args.data)
    label_col = detect_label_col(df)

    # Preprocess
    X, y_enc, le = preprocess(df, label_col, args.sample, args.seed)
    n_classes     = len(le.classes_)

    # Split & scale
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     scaler) = split_and_scale(X, y_enc, args.seed)

    # Random Forest
    rf, rf_pred, rf_acc, rf_f1 = train_random_forest(
        X_train, y_train, X_test, y_test, le, args.seed
    )

    # DNN
    model, history, dl_pred, dl_pred_prob, dl_acc, dl_f1 = train_dnn(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        n_classes, le, args.epochs, args.batch, args.seed,
    )

    # Open-set eval
    unknown_mask, known_acc = open_set_eval(
        dl_pred_prob, dl_pred, y_test, args.threshold
    )

    # Comparison table
    print("\n[Comparison]")
    print("-" * 50)
    print(f"{'Model':<20} {'Accuracy':>10} {'F1-Score':>10}")
    print("-" * 50)
    print(f"{'Random Forest':<20} {rf_acc * 100:>9.2f}% {rf_f1 * 100:>9.2f}%")
    print(f"{'DNN (Residual)':<20} {dl_acc * 100:>9.2f}% {dl_f1 * 100:>9.2f}%")
    print("-" * 50)

    # Plots
    save_plots(history, dl_pred, y_test, le,
               rf_acc, rf_f1, dl_acc, dl_f1, y_enc, args.plots)

    # Save artifacts for deployment
    print(f"\n[8] Saving deployment artifacts → {args.out}/")
    save_artifacts(model, scaler, le, args.out)

    print("\n" + "=" * 60)
    print("  DONE")
    print(f"  RF  → Acc: {rf_acc * 100:.2f}%  F1: {rf_f1 * 100:.2f}%")
    print(f"  DNN → Acc: {dl_acc * 100:.2f}%  F1: {dl_f1 * 100:.2f}%")
    print(f"  Unknown flagged: {unknown_mask.sum()} ({unknown_mask.mean() * 100:.1f}%)")
    print(f"\n  Artifacts in ./{args.out}/")
    print(f"  Plots    in ./{args.plots}/")
    print("=" * 60)

    print(f"""
Next steps:
  1. python export_model.py --model {args.out}/dnn_ddos_model.h5 \\
         --labels "{','.join(le.classes_)}"
  2. git add {args.out}/ && git commit -m "add trained model"
  3. git push  →  Render auto-deploys
""")


if __name__ == "__main__":
    main()
