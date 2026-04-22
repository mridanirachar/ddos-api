import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

DATA_PATH    = "data/cicids2017_cleaned.csv"
MODEL_DIR    = "model"
WEIGHTS_PATH = os.path.join(MODEL_DIR, "dnn_weights.npz")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")
LABELS_PATH  = os.path.join(MODEL_DIR, "labels.txt")
LABEL_COL    = "Attack Type"

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

X = df.drop(columns=[LABEL_COL]).values.astype(np.float32)
y_raw = df[LABEL_COL].values

le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
num_classes = len(le.classes_)
y_cat = to_categorical(y_encoded, num_classes=num_classes)

with open(LABELS_PATH, "w") as f:
    f.write("\n".join(le.classes_))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
)

model = Sequential([
    Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(num_classes, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=256,
    callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    verbose=1,
)

weights = [w.numpy() for w in model.weights]
np.savez(WEIGHTS_PATH, *weights)
print(f"Done. Weights -> {WEIGHTS_PATH}")
