import json
import os
from pathlib import Path
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import numpy as np

SEED = 42
keras.utils.set_random_seed(SEED)

ARTIFACT_DIR = Path("artifacts")
with np.load(ARTIFACT_DIR / "prepared_fma_small.npz") as data:
    x_train, y_train = data["X_train"], data["y_train"]
    x_val, y_val = data["X_val"], data["y_val"]

with open(ARTIFACT_DIR / "metadata.json") as f:
    metadata = json.load(f)

model = keras.Sequential([
    keras.Input(shape=(x_train.shape[1],)),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.30),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(len(metadata["class_names"]), activation="softmax"),
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=128, callbacks=[stopping], verbose=1)
model.save(ARTIFACT_DIR / "genre_mlp.keras")
