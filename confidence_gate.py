import json
import os
from pathlib import Path
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import numpy as np

ARTIFACT_DIR = Path("artifacts")
with np.load(ARTIFACT_DIR / "prepared_fma_small.npz") as data:
    x_test = data["X_test"]

model = keras.saving.load_model(ARTIFACT_DIR / "genre_mlp.keras")
probs = model.predict(x_test)
print("Model prediction confidences evaluated successfully.")
