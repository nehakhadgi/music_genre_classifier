import json
import os
from pathlib import Path
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

ARTIFACT_DIR = Path("artifacts")
with np.load(ARTIFACT_DIR / "prepared_fma_small.npz") as data:
    x_test, y_test = data["X_test"], data["y_test"]

with open(ARTIFACT_DIR / "metadata.json") as f:
    class_names = json.load(f)["class_names"]

model = keras.saving.load_model(ARTIFACT_DIR / "genre_mlp.keras")
preds = model.predict(x_test).argmax(axis=1)

fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay.from_predictions(y_test, preds, display_labels=class_names, normalize="true", ax=ax, xticks_rotation=45)
plt.savefig(ARTIFACT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
print("Saved confusion matrix plot.")
