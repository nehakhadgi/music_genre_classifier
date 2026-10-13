from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

PREPARED_PATH = Path("artifacts/prepared_fma_small.npz")
with np.load(PREPARED_PATH) as data:
    x_train, y_train = data["X_train"], data["y_train"]
    x_test, y_test = data["X_test"], data["y_test"]

baseline = LogisticRegression(C=1.0, max_iter=500)
baseline.fit(x_train, y_train)
preds = baseline.predict(x_test)
print(f"Baseline test accuracy: {accuracy_score(y_test, preds):.4f}")
