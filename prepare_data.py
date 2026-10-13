import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("fma_metadata")
ARTIFACT_DIR = Path("artifacts")
PREPARED_PATH = ARTIFACT_DIR / "prepared_fma_small.npz"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

TRACKS_PATH = DATA_DIR / "tracks.csv"
FEATURES_PATH = DATA_DIR / "features.csv"

if not TRACKS_PATH.exists() or not FEATURES_PATH.exists():
    raise FileNotFoundError("Extract fma_metadata.zip before running this script.")

ARTIFACT_DIR.mkdir(exist_ok=True)

tracks = pd.read_csv(TRACKS_PATH, index_col=0, header=[0, 1], low_memory=False)
features = pd.read_csv(FEATURES_PATH, index_col=0, header=[0, 1, 2])

small_mask = tracks[("set", "subset")].eq("small")
small_ids = tracks.index[small_mask]
small_tracks = tracks.loc[small_ids]
small_features = features.loc[small_ids]

raw_labels = small_tracks[("track", "genre_top")]
labels = raw_labels.astype(str)
class_names = np.array(sorted(labels.unique()), dtype=str)
class_to_id = {name: index for index, name in enumerate(class_names)}

clean_features = small_features.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any").astype("float32")
split_labels = small_tracks[("set", "split")].astype(str)

def extract_split(split_name):
    split_ids = small_tracks.index[split_labels.eq(split_name)]
    x_values = clean_features.loc[split_ids].to_numpy(dtype=np.float32)
    y_values = labels.loc[split_ids].map(class_to_id).to_numpy(dtype=np.int32)
    return split_ids.to_numpy(dtype=np.int64), x_values, y_values

train_ids, x_train, y_train = extract_split("training")
val_ids, x_val, y_val = extract_split("validation")
test_ids, x_test, y_test = extract_split("test")

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train).astype(np.float32)
x_val = scaler.transform(x_val).astype(np.float32)
x_test = scaler.transform(x_test).astype(np.float32)

np.savez(PREPARED_PATH, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
         X_train=x_train, y_train=y_train, X_val=x_val, y_val=y_val, X_test=x_test, y_test=y_test,
         scaler_mean=scaler.mean_.astype(np.float32), scaler_scale=scaler.scale_.astype(np.float32))

project_metadata = {
    "dataset": "FMA small",
    "class_names": class_names.tolist(),
    "feature_columns": [list(map(str, col)) for col in clean_features.columns],
}
with METADATA_PATH.open("w", encoding="utf-8") as file:
    json.dump(project_metadata, file, indent=2)

print(f"Saved dataset partitions and metadata config.")
