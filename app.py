import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras

st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="centered"
)

# Header
st.title("🎵 Deep Learning Music Genre Classifier")
st.write("""
This interactive app uses a pre-trained Deep Neural Network (Multi-Layer Perceptron) 
to classify audio tracks into genres using 518 precomputed acoustic features.
""")

# Get the directory where app.py resides to build absolute paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "genre_mlp.keras"
METADATA_PATH = BASE_DIR / "artifacts" / "metadata.json"
PREPARED_PATH = BASE_DIR / "artifacts" / "prepared_fma_small.npz"

@st.cache_resource
def load_assets():
    # Load class names
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    class_names = metadata["class_names"]
    
    # Load Model
    model = keras.saving.load_model(MODEL_PATH)
    
    # Load Scaler
    with np.load(PREPARED_PATH) as data:
        scaler_mean = data["scaler_mean"]
        scaler_scale = data["scaler_scale"]
        
    return model, class_names, scaler_mean, scaler_scale

try:
    model, class_names, scaler_mean, scaler_scale = load_assets()
    st.success("✅ Model and scaler assets successfully loaded!")
except Exception as e:
    st.error(f"❌ Could not load model assets. Error details: {e}")
    st.stop()

# User Inputs
st.subheader("🔬 Real-Time Classification Input")
st.write("Input a few baseline acoustic statistics to predict the genre, or simulate an analysis:")

if st.button("🎲 Generate Random Audio Feature Sample"):
    np.random.seed(42)
    # Generate a dummy feature vector matching FMA Small's 518 dimensions
    raw_features = np.random.normal(0, 1, size=(1, 518)).astype(np.float32)
    
    # Scale the features using our loaded training statistics
    scaled_features = (raw_features - scaler_mean) / scaler_scale
    
    # Classify
    probabilities = model.predict(scaled_features)[0]
    predicted_id = np.argmax(probabilities)
    confidence = probabilities[predicted_id]
    
    st.markdown("--- ")
    st.subheader(f"🎯 Prediction: **{class_names[predicted_id]}**")
    st.metric(label="Confidence Score", value=f"{confidence*100:.2f}%")
    
    # Show full distribution
    st.write("### Class Probabilities:")
    chart_data = pd.DataFrame({
        'Genre': class_names,
        'Probability': probabilities
    }).set_index('Genre')
    st.bar_chart(chart_data)
