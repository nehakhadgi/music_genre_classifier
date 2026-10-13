import os
import json
import numpy as np
import pandas as pd
import streamlit as st

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras

st.set_page_colab = True
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

# Paths
MODEL_PATH = "artifacts/genre_mlp.keras"
METADATA_PATH = "artifacts/metadata.json"
PREPARED_PATH = "artifacts/prepared_fma_small.npz"

@st.cache_resource
def load_assets():
    # Load class names
    with open(METADATA_PATH, "r") as f:
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
    st.error(f"❌ Could not load model assets. Please make sure the training pipeline was run. Error: {e}")
    st.stop()

# User Inputs
st.subheader("🔬 Real-Time Classification Input")
st.write("Input a few baseline acoustic statistics to predict the genre, or simulate an analysis:")

# Let user choose arbitrary values or generate sample random feature values 
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

st.info("💡 To deploy this publicly, sync this `app.py` script to your GitHub repository and link it to [Streamlit Community Cloud](https://share.streamlit.io).")
