# Music Genre Classifier with Deep Learning (Keras & TensorFlow)

An end-to-end audio machine learning pipeline that classifies music tracks into eight distinct genres. The project processes precomputed acoustic features from the [Free Music Archive (FMA)](https://github.com/mdeff/fma), prevents artist-level data leakage, and benchmarks a baseline **Logistic Regression** classifier against a deep **Multi-Layer Perceptron (MLP)** built using Keras and TensorFlow.

The performance of the models is evaluated using test accuracy, macro F1-score, confidence score distributions, and a normalized confusion matrix.

---

## 📌 Project Overview
Audio classification is a core technology behind modern streaming services, content recommendation engines, and automated tagging systems. However, music audio features are highly non-linear, with overlapping frequencies and rhythmic patterns across genres.

This project solves this classification challenge by:
1. **Structuring a Leak-Free Pipeline**: Using strict, artist-separated splits (FMA small subset) to ensure the model generalizes to new artists rather than memorizing known production signatures.
2. **Standardizing on Training Statistics**: Fitting the feature standardizer strictly on the training partition to prevent validation/test target leakage.
3. **Establishing a Linear Baseline**: Fitting a regularized Logistic Regression model to define a clear performance threshold.
4. **Designing a Custom Deep Neural Network**: Training a Multi-Layer Perceptron in Keras with Dropout regularization and Early Stopping.
5. **Evaluating Operational Confidence**: Inspecting prediction confidence score distributions to understand where the model is uncertain.

---

## 📂 Repository Structure

*   `classify_genres.py`: Production training pipeline executing dataset ingestion, training/validation scaling, Keras model compiling, and evaluations.
*   `fma_metadata_features.csv`: Processed acoustic features (chroma, MFCCs, spectral contrast, etc.) from the FMA repository.
*   `classification_metrics.json`: Auto-saved evaluation results tracking Holdout Accuracy and Macro F1-score.
*   `confusion_matrix.png`: Normalized confusion matrix plotting true vs. predicted musical genres.
*   `README.md`: This comprehensive project documentation.

---

## 🛠️ Data Pipeline & Engineering

### 1. Ingestion & Feature Space
The pipeline utilizes the **FMA Small** subset, consisting of 8,000 tracks balanced perfectly across 8 genres (1,000 tracks each). The input features consist of 518 precomputed audio descriptors extracted using Librosa, capturing:
*   **Timbre**: Mel-Frequency Cepstral Coefficients (MFCCs)
*   **Pitch/Harmony**: Chroma features
*   **Spectral Shape**: Spectral centroid, bandwidth, rolloff, and spectral contrast
*   **Temporal/Rhythm**: Zero-crossing rate and estimated tempo

### 2. Leakage Prevention (Artist-Separated Splits)
Instead of a simple random split, we utilize the official FMA split partitions. This ensures that no artist present in the training set ever appears in the validation or test sets. This forces the neural network to learn core genre-defining acoustic characteristics rather than memorizing individual artists' specific production qualities.

---

## 📊 Models & Architecture

### 1. Baseline Model: Multinomial Logistic Regression
*   **Configuration**: L2 Regularization (Ridge), sag solver, maximum 1,000 iterations.
*   **Purpose**: Serves as a fast, highly interpretable baseline. If our deep learning model cannot beat this linear baseline, the architectural complexity of a neural network is not justified.

### 2. Deep Learning Model: Keras Multi-Layer Perceptron (MLP)
*   **Architecture**:
    *   **Input Layer**: 518 units (matching the audio feature dimension).
    *   **Hidden Layer 1**: 256 units with ReLU activation + Batch Normalization + 30% Dropout.
    *   **Hidden Layer 2**: 128 units with ReLU activation + Batch Normalization + 30% Dropout.
    *   **Output Layer**: 8 units with Softmax activation (multiclass probability distribution).
*   **Optimization**: Adam optimizer (initial learning rate = 0.001) minimizing Categorical Cross-Entropy.
*   **Regularization**: Early Stopping monitors validation loss with a patience of 15 epochs to prevent overfitting.

---

## 📈 Evaluation & Operational Insights

The final model is evaluated across multiple dimensions:

### 1. Classification Performance
*   **Accuracy**: Measures the overall percentage of correctly classified tracks.
*   **Macro F1-Score**: Calculates the unweighted average of F1-scores across all 8 genres, ensuring poor performance on any single genre isn't masked by high performance on others.

### 2. Normalized Confusion Matrix
A normalized confusion matrix is generated to diagnose classification bottlenecks. This typically reveals:
*   **High Precision**: Clear acoustic signatures (e.g., *Classical* vs. *Electronic*).
*   **Semantic Overlap**: Common classification confusion points, such as *Rock* tracks being misclassified as *Pop* or *Indie*, reflecting the subjective boundaries of modern music taxonomy.

### 3. Prediction Confidence
The pipeline logs the softmax probability scores of the correct class predictions. Analyzing this confidence distribution helps set minimum threshold parameters—for instance, setting an automated rule that tracks scoring below a 70% confidence score are routed to human music curators for manual genre tagging.

---

## 🚀 How to Run the Pipeline Locally

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/music-genre-classifier.git
    cd music-genre-classifier
    ```
2.  **Install Required Packages**:
    ```bash
    pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
    ```
3.  **Execute Ingestion & Neural Network Training**:
    ```bash
    python classify_genres.py
    ```

---

## 🚨 Production MLOps Configuration
The model training is configured to auto-evaluate daily at **3:00 AM UTC** to incorporate newly ingested music catalogs and verify that classification performance remains stable over time. If the rolling macro F1-score drops below **0.65**, an alert is automatically dispatched to the engineering team.
##Live Demo is available in 
https://musicgenreclassifiergit-l7oqyanqlkhxsowvcnhxbg.streamlit.app/
