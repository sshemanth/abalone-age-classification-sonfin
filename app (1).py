# ============================================================
# ABALONE AGE CLASSIFICATION STREAMLIT APP
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Abalone Age Classification",
    page_icon="🐚",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_DIR = "models"

HIDDEN_COLUMNS = [
    "Length",
    "Diameter",
    "Height",
    "Whole_weight",
    "Shucked_weight",
    "Viscera_weight",
    "Shell_weight",
    "Sex_F",
    "Sex_I",
    "Sex_M"
]

RAW_COLUMNS = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight"
]

CLASS_LABELS = {
    0: "Young Abalone (Age ≤ 8)",
    1: "Middle Age Abalone (Age 9–10)",
    2: "Old Abalone (Age ≥ 11)"
}


# ============================================================
# ADVANCED FNN MODEL STRUCTURE
# ============================================================

class AdvancedFNN(nn.Module):
    def __init__(self, input_dim=10, num_classes=3):
        super(AdvancedFNN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    models = {}

    models["xgb"] = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    models["lgbm"] = joblib.load(os.path.join(MODEL_DIR, "lgbm_model.pkl"))
    models["cat"] = joblib.load(os.path.join(MODEL_DIR, "cat_model.pkl"))
    models["meta"] = joblib.load(os.path.join(MODEL_DIR, "meta_model.pkl"))

    # Load scaler if available
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    models["scaler"] = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    # Load Advanced FNN
    fnn = AdvancedFNN(input_dim=10, num_classes=3)
    fnn.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "advanced_fnn.pth"),
            map_location=torch.device("cpu")
        )
    )
    fnn.eval()
    models["fnn"] = fnn

    return models


# ============================================================
# PREPROCESS SINGLE OR BATCH INPUT
# ============================================================

def preprocess_input(df, scaler=None):
    df = df.copy()

    # Rename columns if user uploads raw Abalone-style CSV
    rename_map = {
        "Whole weight": "Whole_weight",
        "Shucked weight": "Shucked_weight",
        "Viscera weight": "Viscera_weight",
        "Shell weight": "Shell_weight"
    }

    df = df.rename(columns=rename_map)

    # One-hot encode Sex column if present
    if "Sex" in df.columns:
        df["Sex_F"] = (df["Sex"] == "F").astype(int)
        df["Sex_I"] = (df["Sex"] == "I").astype(int)
        df["Sex_M"] = (df["Sex"] == "M").astype(int)
        df = df.drop(columns=["Sex"])

    # Check required columns
    missing_cols = [col for col in HIDDEN_COLUMNS if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure correct column order
    df = df[HIDDEN_COLUMNS]

    # Convert to float
    X = df.astype(float).values

    # Apply scaler if saved from training notebook
    if scaler is not None:
        X = scaler.transform(X)

    return X.astype(np.float32)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_stacking(models, X):
    xgb_prob = models["xgb"].predict_proba(X)
    lgbm_prob = models["lgbm"].predict_proba(X)
    cat_prob = models["cat"].predict_proba(X)

    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        fnn_prob = torch.softmax(models["fnn"](X_tensor), dim=1).numpy()

    stack_features = np.hstack([
        xgb_prob,
        lgbm_prob,
        cat_prob,
        fnn_prob
    ])

    predictions = models["meta"].predict(stack_features)

    # If meta model supports predict_proba, use it for confidence
    if hasattr(models["meta"], "predict_proba"):
        confidence = models["meta"].predict_proba(stack_features)
    else:
        confidence = None

    return predictions, confidence


# ============================================================
# SIDEBAR INFORMATION
# ============================================================

st.sidebar.title("🐚 Abalone Classifier")
st.sidebar.markdown(
    """
    **Model Used:** Stacking Ensemble  
    **Base Models:** XGBoost, LightGBM, CatBoost, Advanced FNN  
    **Task:** 3-class Abalone age classification  
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Class Meaning")
st.sidebar.markdown("**Class 0:** Age ≤ 8")
st.sidebar.markdown("**Class 1:** Age 9–10")
st.sidebar.markdown("**Class 2:** Age ≥ 11")


# ============================================================
# MAIN TITLE
# ============================================================

st.title("🐚 Abalone Age Classification using Computational Intelligence")
st.markdown(
    """
    This Streamlit application demonstrates real-time Abalone age classification using the
    best-performing stacking ensemble model developed in the project.
    """
)


# ============================================================
# LOAD MODELS SAFELY
# ============================================================

try:
    models = load_models()
    st.success("Models loaded successfully.")
except Exception as e:
    st.error("Model loading failed. Please check that all model files are inside the models/ folder.")
    st.exception(e)
    st.stop()


# ============================================================
# USER MODE SELECTION
# ============================================================

mode = st.tabs(["Manual Prediction", "CSV Batch Prediction"])


# ============================================================
# MANUAL PREDICTION TAB
# ============================================================

with mode[0]:
    st.header("Manual Abalone Measurement Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        sex = st.selectbox("Sex", ["M", "F", "I"])
        length = st.number_input("Length", min_value=0.0, max_value=1.0, value=0.455, step=0.001)
        diameter = st.number_input("Diameter", min_value=0.0, max_value=1.0, value=0.365, step=0.001)

    with col2:
        height = st.number_input("Height", min_value=0.0, max_value=2.0, value=0.095, step=0.001)
        whole_weight = st.number_input("Whole weight", min_value=0.0, max_value=5.0, value=0.5140, step=0.0005)
        shucked_weight = st.number_input("Shucked weight", min_value=0.0, max_value=3.0, value=0.2245, step=0.0005)

    with col3:
        viscera_weight = st.number_input("Viscera weight", min_value=0.0, max_value=2.0, value=0.1010, step=0.0005)
        shell_weight = st.number_input("Shell weight", min_value=0.0, max_value=2.0, value=0.1500, step=0.0005)

    input_df = pd.DataFrame({
        "Sex": [sex],
        "Length": [length],
        "Diameter": [diameter],
        "Height": [height],
        "Whole weight": [whole_weight],
        "Shucked weight": [shucked_weight],
        "Viscera weight": [viscera_weight],
        "Shell weight": [shell_weight]
    })

    st.subheader("Input Preview")
    st.dataframe(input_df, use_container_width=True)

    if st.button("Predict Age Class", type="primary"):
        try:
            X_input = preprocess_input(input_df, scaler=models["scaler"])
            prediction, confidence = predict_stacking(models, X_input)

            pred_class = int(prediction[0])

            st.subheader("Prediction Result")
            st.success(f"Predicted Class: {pred_class} — {CLASS_LABELS[pred_class]}")

            if confidence is not None:
                confidence_df = pd.DataFrame({
                    "Class": ["Class 0", "Class 1", "Class 2"],
                    "Confidence": confidence[0]
                })

                st.subheader("Prediction Confidence")
                st.bar_chart(confidence_df.set_index("Class"))

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)


# ============================================================
# CSV BATCH PREDICTION TAB
# ============================================================

with mode[1]:
    st.header("Upload CSV File for Batch Prediction")

    st.markdown(
        """
        Your CSV should contain these columns:

        `Sex, Length, Diameter, Height, Whole weight, Shucked weight, Viscera weight, Shell weight`
        """
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data Preview")
            st.dataframe(batch_df.head(), use_container_width=True)

            X_batch = preprocess_input(batch_df, scaler=models["scaler"])
            predictions, confidence = predict_stacking(models, X_batch)

            result_df = batch_df.copy()
            result_df["Predicted_Class"] = predictions
            result_df["Prediction_Label"] = result_df["Predicted_Class"].map(CLASS_LABELS)

            if confidence is not None:
                result_df["Confidence_Class_0"] = confidence[:, 0]
                result_df["Confidence_Class_1"] = confidence[:, 1]
                result_df["Confidence_Class_2"] = confidence[:, 2]

            st.subheader("Prediction Results")
            st.dataframe(result_df, use_container_width=True)

            csv_output = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction Results",
                data=csv_output,
                file_name="abalone_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error("Batch prediction failed. Please check your CSV column names and values.")
            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "Developed for Abalone Age Classification using FNN, Neuro-Fuzzy, SONFIN, and Ensemble Learning."
)
