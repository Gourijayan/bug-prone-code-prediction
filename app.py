
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"
DATA_PATH = BASE_DIR / "data" / "cleaned_kc1.csv"

# Page configuration
st.set_page_config(
    page_title="Bug-Prone Code Prediction",
    page_icon="🐞",
    layout="wide"
)

st.title("🐞 Bug-Prone Code Prediction")
st.write(
    "Predict defect-prone software modules "
    "using Machine Learning."
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("Model not found. Run python main.py first.")
    st.stop()

model = load_model()

# Load dataset
df = pd.read_csv(DATA_PATH)

# Dataset overview
st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Modules", len(df))
col2.metric("Input Features", len(df.columns) - 1)
col3.metric("Defect-Prone Modules",
            int(df["defects"].sum()))

# Dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head(10))

# Defect distribution
st.subheader("Defect Distribution")

counts = df["defects"].value_counts()
counts.index = counts.index.map({
    0: "Not Defect-Prone",
    1: "Defect-Prone"
})

st.bar_chart(counts)

# Model comparison
st.header("Model Performance")

comparison_path = (
    BASE_DIR / "models" / "model_comparison.csv"
)

if comparison_path.exists():
    comparison = pd.read_csv(
        comparison_path,
        index_col=0
    )
    st.dataframe(comparison.round(4))
    st.bar_chart(comparison)
else:
    st.info("Run Step 6 to generate model comparison.")

# Prediction section
st.header("Predict Defect-Prone Modules")

st.write(
    "Upload a CSV file containing the same "
    "software metrics used during training."
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    input_data = pd.read_csv(uploaded_file)

    # Use the exact features used during training
    feature_names = list(model.feature_names_in_)

    missing_columns = [
        col for col in feature_names
        if col not in input_data.columns
    ]

    if missing_columns:
        st.error(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )
        st.stop()

    # Select features in correct order
    X_new = input_data[feature_names].copy()

    # Convert inputs to numeric
    for col in feature_names:
        X_new[col] = pd.to_numeric(
            X_new[col],
            errors="coerce"
        )

    if X_new.empty:
        st.error("Uploaded file contains no rows.")
        st.stop()

    if X_new.isna().all(axis=1).any():
        st.error(
            "Some rows have no valid numeric metrics."
        )
        st.stop()

    # Make predictions
    predictions = model.predict(X_new)

    probabilities = model.predict_proba(X_new)[:, 1]

    # Prepare results
    results = input_data.copy()

    results["Prediction"] = [
        "Defect-Prone" if p == 1
        else "Not Defect-Prone"
        for p in predictions
    ]

    results["Defect Risk (%)"] = (
        probabilities * 100
    ).round(2)

    st.subheader("Prediction Results")
    st.dataframe(results)

    # Summary
    st.metric(
        "Predicted Defect-Prone Modules",
        int(sum(predictions))
    )

    # Download results
    csv = results.to_csv(index=False)

    st.download_button(
        "Download Predictions",
        data=csv,
        file_name="bug_predictions.csv",
        mime="text/csv"
    )

st.info(
    "Predictions are estimates based on "
    "historical software metrics and should "
    "be reviewed by software testers."
)
