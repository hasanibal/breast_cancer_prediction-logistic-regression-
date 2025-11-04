import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load model & scaler
model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🔬 Breast Cancer Prediction App")
st.write("Enter tumor characteristics below to predict **Malignant** or **Benign**.")

# Feature columns from dataset (excluding ID/diagnosis)
feature_cols = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Input form
user_data = {}
st.subheader("📊 Enter Tumor Features")

for col in feature_cols:
    user_data[col] = st.number_input(col, value=0.0, format="%.4f")

input_df = pd.DataFrame([user_data])

# Predict button
if st.button("Predict"):
    # Scale input data
    scaled_input = scaler.transform(input_df)

    # Get prediction
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.subheader("🧾 Result")
    if prediction == 1:
        st.error(f"⚠️ Predicted: **Malignant Tumor** \nProbability: `{probability:.2f}`")
    else:
        st.success(f"✅ Predicted: **Benign Tumor** \nProbability: `{probability:.2f}`")
