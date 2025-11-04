import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load saved model and scaler if saved separately
model = joblib.load("logistic_regression_model.pkl")

# Since you trained StandardScaler in Python script, load it if saved
# If you didn't save, re-fit scaler on whole dataset for prediction purposes

# 🔹 Load and prepare data for scaler (same as training dataset)
data = pd.read_csv("breast_cancer_data.csv")
data = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)

scaler = StandardScaler()
scaler.fit(data)

st.title("Breast Cancer Prediction App 🩺")
st.write("Enter patient tumor measurements to predict whether tumor is **Benign (0)** or **Malignant (1)**.")

# UI input fields
def user_inputs():
    inputs = {}
    for col in data.columns:
        inputs[col] = st.number_input(f"{col}", value=float(data[col].mean()))
    
    return pd.DataFrame([inputs])

input_df = user_inputs()

# Prediction
if st.button("Predict Diagnosis"):
    scaled_features = scaler.transform(input_df)
    prediction = model.predict(scaled_features)
    prediction_prob = model.predict_proba(scaled_features)[0][1]

    st.write("### ✅ Prediction Result:")
    if prediction[0] == 1:
        st.error(f"**Malignant** (Cancer Detected) ⚠️\nProbability: {prediction_prob:.2f}")
    else:
        st.success(f"**Benign** (Non-Cancer) ✅\nProbability: {prediction_prob:.2f}")


