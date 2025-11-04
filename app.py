# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, classification_report

# Load model and scaler
model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🔬 Breast Cancer Prediction using Logistic Regression")
st.write("Upload patient data to predict whether tumor is **Malignant** or **Benign**.")

uploaded_file = st.file_uploader("Upload test CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("📂 Input Data")
    st.write(data.head())

    X = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1, errors='ignore')
    
    # Scale input data
    X_scaled = scaler.transform(X)

    # Predict
    preds = model.predict(X_scaled)
    data["Prediction"] = preds
    data["Prediction"] = data["Prediction"].map({1: "Malignant", 0: "Benign"})

    st.subheader("✅ Predictions")
    st.write(data[["Prediction"]])

    # Evaluation (if true labels exist)
    if "diagnosis" in data.columns:
        y_true = data["diagnosis"]
        cm = confusion_matrix(y_true, preds)
        acc = accuracy_score(y_true, preds)

        st.write(f"**Accuracy:** {acc:.4f}")
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        st.pyplot(fig)

        # ROC Curve
        y_prob = model.predict_proba(X_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC={auc:.2f}")
        ax2.plot([0,1], [0,1], 'k--')
        st.subheader("ROC Curve")
        st.pyplot(fig2)
else:
    st.info("👆 Upload a file to test the model.")
