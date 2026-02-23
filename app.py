import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path

# Get the absolute path to the directory where this script is located
BASE_DIR = Path(__file__).parent.absolute()

st.title("🏦 Credit Risk Prediction App")
st.write("Enter applicant information to predict if the credit risk is good or bad")

# Check if model files exist
model_path = BASE_DIR / "extra_xgb_credit_model.pkl"
encoder_files = {
    "Sex": BASE_DIR / "Sex_encoder.pkl",
    "Housing": BASE_DIR / "Housing_encoder.pkl",
    "Saving accounts": BASE_DIR / "Saving accounts_encoder.pkl",
    "Checking account": BASE_DIR / "Checking account_encoder.pkl"
}

# Verify files exist
missing_files = []
if not model_path.exists():
    missing_files.append("extra_xgb_credit_model.pkl")

for name, path in encoder_files.items():
    if not path.exists():
        missing_files.append(f"{name}_encoder.pkl")

if missing_files:
    st.error(f"❌ Missing model files: {', '.join(missing_files)}")
    st.stop()

# Load model and encoders
try:
    model = joblib.load(model_path)
    encoders = {name: joblib.load(path) for name, path in encoder_files.items()}
    st.success("✅ Model and encoders loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model files: {str(e)}")
    st.stop()

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        sex = st.selectbox("Sex", ["male", "female"])
        job = st.number_input("Job (0–3)", min_value=0, max_value=3, value=1)
        housing = st.selectbox("Housing", ["own", "rent", "free"])
    
    with col2:
        saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich"])
        checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich"])
        credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
        duration = st.number_input("Duration (months)", min_value=1, value=12)
    
    submitted = st.form_submit_button("🔮 Predict Risk")

if submitted:
    # Prepare input data
    input_data = {
        "Age": [age],
        "Sex": [encoders["Sex"].transform([sex])[0]],
        "Job": [job],
        "Housing": [encoders["Housing"].transform([housing])[0]],
        "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
        "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
        "Credit amount": [credit_amount],
        "Duration": [duration]
    }
    
    input_df = pd.DataFrame(input_data)
    
    # Make prediction
    try:
        pred = model.predict(input_df)[0]
        
        if pred == 1:
            st.success("✅ The predicted credit risk is **GOOD**")
        else:
            st.error("❌ The predicted credit risk is **BAD**")
            
        # Show prediction probability if available
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            st.write(f"Confidence: Good: {proba[1]:.2%}, Bad: {proba[0]:.2%}")
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
