import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost
import sklearn
import os

st.set_page_config(page_title="Credit Risk Predictor", page_icon="🏦")

# Title and description
st.title("🏦 Credit Risk Prediction App")
st.write("Enter applicant information to predict credit risk (Good/Bad)")

# Load model and encoders with error handling
@st.cache_resource
def load_model():
    """Load the trained model and encoders"""
    try:
        model = joblib.load("extra_xgb_credit_model.pkl")
        
        encoders = {}
        encoder_files = ["Sex", "Housing", "Saving accounts", "Checking account"]
        for col in encoder_files:
            try:
                encoders[col] = joblib.load(f"{col}_encoder.pkl")
            except:
                st.warning(f"Could not load encoder for {col}")
                encoders[col] = None
                
        return model, encoders
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, encoders = load_model()

if model is None:
    st.error("❌ Model not found. Please check that model files are in the correct location.")
    st.stop()

# Create input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        sex = st.selectbox("Sex", ["male", "female"])
        job = st.number_input("Job Classification (0-3)", min_value=0, max_value=3, value=1)
        housing = st.selectbox("Housing", ["own", "rent", "free"])
    
    with col2:
        saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich"])
        checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich"])
        credit_amount = st.number_input("Credit Amount", min_value=100, max_value=20000, value=5000)
        duration = st.number_input("Duration (months)", min_value=1, max_value=72, value=12)
    
    submitted = st.form_submit_button("Predict Risk", type="primary")

if submitted:
    # Prepare input data
    input_data = {
        "Age": age,
        "Sex": sex,
        "Job": job,
        "Housing": housing,
        "Saving accounts": saving_accounts,
        "Checking account": checking_account,
        "Credit amount": credit_amount,
        "Duration": duration
    }
    
    # Display input summary
    st.subheader("📋 Application Summary")
    summary_df = pd.DataFrame([input_data])
    st.dataframe(summary_df, use_container_width=True)
    
    # Transform categorical features
    try:
        input_df = pd.DataFrame({
            "Age": [age],
            "Sex": [encoders["Sex"].transform([sex])[0] if encoders["Sex"] else 0],
            "Job": [job],
            "Housing": [encoders["Housing"].transform([housing])[0] if encoders["Housing"] else 0],
            "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0] if encoders["Saving accounts"] else 0],
            "Checking account": [encoders["Checking account"].transform([checking_account])[0] if encoders["Checking account"] else 0],
            "Credit amount": [credit_amount],
            "Duration": [duration]
        })
        
        # Make prediction
        with st.spinner("Analyzing..."):
            pred = model.predict(input_df)[0]
            
        # Show result
        st.subheader("🎯 Prediction Result")
        if pred == 1:
            st.success("### ✅ GOOD CREDIT RISK")
            st.balloons()
        else:
            st.error("### ❌ BAD CREDIT RISK")
            
        # Show confidence (optional)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            st.write(f"Confidence: Good: {proba[1]:.2%}, Bad: {proba[0]:.2%}")
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Add information section
with st.expander("ℹ️ About this App"):
    st.write("""
    This app predicts credit risk using the German Credit Dataset.
    - **Model**: XGBoost Classifier
    - **Accuracy**: ~67.6%
    - **Features**: Age, Sex, Job, Housing, Saving/Checking accounts, Credit amount, Duration
    """)
