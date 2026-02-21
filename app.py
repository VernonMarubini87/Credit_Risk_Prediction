import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost
import sklearn
import os

st.set_page_config(page_title="Credit Risk Predictor", page_icon="🏦")

st.title("🏦 Credit Risk Prediction App")
st.write("Enter applicant information to predict credit risk (Good/Bad)")

# Show all files in current directory (for debugging)
st.write("📁 Files found in directory:")
files = os.listdir('.')
st.write(files)

# Find model file
model_files = [f for f in files if f.endswith('.pkl') or 'model' in f.lower()]
st.write("🔍 Potential model files:", model_files)

# Load model and encoders with error handling
@st.cache_resource
def load_model():
    """Load the trained model and encoders"""
    try:
        # Try to find any .pkl file that might be the model
        model = None
        for file in files:
            if file.endswith('.pkl') and ('xgb' in file.lower() or 'model' in file.lower()):
                try:
                    model = joblib.load(file)
                    st.success(f"✅ Loaded model from: {file}")
                    break
                except:
                    continue
        
        if model is None:
            st.error("❌ Could not find a valid model file")
            return None, None
        
        encoders = {}
        encoder_files = ["Sex", "Housing", "Saving accounts", "Checking account"]
        for col in encoder_files:
            try:
                # Try different possible filenames
                possible_names = [
                    f"{col}_encoder.pkl",
                    f"{col}.pkl",
                    f"{col.replace(' ', '')}_encoder.pkl"
                ]
                
                for name in possible_names:
                    if name in files:
                        encoders[col] = joblib.load(name)
                        st.success(f"✅ Loaded encoder: {name}")
                        break
                else:
                    st.warning(f"⚠️ Could not load encoder for {col}")
                    encoders[col] = None
                    
            except Exception as e:
                st.warning(f"⚠️ Error loading encoder for {col}: {str(e)}")
                encoders[col] = None
                
        return model, encoders
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, encoders = load_model()

if model is None:
    st.error("❌ Model not found. Please check that model files are in the correct location.")
    st.stop()

# Rest of your form code here...
# (Keep your existing form code from here)
