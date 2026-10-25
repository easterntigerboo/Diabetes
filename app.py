import streamlit as st
import pandas as pd
import joblib # Replace import pickle with import joblib

# Update your loading block to call joblib
@st.cache_resource
def load_assets():
    # Load using joblib's optimized file reconstruction
    model = joblib.load('diabetes_rf_model.joblib')
    features = joblib.load('feature_columns.joblib')
    return model, features

try:
    model, feature_columns = load_assets()
except FileNotFoundError:
    st.error("Error: Joblib model files not found in the current directory.")
    st.stop()
