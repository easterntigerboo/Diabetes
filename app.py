import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set up page configurations
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
st.title("🩺 Medical Assistant: Diabetes Risk Assessment")
st.write("Enter patient metrics below to predict diabetes risk using the trained Random Forest model.")

# 1. Load the saved model and extract features dynamically
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('diabetes_rf_model.joblib')
        # Dynamically pull the exact features the model expects
        feature_columns = list(model.feature_names_in_)
        return model, feature_columns
    except FileNotFoundError:
        st.error("Error: Model files not found. Make sure diabetes_rf_model.joblib is in the same repository folder!")
        st.stop()

model, feature_columns = load_assets()

# 2. Create user interface input forms
st.subheader("Patient Clinical Measurements")
user_inputs = {}

# Layout inputs cleanly into two columns
col1, col2 = st.columns(2)

with col1:
    user_inputs['pregnant'] = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    user_inputs['glucose'] = st.slider("Glucose Level (mg/dL)", min_value=0, max_value=200, value=120)
    user_inputs['bp'] = st.slider("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
    user_inputs['skin'] = st.slider("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    user_inputs['insulin'] = st.slider("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
    user_inputs['bmi'] = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=70.0, value=32.0, step=0.1)
    user_inputs['pedigree'] = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    user_inputs['age'] = st.slider("Age (years)", min_value=1, max_value=120, value=30)

# 3. Predict Button Logic
if st.button("Analyze Diabetes Risk", type="primary"):
    # Map out the exact features matching your joblib layout
    payload = {
        'pregnant': user_inputs['pregnant'],
        'glucose': user_inputs['glucose'],
        'bp': user_inputs['bp'],
        'skin': user_inputs['skin'],
        'insulin': user_inputs['insulin'],
        'bmi': user_inputs['bmi'],
        'pedigree': user_inputs['pedigree'],
        'age': user_inputs['age']
    }
    
    # Pack into a DataFrame matching training structure columns
    input_df = pd.DataFrame([payload])[feature_columns]
    
    # Run predictions
    prediction = model.predict(input_df)
    probabilities = model.predict_proba(input_df)[0] # Extract matching class array bounds safely
    
    st.markdown("---")
    st.subheader("Analysis Outcome")
    
    if prediction[0] == 1:
        confidence = probabilities[1] * 100
        st.error(f"⚠️ **High Risk Identified.** (Confidence: {confidence:.1f}%)")
        st.write("The model profile strongly aligns with historic patterns of clinical diabetes. Further testing is highly advised.")
    else:
        confidence = probabilities[0] * 100
        st.success(f"✅ **Low Risk Identified.** (Confidence: {confidence:.1f}%)")
        st.write("The current patient metrics fall within standard boundaries for low immediate risk.")


