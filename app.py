# 3. Predict Button Logic
if st.button("Analyze Diabetes Risk", type="primary"):
    # Create the single row payload dictionary
    # IMPORTANT: Ensure keys precisely map to the model's expected inputs
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
    
    # Transform dictionary directly into a baseline DataFrame matching feature columns
    input_df = pd.DataFrame([payload])[feature_columns]
    
    # Run the model predictions safely
    prediction = model.predict(input_df)
    probabilities = model.predict_proba(input_df)[0] # Extract array row bounds
    
    st.markdown("---")
    st.subheader("Analysis Outcome")
    
    if prediction[0] == 1:
        st.error(f"⚠️ **High Risk Identified.** (Confidence: {probabilities[1]*100:.1f}%)")
        st.write("The model profile strongly aligns with historic patterns of clinical diabetes. Further testing is highly advised.")
    else:
        st.success(f"✅ **Low Risk Identified.** (Confidence: {probabilities[0]*100:.1f}%)")
        st.write("The current patient metrics fall within standard boundaries for low immediate risk.")

