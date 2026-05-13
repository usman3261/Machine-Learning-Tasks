import streamlit as st
import pandas as pd
import joblib

# Page Config
st.set_page_config(page_title="Churn Predictor", page_icon="📞")
st.title("📞 Telco Customer Churn Predictor")

# Load the production-ready pipeline
@st.cache_resource
def load_production_pipeline():
    return joblib.load("churn_pipeline.joblib")

try:
    pipeline = load_production_pipeline()

    st.markdown("Enter customer details to predict the likelihood of them leaving the service.")

    # Form for User Input
    with st.form("churn_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            tenure = st.number_input("Tenure (Months)", min_value=1, max_value=72, value=12)
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            
        with col2:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            monthly = st.number_input("Monthly Charges ($)", value=50.0)
            total = st.number_input("Total Charges ($)", value=600.0)
        
        submit = st.form_submit_button("Analyze Churn Risk")

    if submit:
        # Create a DataFrame with the exact column names used in training
        input_data = pd.DataFrame([{
            "gender": gender,
            "tenure": tenure,
            "InternetService": internet,
            "Contract": contract,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])
        
        # Inference
        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0][1]
        
        if prediction == 1:
            st.error(f"🚨 **High Risk**: Customer is likely to churn. (Probability: {probability:.2%})")
        else:
            st.success(f"✅ **Low Risk**: Customer is likely to stay. (Probability: {probability:.2%})")

except Exception as e:
    st.error("Model file not found. Please run the training notebook (task.ipynb) first.")