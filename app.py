import streamlit as st
import pandas as pd
import joblib

# Load model and training columns
model, trained_columns = joblib.load("best_insurance_claim_model.pkl")

st.title("Travel Insurance Claim Predictor")

# User inputs
agency = st.text_input("Agency (e.g., CBH, CWT)")
agency_type = st.selectbox("Agency Type", ["Travel Agency", "Airline"])
distribution_channel = st.selectbox("Distribution Channel", ["Offline", "Online"])
product_name = st.text_input("Product Name")
duration = st.number_input("Duration (days)", min_value=1, max_value=365)
destination = st.text_input("Destination")
net_sales = st.number_input("Net Sales")
commission = st.number_input("Commission (in value)")
gender = st.selectbox("Gender", ["M", "F"])
age = st.number_input("Age", min_value=0, max_value=120)

if st.button("Predict Claim"):
    # Convert inputs to DataFrame
    new_data = pd.DataFrame([{
        "Agency": agency,
        "Agency Type": agency_type,
        "Distribution Channel": distribution_channel,
        "Product Name": product_name,
        "Duration": duration,
        "Destination": destination,
        "Net Sales": net_sales,
        "Commision (in value)": commission,
        "Gender": gender,
        "Age": age
    }])

    # One-hot encode like training
    new_data = pd.get_dummies(new_data)

    # Align with training columns
    new_data = new_data.reindex(columns=trained_columns, fill_value=0)

    # Make prediction
    prediction = model.predict(new_data)[0]
    probability = model.predict_proba(new_data)[:,1][0]

    st.subheader(f"Prediction: {'Claim' if prediction == 1 else 'No Claim'}")
    st.write(f"Probability of Claim: {probability:.2f}")
