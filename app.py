import streamlit as st
import joblib
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# App title
st.title("Fraud Detection Model")

# Input fields
st.subheader("Enter transaction details:")

distance_home = st.number_input("Distance from Home", min_value=0.0)
distance_last = st.number_input("Distance from Last Transaction", min_value=0.0)
ratio_price = st.number_input("Ratio to Median Purchase Price", min_value=0.0)
repeat_retailer = st.radio("Repeat Retailer", [0, 1])
used_chip = st.radio("Used Chip", [0, 1])
used_pin = st.radio("Used PIN Number", [0, 1])
online_order = st.radio("Online Order", [0, 1])

if st.button("Predict"):
    # Prepare input array
    numeric_input = np.array([[distance_home, distance_last, ratio_price]])
    scaled_numeric = scaler.transform(numeric_input)

    input_array = np.hstack((scaled_numeric, [[repeat_retailer, used_chip, used_pin, online_order]]))
    
    # Make prediction
    prediction = model.predict(input_array)

    result = "Fraud" if prediction[0] == 1 else "Not Fraud"
    st.success(f"Prediction: {result}")
