import joblib
import numpy as np
import streamlit as st

# Load model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app setup
st.title("Credit Card Fraud Detection")

# Input form for all 7 features
st.write("Enter transaction details:")

distance_from_home = st.number_input("Distance from Home")
distance_from_last_transaction = st.number_input("Distance from Last Transaction")
ratio_to_median_purchase_price = st.number_input("Ratio to Median Purchase Price")
repeat_retailer = st.selectbox("Repeat Retailer", [0, 1])
used_chip = st.selectbox("Used Chip", [0, 1])
used_pin_number = st.selectbox("Used PIN Number", [0, 1])
online_order = st.selectbox("Online Order", [0, 1])

# Predict button
if st.button("Predict"):
    # Prepare the input features for prediction (7 features)
    features = np.array([[distance_from_home, distance_from_last_transaction, ratio_to_median_purchase_price,
                          repeat_retailer, used_chip, used_pin_number, online_order]])

    # Scale only the 3 selected features (using the scaler you have loaded)
    # We need to scale only the first 3 columns (which were used in training)
    features_to_scale = features[:, :3]  # Extract the first 3 columns
    features_scaled = scaler.transform(features_to_scale)  # Apply scaling to those 3 columns

    # Replace the original features with scaled ones
    features[:, :3] = features_scaled  # Replace scaled values in the features array

    # Make prediction
    prediction = model.predict(features)

    # Display result
    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Valid transaction")


