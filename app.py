# import streamlit as st
# import joblib
# import numpy as np
# import joblib

# # Load saved model and scaler
# model = joblib.load("random_forest_model.pkl")
# scaler = joblib.load("scaler.pkl")

# # App title
# st.title("Fraud Detection Model")

# # Input fields
# st.subheader("Enter transaction details:")

# distance_home = st.number_input("Distance from Home", min_value=0.0)
# distance_last = st.number_input("Distance from Last Transaction", min_value=0.0)
# ratio_price = st.number_input("Ratio to Median Purchase Price", min_value=0.0)
# repeat_retailer = st.radio("Repeat Retailer", [0, 1])
# used_chip = st.radio("Used Chip", [0, 1])
# used_pin = st.radio("Used PIN Number", [0, 1])
# online_order = st.radio("Online Order", [0, 1])

# if st.button("Predict"):
#     # Prepare input array
#     numeric_input = np.array([[distance_home, distance_last, ratio_price]])
#     scaled_numeric = scaler.transform(numeric_input)

#     input_array = np.hstack((scaled_numeric, [[repeat_retailer, used_chip, used_pin, online_order]]))
    
#     # Make prediction
#     prediction = model.predict(input_array)

#     result = "Fraud" if prediction[0] == 1 else "Not Fraud"
#     st.success(f"Prediction: {result}")

-------

#import streamlit as st
#import joblib
#import pandas as pd
#import numpy as np
#from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt

# Load saved Random Forest model and scaler
#model = joblib.load("random_forest_model.pkl")
#scaler = joblib.load("scaler.pkl")

#st.title("Credit Card Fraud Detection")

# Load the CSV file (make sure the file is in the same folder or provide full path)
#@st.cache_data
#def load_data():
    #return pd.read_csv("card_transdata.csv")

#df = load_data()

#st.subheader("Transaction Data Preview")
#st.dataframe(df.head())

# Assuming your CSV has columns matching the model input features, list them here:
#numeric_features = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']
#categorical_features = ['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']

# User inputs for prediction
#st.subheader("Enter Transaction Details for Prediction")

#distance_home = st.number_input("Distance from Home", min_value=0.0)
#distance_last = st.number_input("Distance from Last Transaction", min_value=0.0)
#ratio_price = st.number_input("Ratio to Median Purchase Price", min_value=0.0)
#repeat_retailer = st.radio("Repeat Retailer", [0, 1])
#used_chip = st.radio("Used Chip", [0, 1])
#used_pin = st.radio("Used PIN Number", [0, 1])
#online_order = st.radio("Online Order", [0, 1])

#if st.button("Predict"):
    # Prepare input data for model
    #numeric_input = np.array([[distance_home, distance_last, ratio_price]])
    #scaled_numeric = scaler.transform(numeric_input)
    #input_array = np.hstack((scaled_numeric, [[repeat_retailer, used_chip, used_pin, online_order]]))

    #prediction = model.predict(input_array)
    #result = "Fraud" if prediction[0] == 1 else "Not Fraud"
    #st.success(f"Prediction: {result}")

# ------- Now let's do batch predictions on the CSV data -------

#st.subheader("Batch Prediction on Dataset")

# Extract features from dataset
#X_num = df[numeric_features]
#X_cat = df[categorical_features]

# Scale numeric features
#X_num_scaled = scaler.transform(X_num)

# Combine scaled numeric and categorical features
#X = np.hstack((X_num_scaled, X_cat.values))

# Predict fraud for whole dataset
#df['rf_pred'] = model.predict(X)

# Display prediction counts
#st.write("Prediction counts:")
#st.write(df['rf_pred'].value_counts())

# Calculate accuracy if true labels are present
#if 'fraud' in df.columns:
    #y_true = df['fraud']
    #accuracy = accuracy_score(y_true, df['rf_pred'])
    #st.write(f"Random Forest Model Accuracy on dataset: {accuracy:.4f}")

    # Show pie chart of actual vs predicted fraud counts
    #fig, ax = plt.subplots(1, 2, figsize=(12,5))

    # Actual fraud distribution
    #ax[0].pie(y_true.value_counts(), labels=['Not Fraud', 'Fraud'], autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff6666'])
    #ax[0].set_title('Actual Fraud Distribution')

    # Predicted fraud distribution
    #ax[1].pie(df['rf_pred'].value_counts(), labels=['Not Fraud', 'Fraud'], autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff6666'])
    #ax[1].set_title('Predicted Fraud Distribution')

    #st.pyplot(fig)

#else:
    #st.warning("No 'fraud' column found in dataset to calculate accuracy.")

# Optional: show data with predictions
#if st.checkbox("Show dataset with predictions"):
    #st.dataframe(df)

----------


import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Credit Card Fraud Detection")

distance_home = st.number_input("Distance from Home", min_value=0.0, step=0.1, value=0.0)
distance_last = st.number_input("Distance from Last Transaction", min_value=0.0, step=0.1, value=0.0)
ratio_price = st.number_input("Ratio to Median Purchase Price", min_value=0.0, step=0.1, value=0.0)

repeat_retailer = st.radio("Repeat Retailer", [0, 1], horizontal=True)
used_chip = st.radio("Used Chip", [0, 1], horizontal=True)
used_pin = st.radio("Used PIN Number", [0, 1], horizontal=True)
online_order = st.radio("Online Order", [0, 1], horizontal=True)

threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.01)

def fraud_indicator(is_fraud, proba):
    if is_fraud:
        color = "#ff4b4b"  # Red
        text = "⚠️ FRAUD DETECTED!"
    else:
        color = "#4CAF50"  # Green
        text = "✅ Transaction Safe"
    html_code = (
        f'<div style="background-color: {color}; color: white; '
        'font-size: 32px; font-weight: bold; border-radius: 50%; '
        'width: 150px; height: 150px; display: flex; '
        'justify-content: center; align-items: center; margin: 20px auto;">'
        f'{text}</div>'
        f'<p style="text-align:center; font-size:16px;">Probability: {proba:.2f}</p>'
    )
    st.markdown(html_code, unsafe_allow_html=True)

if st.button("Predict Transaction"):
    if distance_home < 0 or distance_last < 0 or ratio_price < 0:
        st.error("Distance and Ratio values cannot be negative.")
    else:
        numeric_input = np.array([[distance_home, distance_last, ratio_price]])
        scaled_numeric = scaler.transform(numeric_input)
        input_array = np.hstack((scaled_numeric, [[repeat_retailer, used_chip, used_pin, online_order]]))
        proba = model.predict_proba(input_array)[0, 1]
        prediction = 1 if proba >= threshold else 0
        fraud_indicator(prediction == 1, proba)
        if prediction == 1 and st.checkbox("Flag this transaction for manual review"):
            st.info("Transaction flagged for review.")


