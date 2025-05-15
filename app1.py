import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load saved Random Forest model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Credit Card Fraud Detection")

# Load the CSV file (make sure the file is in the same folder or provide full path)
@st.cache_data
def load_data():
    return pd.read_csv("card_transdata.csv")

df = load_data()

st.subheader("Transaction Data Preview")
st.dataframe(df.head())

# Assuming your CSV has columns matching the model input features, list them here:
numeric_features = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']
categorical_features = ['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']

# User inputs for prediction
st.subheader("Enter Transaction Details for Prediction")

distance_home = st.number_input("Distance from Home", min_value=0.0)
distance_last = st.number_input("Distance from Last Transaction", min_value=0.0)
ratio_price = st.number_input("Ratio to Median Purchase Price", min_value=0.0)
repeat_retailer = st.radio("Repeat Retailer", [0, 1])
used_chip = st.radio("Used Chip", [0, 1])
used_pin = st.radio("Used PIN Number", [0, 1])
online_order = st.radio("Online Order", [0, 1])

if st.button("Predict"):
    # Prepare input data for model
    numeric_input = np.array([[distance_home, distance_last, ratio_price]])
    scaled_numeric = scaler.transform(numeric_input)
    input_array = np.hstack((scaled_numeric, [[repeat_retailer, used_chip, used_pin, online_order]]))

    prediction = model.predict(input_array)
    result = "Fraud" if prediction[0] == 1 else "Not Fraud"
    st.success(f"Prediction: {result}")

# ------- Now let's do batch predictions on the CSV data -------

st.subheader("Batch Prediction on Dataset")

# Extract features from dataset
X_num = df[numeric_features]
X_cat = df[categorical_features]

# Scale numeric features
X_num_scaled = scaler.transform(X_num)

# Combine scaled numeric and categorical features
X = np.hstack((X_num_scaled, X_cat.values))

# Predict fraud for whole dataset
df['rf_pred'] = model.predict(X)

# Display prediction counts
st.write("Prediction counts:")
st.write(df['rf_pred'].value_counts())

# Calculate accuracy if true labels are present
if 'fraud' in df.columns:
    y_true = df['fraud']
    accuracy = accuracy_score(y_true, df['rf_pred'])
    st.write(f"Random Forest Model Accuracy on dataset: {accuracy:.4f}")

    # Show pie chart of actual vs predicted fraud counts
    fig, ax = plt.subplots(1, 2, figsize=(12,5))

    # Actual fraud distribution
    ax[0].pie(y_true.value_counts(), labels=['Not Fraud', 'Fraud'], autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff6666'])
    ax[0].set_title('Actual Fraud Distribution')

    # Predicted fraud distribution
    ax[1].pie(df['rf_pred'].value_counts(), labels=['Not Fraud', 'Fraud'], autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff6666'])
    ax[1].set_title('Predicted Fraud Distribution')

    st.pyplot(fig)

else:
    st.warning("No 'fraud' column found in dataset to calculate accuracy.")

# Optional: show data with predictions
if st.checkbox("Show dataset with predictions"):
    st.dataframe(df)
