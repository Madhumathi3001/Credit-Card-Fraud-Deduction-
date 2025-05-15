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




import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
import plotly.express as px

# Load scaler and Random Forest model
scaler = joblib.load("scaler.pkl")
rf_model = joblib.load("random_forest_model.pkl")

# Load dataset with true labels and features for evaluation
df = pd.read_csv("card_transdata.csv")

# Define feature columns (adjust to your dataset)
features = [
    'distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price',
    'repeat_retailer', 'used_chip', 'used_pin_number', 'online_order'
]

# Extract features and true labels
X = df[features]
y_true = df['fraud']  # Change if your label column has a different name

# Scale numeric features only
numeric_features = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']
X_numeric = scaler.transform(X[numeric_features])
X_scaled = np.hstack((X_numeric, X[['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']].values))

# Predict fraud using Random Forest
df['rf_pred'] = rf_model.predict(X_scaled)

# Calculate accuracy for Random Forest
accuracy = accuracy_score(y_true, df['rf_pred'])

# Prepare dataframe for accuracy display
accuracy_df = pd.DataFrame({
    'Model': ['Random Forest'],
    'Accuracy': [accuracy]
})

# Streamlit UI
st.title("Credit Card Fraud Detection Dashboard")

st.subheader("Random Forest Model Accuracy (Real-time)")
fig = px.bar(accuracy_df, x='Model', y='Accuracy', color='Model', text='Accuracy',
             title="Random Forest Model Accuracy")
st.plotly_chart(fig)

st.subheader("Enter transaction details for single prediction")

# Input fields
distance_home = st.number_input("Distance from Home", min_value=0.0)
distance_last = st.number_input("Distance from Last Transaction", min_value=0.0)
ratio_price = st.number_input("Ratio to Median Purchase Price", min_value=0.0)
repeat_retailer = st.radio("Repeat Retailer", [0, 1])
used_chip = st.radio("Used Chip", [0, 1])
used_pin = st.radio("Used PIN Number", [0, 1])
online_order = st.radio("Online Order", [0, 1])

if st.button("Predict with Random Forest"):
    # Scale numeric inputs
    numeric_input = np.array([[distance_home, distance_last, ratio_price]])
    scaled_numeric = scaler.transform(numeric_input)
    
    # Combine scaled and categorical inputs
    input_array = np.hstack((scaled_numeric, [[repeat_retailer, used_chip, used_pin, online_order]]))
    
    # Predict
    prediction = rf_model.predict(input_array)
    result = "Fraud" if prediction[0] == 1 else "Not Fraud"
    st.success(f"Prediction: {result}")
