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
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# Load saved model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

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

# -------------------------------
# Accuracy Dashboard Section
# -------------------------------
st.markdown("---")
st.subheader("📊 ML Algorithms Accuracy Dashboard")

# Sample data - Replace with your actual results if different
ml_scores = {
    "Algorithm": ["Random Forest", "Logistic Regression", "SVM", "KNN", "XGBoost"],
    "Accuracy": [0.96, 0.91, 0.89, 0.88, 0.95]
}

df_scores = pd.DataFrame(ml_scores)

# Plotly bar chart
fig = px.bar(df_scores, x="Algorithm", y="Accuracy",
             color="Accuracy", title="Model Accuracy Comparison",
             text="Accuracy", height=400)

fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
fig.update_layout(yaxis_tickformat='.0%', yaxis_range=[0, 1.05])

st.plotly_chart(fig)

# Show data table
with st.expander("📄 Show Raw Accuracy Data"):
    st.dataframe(df_scores)

