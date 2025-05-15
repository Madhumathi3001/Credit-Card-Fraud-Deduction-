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
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load saved model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Credit Card Fraud Detection")

# Cache dataset loading for performance
@st.cache_data
def load_data():
    return pd.read_csv("card_transdata.csv")

# Upload CSV option for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction (optional)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Uploaded file loaded successfully!")
else:
    df = load_data()

st.subheader("Transaction Data Preview")
st.dataframe(df.head())

# Features
numeric_features = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']
categorical_features = ['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']

# Input validation function
def validate_inputs(distance_home, distance_last, ratio_price):
    errors = []
    if distance_home < 0:
        errors.append("Distance from Home cannot be negative.")
    if distance_last < 0:
        errors.append("Distance from Last Transaction cannot be negative.")
    if ratio_price < 0:
        errors.append("Ratio to Median Purchase Price cannot be negative.")
    return errors

# Single transaction input section
st.subheader("Enter Transaction Details for Prediction")

distance_home = st.number_input("Distance from Home", min_value=0.0, value=0.0, step=0.1)
distance_last = st.number_input("Distance from Last Transaction", min_value=0.0, value=0.0, step=0.1)
ratio_price = st.number_input("Ratio to Median Purchase Price", min_value=0.0, value=0.0, step=0.1)
repeat_retailer = st.radio("Repeat Retailer", [0, 1], horizontal=True)
used_chip = st.radio("Used Chip", [0, 1], horizontal=True)
used_pin = st.radio("Used PIN Number", [0, 1], horizontal=True)
online_order = st.radio("Online Order", [0, 1], horizontal=True)

# Threshold slider
st.subheader("Adjust Prediction Threshold")
threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.01)

# Validate inputs
errors = validate_inputs(distance_home, distance_last, ratio_price)
if errors:
    for err in errors:
        st.error(err)

# Prediction button for single input
if st.button("Predict Single Transaction") and not errors:
    numeric_input = np.array([[distance_home, distance_last, ratio_price]])
    scaled_numeric = scaler.transform(numeric_input)
    input_array = np.hstack((scaled_numeric, [[repeat_retailer, used_chip, used_pin, online_order]]))

    proba = model.predict_proba(input_array)[0,1]
    prediction = 1 if proba >= threshold else 0

    # Display with colored alert boxes
    if prediction == 1:
        st.markdown(
            f"""
            <div style="background-color:#ff4b4b;padding:10px;border-radius:5px;color:white;font-weight:bold;">
            ⚠️ ALERT: This transaction is predicted as <u>FRAUD</u> with probability {proba:.2f}
            </div>
            """, unsafe_allow_html=True)
        if st.checkbox("Flag this transaction for manual review"):
            st.info("Transaction flagged for review.")
    else:
        st.markdown(
            f"""
            <div style="background-color:#4CAF50;padding:10px;border-radius:5px;color:white;font-weight:bold;">
            ✅ Prediction: NOT Fraud (Probability: {proba:.2f})
            </div>
            """, unsafe_allow_html=True)

# -------- Batch Prediction --------

st.subheader("Batch Prediction on Dataset")

missing_cols = [col for col in numeric_features + categorical_features if col not in df.columns]
if missing_cols:
    st.error(f"Dataset is missing required columns: {missing_cols}")
else:
    X_num = df[numeric_features]
    X_cat = df[categorical_features]

    X_num_scaled = scaler.transform(X_num)
    X = np.hstack((X_num_scaled, X_cat.values))

    # Predict probabilities and binary predictions using threshold
    probs = model.predict_proba(X)[:,1]
    preds = (probs >= threshold).astype(int)
    df['rf_pred'] = preds
    df['rf_proba'] = probs

    st.write("Prediction counts:")
    st.write(df['rf_pred'].value_counts())

    flagged_df = df[df['rf_pred'] == 1]
    st.write(f"Number of transactions flagged as fraud: {len(flagged_df)}")
    if st.checkbox("Show flagged transactions"):
        st.dataframe(flagged_df)

    # Metrics and confusion matrix if true labels exist
    if 'fraud' in df.columns:
        y_true = df['fraud']

        accuracy = accuracy_score(y_true, preds)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        st.write(f"Model Performance at Threshold = {threshold:.2f}:")
        st.write(f"- Accuracy: {accuracy:.4f}")
        st.write(f"- Precision: {precision:.4f}")
        st.write(f"- Recall: {recall:.4f}")
        st.write(f"- F1 Score: {f1:.4f}")

        cm = confusion_matrix(y_true, preds)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Fraud', 'Fraud'],
                    yticklabels=['Not Fraud', 'Fraud'], ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

    else:
        st.warning("No 'fraud' column found in dataset to calculate accuracy and other metrics.")

# Show dataset with predictions
if st.checkbox("Show dataset with predictions"):
    st.dataframe(df)

# Download button for predictions CSV
if 'rf_pred' in df.columns:
    csv = df.to_csv(index=False)
    st.download_button("Download predictions as CSV", data=csv, file_name="predictions.csv")


