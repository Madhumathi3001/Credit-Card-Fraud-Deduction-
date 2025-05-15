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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Set Streamlit page config
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# Title
st.title("🔍 Credit Card Fraud Detection Dashboard")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Predict Fraud", "Model Performance", "Data Visualization"])

# Load model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load prediction dataset
df = pd.read_csv("card_transdata.csv")  # replace with correct path if needed

# Predict on loaded dataset
X_columns = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price',
             'repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']
X = df[X_columns]
X_scaled = scaler.transform(X)
y_true = df['fraud']
y_pred = model.predict(X_scaled)

# --- Section 1: Fraud Prediction Interface ---
if section == "Predict Fraud":
    st.header("💳 Enter Transaction Details")

    distance_home = st.number_input("Distance from Home", min_value=0.0)
    distance_last = st.number_input("Distance from Last Transaction", min_value=0.0)
    ratio_price = st.number_input("Ratio to Median Purchase Price", min_value=0.0)
    repeat_retailer = st.radio("Repeat Retailer", [0, 1])
    used_chip = st.radio("Used Chip", [0, 1])
    used_pin = st.radio("Used PIN Number", [0, 1])
    online_order = st.radio("Online Order", [0, 1])

    if st.button("Predict"):
        input_numeric = np.array([[distance_home, distance_last, ratio_price]])
        scaled_input = scaler.transform(input_numeric)
        final_input = np.hstack((scaled_input, [[repeat_retailer, used_chip, used_pin, online_order]]))

        prediction = model.predict(final_input)
        result = "Fraud ❗" if prediction[0] == 1 else "Not Fraud ✅"
        st.success(f"Prediction: {result}")

# --- Section 2: Model Performance ---
elif section == "Model Performance":
    st.header("📊 Model Evaluation Metrics")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Fraud", "Fraud"],
                yticklabels=["Not Fraud", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    acc = accuracy_score(y_true, y_pred)
    st.metric("Model Accuracy", f"{acc:.4f}")

    st.subheader("Accuracy Comparison")
    acc_data = pd.DataFrame({
        "Model": ["Random Forest", "Logistic Regression", "KNN", "SVM", "Naive Bayes"],
        "Accuracy": [acc, 0.97, 0.94, 0.96, 0.91]  # placeholder values for demo
    })

    fig2, ax2 = plt.subplots()
    sns.barplot(data=acc_data, x="Model", y="Accuracy", palette="viridis")
    ax2.set_ylim(0.85, 1.00)
    st.pyplot(fig2)

# --- Section 3: Data Visualization ---
elif section == "Data Visualization":
    st.header("📈 Data Trend Analysis")

    st.subheader("Fraud vs Non-Fraud Distribution")
    fraud_counts = df['fraud'].value_counts()
    st.bar_chart(fraud_counts)

    st.subheader("Pie Chart of Fraud Ratio")
    fig3, ax3 = plt.subplots()
    ax3.pie(fraud_counts, labels=['Not Fraud', 'Fraud'], autopct='%1.1f%%', colors=["green", "red"], startangle=90)
    ax3.axis("equal")
    st.pyplot(fig3)

    st.subheader("Feature Trend by Fraud")
    selected_feature = st.selectbox("Select a feature to view trends", X_columns)
    fig4, ax4 = plt.subplots()
    sns.boxplot(data=df, x="fraud", y=selected_feature, palette="Set2")
    ax4.set_xticklabels(["Not Fraud", "Fraud"])
    st.pyplot(fig4)

    st.subheader("Correlation Heatmap")
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[X_columns + ['fraud']].corr(), annot=True, cmap="coolwarm", ax=ax5)
    st.pyplot(fig5)
