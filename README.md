 Credit Card Fraud Detection Using Machine Learning Models

This project builds a machine learning model to predict whether a transaction is fraudulent or valid. The model is trained using a dataset downloaded from Kaggle and deployed using Streamlit.

🔍 Overview

- ✅ Performed Exploratory Data Analysis (EDA)
- 📊 Identified and standardized skewed numerical features
- 🧠 Trained various ML models; selected **Random Forest** for its accuracy
- 🚀 Deployed the model using **Streamlit** for interactive use

 📁 Dataset

- Source: [Kaggle – Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud)
- Features include:
  - `distance_from_home`
  - `distance_from_last_transaction`
  - `ratio_to_median_purchase_price`
  - `repeat_retailer`
  - `used_chip`
  - `used_pin_number`
  - `online_order`
  - `fraud` (target variable)

 ⚙️ Installation

1. Clone the repository and install the required packages:

   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   pip install -r requirements.txt

📊 EDA & Data Preprocessing

  - Performed basic EDA: checked data types, dataset shape, null values, and value counts
  - Analyzed the distribution of binary variables with respect to the target variable
  - Detected skewness in 3 numeric variables and applied standardization
  - The dataset was imbalanced; performed undersampling to handle class imbalance

🧠 Model Selection

  - Tried multiple machine learning models: Logistic Regression, Decision Tree, KNN
  - Random Forest Classifier performed best in terms of accuracy and precision
  - Final model deployed using Streamlit







