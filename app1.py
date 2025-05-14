import gradio as gr
import joblib
import numpy as np

# Load your trained model
model = joblib.load("fraud_model.pkl")  # Replace with your model's filename

# Define prediction function
def predict_fraud(distance_home, distance_last, ratio_price, repeat_retailer, used_chip, used_pin, online_order):
    input_data = np.array([[distance_home, distance_last, ratio_price,
                            repeat_retailer, used_chip, used_pin, online_order]])
    prediction = model.predict(input_data)
    return "Fraud" if prediction[0] == 1 else "Not Fraud"

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_fraud,
    inputs=[
        gr.Number(label="Distance from Home"),
        gr.Number(label="Distance from Last Transaction"),
        gr.Number(label="Ratio to Median Purchase Price"),
        gr.Radio([0, 1], label="Repeat Retailer"),
        gr.Radio([0, 1], label="Used Chip"),
        gr.Radio([0, 1], label="Used PIN Number"),
        gr.Radio([0, 1], label="Online Order"),
    ],
    outputs="text",
    title="Fraud Detection Model"
)

iface.launch()
