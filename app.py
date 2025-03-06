import streamlit as st
import os
os.system('pip install joblib')
import joblib
import pandas as pd

# Load trained models
lr_model = joblib.load("fuel_efficiency_lr.pkl")
dt_model = joblib.load("fuel_efficiency_dt.pkl")

# Streamlit App Title
st.title("Fuel Efficiency Prediction Dashboard")

# Sidebar for Model Selection
st.sidebar.header("Select Model")
model_choice = st.sidebar.selectbox("Choose a Model", ["Linear Regression", "Decision Tree"])

# User Inputs
st.subheader("Enter the vehicle details:")

qty_taken = st.number_input("Quantity Taken (Liters)", min_value=0.0, step=0.1)
odometer = st.number_input("Odometer Reading (KM)", min_value=0.0, step=1.0)
price = st.number_input("Fuel Price per Liter", min_value=0.0, step=0.01)

# Prediction Button
if st.button("Predict Fuel Efficiency"):
    # Create input DataFrame
    input_data = pd.DataFrame([[qty_taken, odometer, price]], columns=["QtyTaken", "odometer", "Price"])

    # Choose Model
    if model_choice == "Linear Regression":
        prediction = lr_model.predict(input_data)[0]
    else:
        prediction = dt_model.predict(input_data)[0]

    # Display Result
    st.success(f"Predicted Fuel Efficiency: {prediction:.2f} km per liter")
