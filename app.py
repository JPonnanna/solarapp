import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import time

from model import preprocess_data, train_model, test_model

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

st.title("Solar Energy Prediction with XGBoost")

# Sidebar navigation
menu = st.sidebar.selectbox("Select an option", [
    "Home", "Preprocess Data", "Train Model", "Test Model", "Visualizations"
])

if menu == "Home":
    st.write("### Welcome to the Solar Energy Prediction App")
    st.write("This app uses XGBoost to predict solar energy generation based on weather parameters.")
    st.write("Select an option from the sidebar to proceed.")

elif menu == "Preprocess Data":
    if st.button("Run Preprocessing"):
        st.write("This step prepares raw data by loading two CSV files (solar and weather data). It processes timestamps to ensure they align and converts them to a consistent time zone. The function merges both datasets, cleans missing values, and shuffles the data. It then splits the data into training and testing sets, saving them to CSV files for further use in modeling.")
        preprocess_data()
        st.success("Data Preprocessing Completed! Train and test datasets have been created.")
        test_data = pd.read_csv("./out/test.csv")
        train_data = pd.read_csv("./out/train.csv")
        st.write("### Training Data")
        st.write(train_data.head())
        st.write("### Test Data")
        st.write(test_data.head())
        

elif menu == "Train Model":
    if st.button("Train XGBoost Model"):
        train_model()
        st.success("Model Training Completed! The trained model has been saved.")

elif menu == "Test Model":
    if st.button("Test Model & Generate Predictions"):
        test_model()
        st.success("Model Testing Completed! Predictions saved in 'predicted_data.csv'.")
        pred_data = pd.read_csv("./out/predicted_data.csv")

elif menu == "Outputs":
    st.write("### Model Outputs & Analysis")

    if os.path.exists("./out/importance.png"):
        st.image("./out/importance.png", caption="Feature Importance", use_container_width=True)
    if os.path.exists("./out/tree.png"):
        st.image("./out/tree.png", caption="Decision Tree", use_container_width=True)
    if os.path.exists("./out/error.png"):
        st.image("./out/error.png", caption="Error Analysis", use_container_width=True)
    if os.path.exists("./out/scatterplot.png"):
        st.image("./out/scatterplot.png", caption="Actual vs Predicted", use_container_width=True)

    if os.path.exists("./out/predicted_data.csv"):
        df = pd.read_csv("./out/predicted_data.csv")
        st.write("### Predicted Data Sample")
        st.dataframe(df.head())

elif menu == "Visualizations":
    st.write("### Model Outputs & Analysis")

    # Ensure the predicted data is available for visualization
    if os.path.exists("./out/predicted_data.csv"):
        pred_data = pd.read_csv("./out/predicted_data.csv")
        st.write("### Predicted Data Sample")
        st.dataframe(pred_data.head())

    # Ensure the train and test data are available for visualization
    if os.path.exists("./out/train.csv") and os.path.exists("./out/test.csv"):
        train_data = pd.read_csv("./out/train.csv")
        test_data = pd.read_csv("./out/test.csv")
        
        # Call the visualization function with necessary data
        visualize_all(train_data, test_data, pred_data)  