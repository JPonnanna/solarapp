import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import time

from src.model import preprocess_data, train_model, test_model

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
        preprocess_data()
        st.success("Data Preprocessing Completed! Train and test datasets have been created.")

elif menu == "Train Model":
    if st.button("Train XGBoost Model"):
        train_model()
        st.success("Model Training Completed! The trained model has been saved.")

elif menu == "Test Model":
    if st.button("Test Model & Generate Predictions"):
        test_model()
        st.success("Model Testing Completed! Predictions saved in 'predicted_data.csv'.")

elif menu == "Visualizations":
    st.write("### Model Outputs & Analysis")

    if os.path.exists("./out/importance.png"):
        st.image("./out/importance.png", caption="Feature Importance", use_column_width=True)
    if os.path.exists("./out/tree.png"):
        st.image("./out/tree.png", caption="Decision Tree", use_column_width=True)
    if os.path.exists("./out/error.png"):
        st.image("./out/error.png", caption="Error Analysis", use_column_width=True)
    if os.path.exists("./out/scatterplot.png"):
        st.image("./out/scatterplot.png", caption="Actual vs Predicted", use_column_width=True)

    if os.path.exists("./out/predicted_data.csv"):
        df = pd.read_csv("./out/predicted_data.csv")
        st.write("### Predicted Data Sample")
        st.dataframe(df.head())
