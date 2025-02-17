import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Correlation Heatmap
def plot_correlation_heatmap(data):
    corr = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    return plt

# Time Series Plot for Generated Energy
def plot_time_series_energy(test, predicted_data):
    sns.set(font_scale=1.25, rc={"figure.figsize": (20, 10)})
    sns.set_style("whitegrid")
    sns.lineplot(data=test, x="DateTime", y="Generated", label="Actual Energy Generation")
    sns.lineplot(data=predicted_data, x="DateTime", y="Predicted", label="Predicted Energy Generation")
    plt.xlabel("Date")
    plt.ylabel("Energy Generation (Watts)")
    plt.title("Energy Generation Over Time")
    return plt

# Feature Distribution
def plot_feature_distribution(data, feature, title, xlabel, ylabel):
    sns.histplot(data[feature], kde=True, color="blue", bins=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return plt

# Model Performance Metrics (Bar Plot)
def plot_model_performance_metrics(bst, train, test):
    eval_results = bst.eval([("train", train), ("test", test)])
    metrics = [result.split(":") for result in eval_results.split("\n")]
    metric_names = [m[0] for m in metrics]
    metric_values = [float(m[1]) for m in metrics]
    sns.barplot(x=metric_names, y=metric_values)
    plt.title("Model Performance Metrics")
    plt.ylabel("Value")
    return plt

# Seasonal Patterns
def plot_seasonal_patterns(data):
    data['Month'] = data['DateTime'].dt.month
    monthly_avg = data.groupby('Month')['Generated'].mean()
    sns.barplot(x=monthly_avg.index, y=monthly_avg.values)
    plt.title("Average Monthly Energy Generation")
    plt.xlabel("Month")
    plt.ylabel("Average Energy Generation (Watts)")
    return plt

# Wind Speed vs. Energy Generation
def plot_wind_speed_vs_energy_generation(data):
    sns.scatterplot(x=data['WindSpeed10m'], y=data['Generated'])
    plt.title("Wind Speed vs Energy Generation")
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Energy Generation (Watts)")
    return plt

# Error Distribution
def plot_error_distribution(actual, pred):
    error_values = np.abs(actual - pred)
    sns.histplot(error_values, kde=True, color="green", bins=50)
    plt.title("Error Distribution")
    plt.xlabel("Error (Watts)")
    plt.ylabel("Frequency")
    return plt

# Power Consumption vs Temperature
def plot_power_consumption_vs_temperature(data):
    sns.scatterplot(x=data['AirTemp'], y=data['Generated'])
    plt.title("Power Generation vs Temperature")
    plt.xlabel("Air Temperature (°C)")
    plt.ylabel("Energy Generation (Watts)")
    return plt

# Caller function to display all visualizations
def visualize_all(data, test, predicted_data, bst, train, actual, pred):
    # Call all the plotting functions and display them using Streamlit's st.pyplot()
    st.pyplot(plot_correlation_heatmap(data))
    st.pyplot(plot_time_series_energy(test, predicted_data))
    st.pyplot(plot_feature_distribution(data, "AirTemp", "Distribution of Air Temperature", "Temperature (°C)", "Frequency"))
    st.pyplot(plot_model_performance_metrics(bst, train, test))
    st.pyplot(plot_seasonal_patterns(data))
    st.pyplot(plot_wind_speed_vs_energy_generation(data))
    st.pyplot(plot_error_distribution(actual, pred))
    st.pyplot(plot_power_consumption_vs_temperature(data))
