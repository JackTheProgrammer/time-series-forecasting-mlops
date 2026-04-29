import streamlit as st
import matplotlib.pyplot as plt
# import plotly.express as px
import os
import numpy as np

from sys import path
path.append('.')
path.append('./')

from scripts.app.server.forecasting import gold_stock_dataloader, todays_gold_stock_price
from scripts.app.server.server import make_forecast_request

# python command to run the FastAPI server: `uvicorn scripts.api.main:app --reload`
os.system('python scripts/api/main.py')  # Start the FastAPI server in the background

# making charts of original + forecasted stock series
# using data visualizations
def plot_forecasted_series(forecasted_series):
    forecasted_numpy = np.array(forecasted_series)  # Convert the list of lists to a NumPy array
    forecasted_reshaped = forecasted_numpy.reshape(-1)  # Reshape to a 1D array for plotting
    plt.figure(figsize=(10, 5))
    plt.plot(forecasted_reshaped, label='Forecasted Stock Price')
    plt.title('Forecasted Gold Stock Price')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid()
    plt.show()

def plot_original_series():
    original_series = todays_gold_stock_price.values  # Get the current gold stock price series of today!!
    original_reshaped = np.array(original_series).reshape(-1)  # Reshape to a 1D array for plotting
    plt.figure(figsize=(10, 5))
    plt.plot(original_reshaped, label='Original Stock Price')
    plt.title('Original Gold Stock Price')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid()
    plt.show()

def home():
    st.title("Gold Stock Price Forecasting")
    st.write("This application forecasts the future gold stock prices based on historical data.")

    # Plot original series
    st.subheader("Original Gold Stock Price Series")
    plot_original_series()

    # Make forecast request and plot forecasted series
    st.subheader("Forecasted Gold Stock Price Series")
    # forecast_series = np.random.rand(32, 30, 1).tolist()  # Example input data
    forecasted_series = make_forecast_request({
        "forecast_series": gold_stock_dataloader.dataset.stock_prices['Close'].values[-30:].tolist()  # Use the last 30 values of the original series as input for forecasting
    })
    
    if forecasted_series is not None:
        plot_forecasted_series(forecasted_series)
    else:
        st.error("Failed to retrieve forecasted series from the server.")
        
if __name__ == "__main__":
    home()