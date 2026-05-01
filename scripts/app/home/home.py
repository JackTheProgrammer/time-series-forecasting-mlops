import streamlit as st
import matplotlib.pyplot as plt
# import subprocess, time
from datetime import datetime
import numpy as np

from sys import path
path.append('.')
path.append('./')

from scripts.app.server.forecasting import (
    gold_stock_dataloader,
    todays_gold_stock_price,
    scaler
)
from scripts.app.server.server import make_forecast_request

# def inverse_min_max(scaled_val:np.ndarray, min_val=1500, max_val=5500):
#     """Manually reverses Min-Max scaling."""
#     return scaled_val * (max_val - min_val) + min_val

# making charts of original + forecasted stock series
# using data visualizations
def plot_forecasted_series(forecasted_series):
    forecasted_numpy = np.array(forecasted_series)
    forecasted_reshaped = forecasted_numpy.reshape(-1)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(forecasted_reshaped, label='Forecasted Stock Price', color='orange')
    ax.set_title('Forecasted Gold Stock Price')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Stock Price')
    ax.legend()
    ax.grid(True)
    return fig # Return the figure object

# def plot_original_series():
#     original_series = todays_gold_stock_price.values 
#     original_reshaped = np.array(original_series).reshape(-1)
    
#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.plot(original_reshaped, label='Original Stock Price', color='blue')
#     ax.set_title('Original Gold Stock Price')
#     ax.set_xlabel('Time Steps')
#     ax.set_ylabel('Stock Price')
#     ax.legend()
#     ax.grid(True)
#     return fig # Return the figure object

def home():
    st.title(f"Gold Stock Price Forecasting {datetime.today().strftime('%Y-%m-%d')}")
    st.write("This application forecasts future gold stock prices based on historical data.")

    # Plot original series
    st.subheader("Original Gold Stock Price Series")
    fig, ax = plt.subplots()
    todays_gold_stock_price[['Close']].plot(kind='line', ax=ax)

    # Render in Streamlit
    st.pyplot(fig)

    # Make forecast request
    st.subheader(f"Forecasted Gold Stock Price Series {datetime.today().strftime('%Y-%m-%d')}")
    
    # Get last 30 values
    # input_data = gold_stock_dataloader.dataset.stock_prices['Close'].values[-30:].tolist()
    
    # forecasted_series = make_forecast_request({"forecast_series": input_data})
    
    forecasted_values = []
    for input_batch in gold_stock_dataloader:
        # FIX 1: Convert Tensor to list so it can be sent as JSON
        # If input_batch is a tensor, use .tolist()
        # If it's a list containing [features, targets], use input_batch[0].tolist()
        batch_data = input_batch[0].tolist() if isinstance(input_batch, list) else input_batch.tolist()

        forecast_response = make_forecast_request({
            "forecast_series": batch_data
        })
        
        # FIX 2: Extract the actual numbers from the dictionary
        # Based on your logs, the key is 'forecasted_price'
        if forecast_response and 'forecasted_price' in forecast_response:
            # 1. Convert to numpy array and reshape to (N, 1) for sklearn
            vals_2d = np.array(forecast_response['forecasted_price']).reshape(-1, 1)
            
            # 2. Use the exact scaler from forecasting.py to reverse the math
            actual_prices_2d = scaler.inverse_transform(vals_2d)
            
            # 3. Flatten back to a simple 1D list for matplotlib
            actual_prices = actual_prices_2d.flatten().tolist()
            forecasted_values.extend(actual_prices)
    
    if len(forecasted_values) > 0:
        # Pass the clean list of numbers to the plotting function
        fig_forecast = plot_forecasted_series(forecasted_values)
        st.pyplot(fig_forecast)
    else:
        st.error("Failed to retrieve forecasted series or data was empty.")
        
if __name__ == "__main__":
    home()