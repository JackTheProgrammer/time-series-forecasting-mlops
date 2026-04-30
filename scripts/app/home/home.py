import streamlit as st
import matplotlib.pyplot as plt
# import subprocess, time
import numpy as np

from sys import path, executable
path.append('.')
path.append('./')

from scripts.app.server.forecasting import (
    gold_stock_dataloader,
    todays_gold_stock_price
)
from scripts.app.server.server import make_forecast_request

# python command to run the FastAPI server: `uvicorn scripts.api.main:app --reload`
# os.system('python scripts/api/main.py')  # Start the FastAPI server in the background

# if 'server_started' not in st.session_state:
#     # We call uvicorn as a module (-m) to handle paths correctly
#     st.session_state.server_started = subprocess.Popen(
#         [executable, "-m", "uvicorn", "scripts.api.main:app", "--host", "127.0.0.1", "--port", "5050"],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE
#     )
#     st.session_state.api_started = True
#     st.info("Starting FastAPI server...")
#     time.sleep(3)  # FastAPI/Uvicorn needs a few seconds to initialize
#     st.success("API is running!")

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

def plot_original_series():
    original_series = todays_gold_stock_price.values 
    original_reshaped = np.array(original_series).reshape(-1)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(original_reshaped, label='Original Stock Price', color='blue')
    ax.set_title('Original Gold Stock Price')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Stock Price')
    ax.legend()
    ax.grid(True)
    return fig # Return the figure object

def home():
    st.title("Gold Stock Price Forecasting")
    st.write("This application forecasts future gold stock prices based on historical data.")

    # Plot original series
    st.subheader("Original Gold Stock Price Series")
    fig_orig = plot_original_series()
    st.pyplot(fig_orig) # FIX 3: Explicitly call st.pyplot

    # Make forecast request
    st.subheader("Forecasted Gold Stock Price Series")
    
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
            # forecast_response['forecasted_price'] looks like [[val], [val], ...]
            # We flatten it to just [val, val, ...]
            vals = np.array(forecast_response['forecasted_price']).flatten().tolist()
            forecasted_values.extend(vals)
    
    if len(forecasted_values) > 0:
        # Pass the clean list of numbers to the plotting function
        fig_forecast = plot_forecasted_series(forecasted_values)
        st.pyplot(fig_forecast)
    else:
        st.error("Failed to retrieve forecasted series or data was empty.")
        
if __name__ == "__main__":
    home()