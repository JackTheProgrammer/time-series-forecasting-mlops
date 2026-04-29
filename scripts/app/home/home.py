import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

from sys import path
path.append('.')
path.append('./')

from scripts.app.server.forecasting import gold_stock_dataloader
from scripts.app.server.server import make_forecast_request

# making chart of forecasted stock series
# using data visualizations
def plot_forecasted_series(forecasted_series):
    # The API response as per my defined schema is a flattened list but on GPU, so we need to make it available on CPU
    forecasted_numpy = forecasted_series.cpu()  # Convert to numpy and remove extra dimensions
    plt.figure(figsize=(10, 5))
    plt.plot(forecasted_numpy, label='Forecasted Stock Price')
    plt.title('Forecasted Gold Stock Price')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid()
    plt.show()

def plot_original_series():
    original_series = next(iter(gold_stock_dataloader))[0].cpu()  # Get the first batch and convert to numpy
    plt.figure(figsize=(10, 5))
    plt.plot(original_series, label='Original Stock Price')
    plt.title('Original Gold Stock Price')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid()
    plt.show()

def home():
    pass