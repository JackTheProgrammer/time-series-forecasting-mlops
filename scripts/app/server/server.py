import requests
from sys import path
from logging import log

import torch

path.append('.')
path.append('./')

api_url = 'http://localhost:5050'

def is_up_and_running():
    try:
        response = requests.get(api_url + '/health')
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException as e:
        log(msg = f'ERROR: {str(e)}', level=40)
    return False

def make_forecast_request(data):
    try:
        if is_up_and_running():
            response = requests.post(api_url + '/forecast', json=data)
            if response.status_code == 200:
                return response.json()
            else:
                log(msg = f'ERROR: Received status code {response.status_code}', level=40)
        else:
            log(msg = 'ERROR: Server is not up and running', level=40)
    except requests.exceptions.RequestException as e:
        log(msg = f'ERROR: {str(e)}', level=40)
    return None

torch.manual_seed(42)  # For reproducibility
# shape is such that 32 is the batch size, 30 is the sequence length 
# and 1 is the number of features (e.g., stock price)
forecast_series = torch.rand((32, 30, 1)).tolist()  # Example input data

forecasted_series = make_forecast_request({
    "forecast_series": forecast_series
})

print("Forecasted series:", forecasted_series)