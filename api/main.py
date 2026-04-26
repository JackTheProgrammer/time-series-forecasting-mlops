# making a REST API based exposure
# of the stock forecasting best model
# we mmake usage of FastAPI because its easy
# scale our code further

from fastapi.applications import FastAPI
from sys import path
from torch import tensor
import torch
from uvicorn import run

# to ensure we can import from the src directory
# regardless of where this script is run from
path.append('.')
path.append('./')

from src.ingestion import forecast_next_price
from api.schemas import ForecastRequest, ForecastResponse

app = FastAPI()

@app.post('/forecast', response_model=ForecastResponse, summary="Get the forecasted stock price for the next time step")
def forecast(request: ForecastRequest):
    series_tensor = tensor(request.series_frame, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))  # Convert the list to a PyTorch tensor
    print(f"Received series frame for forecasting: {series_tensor} with shape {series_tensor.shape}")
    forecasted_price = forecast_next_price(series_tensor)
    return {"forecasted_price": forecasted_price}

@app.get('/')
def root():
    return {"message": "Welcome to the stock forecasting API. Use the /forecast endpoint to get predictions."}

@app.get('/health')
def health():
    return {"status": "API is healthy and running."}

if __name__ == "__main__":
    run(app=app, port=5050)