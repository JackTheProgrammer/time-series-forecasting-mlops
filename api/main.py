# making a REST API based exposure
# of the stock forecasting best model
# we mmake usage of FastAPI because its easy
# scale our code further

from fastapi.applications import FastAPI
from sys import path
from uvicorn import run

# to ensure we can import from the src directory
# regardless of where this script is run from
path.append('.')
path.append('./')

from src.ingestion import forecast_next_price
from api.schemas import ForecastRequest, ForecastResponse

app = FastAPI()

@app.post('/forecast', response_model=ForecastResponse)
def forecast(request:ForecastRequest):
    forecasted_price = forecast_next_price(request.series_frame)
    return ForecastResponse(forecasted_price=forecasted_price)

@app.get('/')
def root():
    return {"message": "Welcome to the stock forecasting API. Use the /forecast endpoint to get predictions."}

@app.get('/health')
def health():
    return {"status": "API is healthy and running."}

if __name__ == "__main__":
    run(app=app, port=5050)