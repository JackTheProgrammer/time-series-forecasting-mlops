# MLOps based gold stock forecasting

## Introduction

## Methodology

## Architecture

## Features

## 💻 Demo screenshot

![Alt Screenshot of the dockerized web app running](demos/streamlit_app_running_screenshot.png)

## 📂 Project structure

```text
time-series-forecasting-mlops/
├── .dockerignore
├── .dvc/
│   └── .gitignore
├── .dvcignore
├── .github/
│   └── workflows/
│       └── actions.yaml
├── .gitignore
├── compose.yml
├── data.dvc
├── demos/
│   └── streamlit_app_running_screenshot.png
├── Dockerfile
├── forecasts.Dockerfile
├── forecasts.Dockerfile.dockerignore
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── LICENSE
├── models.dvc
├── notebooks/
│   └── timeseries_modelling.ipynb
├── README.md
├── requirements.txt
├── scaled_transform.dvc
├── scripts/
│   ├── __init__.py
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   ├── app/
│   │   ├── home/
│   │   │   ├── daily_forecasts.py
│   │   │   └── home.py
│   │   └── server/
│   │       ├── forecasting.py
│   │       ├── latest_model_forecasting.py
│   │       └── server.py
│   ├── architectures/
│   │   ├── conv1d.py
│   │   ├── gru.py
│   │   └── lstm.py
│   ├── dl_pipeline.py
│   ├── evaluate.py
│   ├── ingestion.py
│   └── preprocessing.py
└── winner_models.dvc
```

## Installation Guide

### Method 1: Using git clone and pip install

```bash
git clone https://github.com/JackTheProgrammer/time-series-forecasting-mlops.git
cd time-series-forecasting-mlops
pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.6.0
pip install --no-cache-dir -r requirements.txt
```

### Method 2: Using docker image

#### Pulling the image which is runs the entire architectual flow

```bash
docker pull fawadawan143/gold_stock_prices_predictions_mlops:latest
docker run -p 8501:8501 5050:5050 fawadawan143/gold_stock_prices_predictions_mlops:latest
```

#### Pulling the image which is only for forecasting using the latest model

```bash
docker pull fawadawan143/daily-forecasting:latest
docker run -p 8501:8501 5050:5050 fawadawan143/daily-forecasting:latest
```

## Resources
