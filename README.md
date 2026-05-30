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

## Resources
