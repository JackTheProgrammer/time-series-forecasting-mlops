# MLOps based gold stock forecasting

## 📖 Introduction

This project focuses on forecasting gold stock prices using deep learning models and implementing MLOps practices for efficient model development, deployment, and monitoring. The main objective is to create a robust and scalable system that can provide accurate forecasts of gold stock prices while ensuring the maintainability and reliability of the deployed models.

## 🔧 Tools

- [Docker](https://www.docker.com/)
- [Kubernetes](https://kubernetes.io/)
- [DVC](https://dvc.org/)
- [GCP](https://cloud.google.com/)

## 🔬 Methodology

My approach included:

1. **Data Ingestion and Preprocessing**:
I acquired the gold stock from 2010-01-01 to every current date using yfinance library, and then I performed preprocessing steps such as handling missing values, scaling the data, and creating sequences for time series forecasting.

2. **Model Development and Training**:
I developed and trained multiple deep learning models, including LSTM, GRU, and Conv1D architectures, to forecast the gold stock prices. I used a systematic approach to train and evaluate each model, ensuring that I selected the best-performing model based on evaluation metrics.

3. **Model Evaluation and Selection**:
I evaluated the performance of each model using appropriate metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). Based on the evaluation results, I selected the best-performing model for deployment.

4. **Deployment and Monitoring**:
I deployed the selected model using FastAPI and Streamlit to create a user-friendly interface for forecasting. I also implemented monitoring mechanisms to track the performance of the deployed model and ensure that it continues to provide accurate forecasts over time. Then, I at first deployed them all to DVC based remote of the google drive, and then I containerized the entire architectural flow using Docker, and finally, I deployed the forecasting app to Kubernetes to simulate a cloud environment deployment.

## ⚙️ Architecture

### Overall architecture

![Alt Overall architectural system flow](diagrams/overall_generic_diag.jpg)

### CI/CD workflow architecture

![Alt CI/CD workflow architecture](diagrams/mlops_ci_cd_workflow.jpg)

## 🔄 Synching to local machine (Windows 🪟 OS)

### Method 1: Using command line

Search for the CMD, and then run the following command to execute the task immediately:

```bat
schtasks /run /tn "MLOps_TimeSeries_Sync"
```

### Method 2: Using scheduler GUI

1. Press `Win + R` to open the Run dialog, type `taskschd.msc`, and press Enter to open the Task Scheduler.
2. In the Task Scheduler, navigate to the "Task Scheduler Library" on the left pane
3. Look for the task named "MLOps_TimeSeries_Sync" in the middle pane.
4. Right-click on the "MLOps_TimeSeries_Sync" task and select "Run" from the context menu to execute the task immediately.

### Task deployment demos

To deploy the task, you can follow these steps:

i. Open the CMD as admin and type the following into it to create the task as per the cron trigger of `0 17 1 */4 *`

```bat
schtasks /create /tn "MLOps_TimeSeries_Sync" /tr "'D:\path\to\your\sync_pipeline.bat'" /sc monthly /mo 4 /d 1 /st 17:00
```

ii. On my machine, I have the `sync_pipeline.bat` file located at `D:\MLOps_TimeSeries_Sync\sync_pipeline.bat`, so the command I ran was:

![Alt CMD command](demos/successfull_running_local_synching_cron_peshawar_pkt.png)

iii. Also, in my task scheduler, the task looked like this:

![Alt Task in task scheduler](demos/schtask_mlops_time_series_synch.png)

iv. If you look at my [Synching pipeline](sync_pipeline.bat) bat script, you'll see that it has `cd "D:\ml projects\mlops_time_series_modeling"`, why? Because, I have my entire project there, in your local setup, you can have it as per your this project's location on your local machine, just make sure to update the path in the `sync_pipeline.bat` file accordingly.

## 🚀 Features

- **Data Ingestion**: Automated data ingestion from yfinance to acquire gold stock prices.
- **Preprocessing**: Handling missing values, scaling data, and creating sequences for time series
- **Monthly model training and evaluation**: Systematic training and evaluation of multiple deep learning models (LSTM, GRU, Conv1D) to select the best-performing model done every 4 months of deployment.
- **Deployment**: Deployment of the best-performing model using FastAPI and Streamlit for user-friendly forecasting.

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
│   ├── schtask_mlops_time_series_synch.png
│   ├── streamlit_app_running_screenshot.png
│   └── successfull_running_local_synching_cron_peshawar_pkt.png
├── diagrams/
│   ├── mlops_ci_cd_workflow.jpg
│   └── overall_generic_diag.jpg
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
├── sync_pipeline.bat
└── winner_models.dvc
```

## 🛠️ Installation Guide

### Method 1: Using git clone and pip install

```bash
git clone https://github.com/JackTheProgrammer/time-series-forecasting-mlops.git
cd time-series-forecasting-mlops
pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.6.0
pip install --no-cache-dir -r requirements.txt
python scripts/api/main.py && streamlit run scripts/app/home/home.py
```

### Method 2: Using docker image

#### Pulling the image which runs the entire architectual flow

```bash
docker pull fawadawan143/fawadawan143/gold_stock_prediction:latest
docker run -p 8501:8501 5050:5050 fawadawan143/gold_stock_prediction:latest
```

#### Pulling the image which is only for forecasting using the latest model

```bash
docker pull fawadawan143/daily-forecasting:latest
docker run -p 8501:8501 5050:5050 fawadawan143/daily-forecasting:latest
```

### Method 3: Using docker compose

```bash
git clone https://github.com/JackTheProgrammer/time-series-forecasting-mlops.git
cd time-series-forecasting-mlops
docker compose up --build -d
docker compose up -d ml-pipeline # This will start the entire architectural flow, including data ingestion, preprocessing, model training, and evaluation. The ml-pipeline service will run all the necessary steps to process the data, train the models, and evaluate their performance. You can monitor the logs of this service to see the progress of each step in the pipeline.
docker compose up -d forecasting-app # This will start the forecasting app service, which will run the forecasting app with the latest DVC and trsining artifacts.
```

To stop the services, you can use the following command:

```bash
docker compose down
```

### Method 4: Using kubernetes

```bash
git clone https://github.com/JackTheProgrammer/time-series-forecasting-mlops.git
cd time-series-forecasting-mlops
docker pull fawadawan143/daily-forecasting:latest # You can build your own as well, using the forecasts.Dockerfile and forecasts.Dockerfile.dockerignore files in the root directory of the project, and then tag it as daily-forecasting:latest, but my image is already available on Docker Hub, so no need to re-invent the wheel.
docker run -p 8501:8501 5050:5050 daily-forecasting:latest
minikube start --driver=docker # Start minikube with the Docker driver
kubectl apply -f k8s/ # Apply the Kubernetes deployment and service configurations in the k8s/ directory on the minikube cluster
minikube image load daily-forecasting:latest # Load the Docker image into minikube
# In an another command line tool window, do:
minikube tunnel # Start the minikube tunnel to access the LoadBalancer service, this simulates the cloud environment where LoadBalancer services are commonly used to expose applications to the internet. The tunnel will allow you to access the service using the external IP address assigned by minikube. once the tunnel is running, you can access the forecasting API at http://<minikube_ip>:80/forecast or http://<minikube_ip>:80/latest-forecast depending on the endpoint you want to use.
# Do it in another CLI window, because the tunnel command will keep running and will not return to the command prompt until you stop it.
kubectl expose deployment gold-forecasting-deployment --type=LoadBalancer --port=80 --target-port=80
```

Now to end it all up after being done testing, do:

```bash
kubectl delete service gold-forecasting-deployment
kubectl delete deployment gold-forecasting-deployment
minikube stop
```

## 🔗 Resources

- [**My own GitHub repo: Time-Series-Forecasting-and-Analysis**](https://github.com/JackTheProgrammer/Time-Series-Forecasting-and-Analysis)
- [Docker for Machine Learning | Docker Crash Course | CampusX](https://youtu.be/GToyQTGDOS4?si=Lkb3lNbxzPKwPf2I)
- [Docker Simply Explained with a Machine Learning Project for Beginners](https://youtu.be/-l7YocEQtA0?si=GQ425cnC0SJnHLG9)
- [Data Version Control | DVC | How to Push Datasets to Google Drive Easily | ‪@CodeKamikaze‬ (4)](https://youtu.be/e3GuonR1r-0?si=1f7f4dniQwH-6uCi)
- [Automating Data Pipelines with Python & GitHub Actions [Code Walkthrough]](https://youtu.be/wJ794jLP2Tw?si=mXsWGGRQGxmI1rfl)
- [Deploying Machine Learning Models with Docker and Kubernetes](https://medium.com/@rahulholla1/deploying-machine-learning-models-with-docker-and-kubernetes-e267543cf5aa)
- [How to Set Up a CI/CD Pipeline with GitHub Actions for Automated Deployments](https://dev.to/vishnusatheesh/how-to-set-up-a-cicd-pipeline-with-github-actions-for-automated-deployments-j39)
