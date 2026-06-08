# 🪙 Automated MLOps Gold Price Forecasting Pipeline

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![DVC](https://img.shields.io/badge/DVC-Data_Version_Control-9CF.svg?style=flat-square&logo=data-version-control)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?style=flat-square&logo=docker)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Orchestrated-326CE5.svg?style=flat-square&logo=kubernetes)](https://kubernetes.io/)
[![GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub_Actions-2088FF.svg?style=flat-square&logo=github-actions)](https://github.com/features/actions)

An enterprise-grade, fully automated end-to-end MLOps pipeline for time-series forecasting. This system abstracts heavy data and deep learning model weights using content-addressable storage (DVC), orchestrates decentralized compute layers via GitHub Actions, and handles immutable production deliveries using Docker and Kubernetes.

---

## 📖 Introduction

This project implements a self-correcting, scheduled deep learning retraining mechanism to forecast gold stock prices. By treating data, preprocessing parameters, and neural network weights as versioned assets distinct from code, the system mitigates data drift, handles continuous delivery, and provides a scalable runtime environment with zero human intervention.

## 🛠️ Tech Stack & Ecosystem

* **Core Engine:** Python 3.13, PyTorch 2.6 (CPU-optimized compilation)
* **Data Tracking:** Data Version Control (DVC) backed by Remote Cloud Cache Storage
* **CI/CD Orchestration:** GitHub Actions Engine (Decoupled Runner Architectures)
* **Container Delivery:** Docker, Multi-stage Buildx, Docker Hub Registry
* **Production Cluster:** Kubernetes (Minikube Cluster Simulation, LoadBalancer Ingress)
* **Presentation Layer:** FastAPI (Backend Prediction Engine) & Streamlit (Frontend Client UI)

---

## 🔬 Methodology & System Operations

1. **Automated Ingestion & Preprocessing (`ingestion.py`, `preprocessing.py`)**
   * Dynamically tracks real-time historical market data from 2010 to the present moment using `yfinance`.
   * Standardizes raw values, manages historical sequence lengths, and packages scaling parameters into isolated data serialization formats (`scaled_transform.pkl`).

2. **Decoupled Model Development (`dl_pipeline.py`, `architectures/`)**
   * Parallel training of multiple deep learning architectures: Recurrent neural paths (**LSTM**, **GRU**) and convolutional temporal matchers (**Conv1D**).
   * Models are benchmarked via automated validation scripts (`evaluate.py`) tracking Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to dynamically flag the absolute "Winner Model."

3. **Continuous Deployment & Storage Decoupling**
   * Heavy binaries are offloaded to an abstract content-addressable storage remote via an automated cloud credential handshake.
   * Runtime images are compiled with strict metadata layers and deployed into simulated container orchestration layers for target scale verification.

---

## ⚙️ Architecture Blueprints

### Overall System & Data Flow

The system keeps the application runtime lightweight by cleanly separating code logic (tracked in Git) from heavy binaries (managed by DVC).

![Overall System Flow](diagrams/overall_generic_diag.jpg)

### CI/CD Scheduled Orchestration

The retraining pipeline operates on an independent serverless virtual machine context, checking out code pointers, parsing delta transformations, writing back metadata hashes, and delivering production-ready containers.

![CI/CD Workflow Architecture](diagrams/mlops_ci_cd_workflow.jpg)

---

## 🔄 Local Workstation Synchronization (Windows OS 🪟)

To sync your local environment with the cloud infrastructure context, a dedicated batch file (`sync_pipeline.bat`) executes safe upstream rebase pulls and hydrates your local data cache.

### Method 1: Execution via Command Line

Open a standard terminal and trigger the Task Scheduler engine immediately:

```bat
schtasks /run /tn "MLOps_TimeSeries_Sync"
```

### Method 2: Execution via Task Scheduler GUI

1. Press `Win + R`, type `taskschd.msc`, and press **Enter**.
2. Navigate to the **Task Scheduler Library** in the left sidebar directory.
3. Locate **`MLOps_TimeSeries_Sync`** in the middle console pane.
4. Right-click the task entry and select **Run**.

### Task Orchestration & Deployment Setup

1. Open **Command Prompt as Administrator** and map the script execution to run precisely at 17:00 (5:00 PM PKT) on Day 1 of every 4th month to stay perfectly aligned with the cloud cron architecture (`0 17 1 */4 *`):

```bat
schtasks /create /tn "MLOps_TimeSeries_Sync" /tr "'D:\path\to\your\sync_pipeline.bat'" /sc monthly /mo 4 /d 1 /st 17:00
```

1. **Workstation Deployment References:**

*Local Command Prompt initialization verification.*

*Active operational task mapped inside the native Windows Task Scheduler environment.*

> ⚠️ **Configuration Note:** The local [Synchronization Script](https://www.google.com/search?q=sync_pipeline.bat) contains hardcoded routing parameters pointing to `cd "D:\ml projects\mlops_time_series_modeling"`. Ensure you update this path to reflect the absolute workspace path on your local system before deploying the scheduler task.

---

## 🚀 Key Automation Features

* **Data Drift Safeguards:** Automated periodic cron execution fetches real-time updates and retrains models down to the latest decimal without manual workflow initialization.
* **Smart Delta Synchronization:** The DVC delivery layer evaluates system state hashes at the block level; if retraining yields identical artifacts, network serialization uploads are safely skipped (`2 files pushed`).
* **Self-Healing Git Loops:** The CI/CD runner checks changes using `git status --porcelain`. If tracking hashes diverge, it signs, updates, commits, and pushes pointer metadata back to `main` while injecting `[skip ci]` flags to completely block recursive execution loops.

---

## 💻 Application Client Interface

![Alt demo](demos/streamlit_app_running_screenshot.png)

---

## 📂 Project Directory Structure

```text
time-series-forecasting-mlops/
├── .dockerignore
├── .dvc/
│   └── .gitignore
├── .dvcignore
├── .github/
│   └── workflows/
│       └── actions.yaml          # Scheduled Retraining & CD Infrastructure Blueprint
├── .gitignore
├── compose.yml                   # Multi-Container Application Stack Topology
├── data.dvc                      # DVC Data Vector Tracking Pointer
├── demos/                        # Visual Execution Logs & Demos
├── diagrams/                     # System Infrastructure Flow Charts
├── Dockerfile                    # Multi-Stage Complete Architectural Image Build
├── forecasts.Dockerfile          # Lightweight Production Target Forecasting Runtime
├── forecasts.Dockerfile.dockerignore
├── k8s/                          # Production Deployment Declarative Contexts
│   ├── deployment.yaml
│   └── service.yaml
├── LICENSE
├── models.dvc                    # DVC Trained Weights Tracking Pointer
├── notebooks/
│   └── timeseries_modelling.ipynb
├── README.md
├── requirements.txt
├── scaled_transform.dvc          # DVC Data Processing State Pointer
├── scripts/
│   ├── __init__.py
│   ├── api/                      # FastAPI Backend Core Rest Layer
│   │   ├── main.py
│   │   └── schemas.py
│   ├── app/                      # Streamlit Frontend Client Rendering Interface
│   │   ├── home/
│   │   │   ├── daily_forecasts.py
│   │   │   └── home.py
│   │   └── server/
│   │       ├── forecasting.py
│   │       ├── latest_model_forecasting.py
│   │       └── server.py
│   ├── architectures/            # Deep Learning Layer Paradigms (LSTM, GRU, Conv1D)
│   │   ├── conv1d.py
│   │   ├── gru.py
│   │   └── lstm.py
│   ├── dl_pipeline.py            # Deep Learning Core Training Loop Execution Script
│   ├── evaluate.py               # Model Metric Evaluation Matrix Block
│   ├── ingestion.py              # Real-time API Asset Pipeline
│   └── preprocessing.py          # Vector Normalization & Sequential Splitting
├── sync_pipeline.bat             # Native Local Windows Task Scheduler Automation Target
└── winner_models.dvc             # DVC Production Model Artifact Tracker

```

---

## 🚀 Application Deployment Guide

### Method 1: Local Native Installation

```bash
# Clone the repository code base
git clone https://github.com/JackTheProgrammer/time-series-forecasting-mlops.git
cd time-series-forecasting-mlops

# Build the base CPU-optimized computing environment and requirements
pip install --no-cache-dir --extra-index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu) torch==2.6.0
pip install --no-cache-dir -r requirements.txt

# Initialize the DVC
# step 1: install DVC via pip
pip install dvc --no-cache-dir
# step 2: run the preprocessing and dl_pipeline scripts to generate the required DVC tracked assets (data, models, winner_models)
python scripts/preprocessing.py
python scripts/dl_pipeline.py
# step 3: Add the generated artifacts to the DVC tracking, no need to craft .dvcignore though
dvc add data
dvc add models
dvc add winner_models

# Launch Backend Engine & Frontend Interface simultaneously so that you can also generate the scaled_transform folder as well, containing the scalers as pickled files
python scripts/api/main.py && streamlit run scripts/app/home/home.py

# adding the scaled_transform folder to DVC artifacts
dvc add scaled_transform
dvc commit -f
dvc remote add -d myremote <LINK_TO_YOUR_REMOTE_STORAGE> # I used google drive as remote storage cause this method was free entirely
# you may need to do the necessary configurations related to the remote storage you are using, for example in case of google storage,
# I followed this tutorial: https://youtu.be/e3GuonR1r-0?si=1f7f4dniQwH-6uCi
dvc push -r myremote
```

### Method 2: Isolated Docker Runtimes

*(Note: Corrected explicit multi-port mappings via decoupled `-p` runtime flags)*

#### Option A: Run Full Ingestion, Training, & Prediction Ecosystem

```bash
docker pull fawadawan143/gold_stock_prediction:latest
docker run -p 8501:8501 -p 5050:5050 fawadawan143/gold_stock_prediction:latest
```

#### Option B: Run Optimized Production Serving UI (Latest Assets Only)

```bash
docker pull fawadawan143/daily-forecasting:latest
docker run -p 8501:8501 -p 5050:5050 fawadawan143/daily-forecasting:latest
```

### Method 3: Multi-Container Orchestration via Docker Compose

```bash
git clone https://github.com/JackTheProgrammer/time-series-forecasting-mlops.git
cd time-series-forecasting-mlops

# Launch system instances in safe daemon modes
docker compose up --build -d

# Spin up isolated processing nodes or production endpoints independently
docker compose up -d ml-pipeline       # Executes full ingestion, retraining, and testing
docker compose up -d forecasting-app    # Serves prediction features using the latest assets

# Spin down the active stack environment safely
docker compose down
```

### Method 4: Production Cluster Deployment (Kubernetes)

```bash
git clone https://github.com/JackTheProgrammer/time-series-forecasting-mlops.git
cd time-series-forecasting-mlops

# Start your localized cluster management context
minikube start --driver=docker
# Apply declarative state manifests into active pods and nodes
kubectl apply -f k8s/
# Inject local image assets directly into Minikube container runtime scopes
minikube image load fawadawan143/daily-forecasting:latest
# Expose your cluster services directly to external clients via LoadBalancer emulation
kubectl expose deployment gold-forecasting-deployment --type=LoadBalancer --port=80 --target-port=80
# In a separate terminal session, initiate the network ingress bridge tunnel
minikube tunnel
```

*Your application endpoint route paths are now accessible at `http://localhost:80/forecast` or `http://localhost:80/latest-forecast` mapping directly into your cluster load-balancer matrix.*

#### Cluster Cleanup Routine

```bash
kubectl delete service gold-forecasting-deployment
kubectl delete deployment gold-forecasting-deployment
minikube stop
```

---

## 🔗 Engineering References & Resources

* **[Core Analytical Blueprint Repository](https://github.com/JackTheProgrammer/Time-Series-Forecasting-and-Analysis)**
* [Docker for Machine Learning | Docker Crash Course | CampusX](https://youtu.be/GToyQTGDOS4?si=Lkb3lNbxzPKwPf2I)
* [Docker Simply Explained with a Machine Learning Project for Beginners](https://youtu.be/-l7YocEQtA0?si=GQ425cnC0SJnHLG9)
* [Data Version Control | DVC | Remote Target Implementations | CodeKamikaze](https://youtu.be/e3GuonR1r-0?si=1f7f4dniQwH-6uCi)
* [Automating Data Pipelines with Python & GitHub Actions Engine](https://youtu.be/wJ794jLP2Tw?si=mXsWGGRQGxmI1rfl)
