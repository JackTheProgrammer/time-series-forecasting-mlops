pipeline {
    // Runs on any available Jenkins agent
    agent any

    environment {
        APP_NAME = 'daily-forecasting'
        TAG = 'latest'
        K8S_DIR = 'k8s'
    }

    stages {
        stage('Checkout Code') {
            steps {
                // Automatically pulls the branch that triggered the build
                checkout scm
            }
        }

        stage('DVC Model Synchronization') {
            steps {
                echo 'Pulling latest gold forecasting models and data via DVC...'
                // Fetches the .pt and .pkl files into the workspace
                sh 'dvc pull'
            }
        }

        stage('Build Serving Image') {
            steps {
                echo 'Building the FastAPI + Streamlit deployment image...'
                sh "docker build -t ${APP_NAME}:${TAG} -f forecasts.Dockerfile ."
            }
        }

        stage('Load Image to Minikube') {
            steps {
                echo 'Injecting the new image directly into the Minikube cluster...'
                // Bypasses the need for Docker Hub by loading directly to your local cluster
                sh "minikube image load ${APP_NAME}:${TAG}"
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                echo 'Applying Kubernetes manifests...'
                sh "kubectl apply -f ${K8S_DIR}/deployment.yaml"
                sh "kubectl apply -f ${K8S_DIR}/service.yaml"
                
                // Forces Kubernetes to spin up new pods with the freshly loaded image
                sh "kubectl rollout restart deployment gold-forecasting-deployment"
            }
        }
    }

    post {
        success {
            echo '✅ Pipeline executed successfully! The updated gold forecasting dashboard is live.'
        }
        failure {
            echo '❌ Pipeline failed. Please review the stage logs to identify the bottleneck.'
        }
    }
}