pipeline {
    agent any

    triggers {
        // Triggers automatically every 4 months on the 1st day at midnight
        cron('0 0 1 */4 *')
    }

    environment {
        APP_NAME = 'daily-forecasts'
        TAG = 'latest'
    }

    stages {
        stage('1. Run Execution Phase') {
            steps {
                echo 'Executing preprocessing and training inside the ml container...'
                bat "docker compose run --rm ml-pipeline python scripts/preprocessing.py"
                bat "docker compose run --rm ml-pipeline python scripts/dl_pipeline.py"
            }
        }

        stage('2. Sync Model Weights') {
            steps {
                echo 'Tracking new data/model hashes with DVC...'
                bat "dvc add winner_models/"
                bat "dvc push"
                
                echo 'Updating git pointer files...'
                bat "git add winner_models.dvc"
                bat "git diff --quiet && git diff --staged --quiet || git commit -m 'automated: 4-month scheduled model retraining update'"
                bat "git push origin main"
            }
        }

        stage('3. Build Deployment Image') {
            steps {
                echo 'Building production docker image...'
                bat "docker build -t %APP_NAME%:%TAG% -f forecasts.Dockerfile ."
            }
        }

        stage('4. Local K8s Orchestration') {
            steps {
                echo 'Stopping any local compose services to clear ports...'
                bat "docker compose down"
                
                echo 'Loading image into Minikube and spinning up services...'
                bat "minikube image load %APP_NAME%:%TAG%"
                bat "kubectl apply -f k8s/deployment.yaml"
                bat "kubectl apply -f k8s/service.yaml"
                bat "kubectl rollout restart deployment gold-forecasting-deployment"
            }
        }
    }

    post {
        success {
            echo '✅ Local retraining and orchestration successful!'
        }
        failure {
            echo '❌ Pipeline failed. Check Jenkins console logs.'
        }
    }
}