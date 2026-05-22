pipeline {
    agent any

    triggers {
        // Triggers automatically every 4 months on the 1st day at midnight
        cron('0 0 1 */4 *')
    }

    environment {
        // REPLACE 'your_dockerhub_username' with your actual Docker Hub account name
        DOCKER_HUB_USER = 'your_dockerhub_username'
        APP_NAME        = 'daily-forecasts'
        TAG             = 'latest'
        COMMIT_DATE     = ''
    }

    stages {
        stage('1. Continuous Integration: Run ML Pipeline') {
            steps {
                echo 'Executing preprocessing and deep learning training layers...'
                bat "docker compose run --rm ml-pipeline python scripts/preprocessing.py"
                bat "docker compose run --rm ml-pipeline python scripts/dl_pipeline.py"
            }
        }

        stage('2. Continuous Delivery: Sync Model Weights & Code') {
            steps {
                script {
                    env.COMMIT_DATE = new Date().format("yyyy-MM-dd HH:mm:ss")
                }

                echo 'Tracking new data/model hashes with DVC...'
                bat "dvc add data/"
                bat "dvc add winner_models/"
                bat "dvc add scaled_transform/"
                
                echo 'Pushing model weights to Google Drive Remote...'
                bat "dvc push -r gdrive_remote"
                
                echo 'Committing structural pointer files to Git...'
                bat "git add ."
                bat "git commit -m 'automated: 4-month scheduled model retraining update at %COMMIT_DATE%'"
                bat "git push -u origin main"
            }
        }

        stage('3. Continuous Delivery: Publish to Docker Hub') {
            steps {
                echo 'Building production-ready Docker image...'
                bat "docker build -t %DOCKER_HUB_USER%/%APP_NAME%:%TAG% -f forecasts.Dockerfile ."
                
                echo 'Logging into Docker Hub and pushing image...'
                // 'docker-hub-credentials' must match the ID of the username/password credential stored in Jenkins
                withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'USER', passwordVariable: 'PASS')]) {
                    bat "docker login -u %USER% -p %PASS%"
                    bat "docker push %DOCKER_HUB_USER%/%APP_NAME%:%TAG%"
                }
            }
        }

        stage('4. Continuous Deployment: Local K8s Orchestration') {
            steps {
                echo 'Tearing down old compose configurations to free system ports...'
                bat "docker compose down"
                
                echo 'Sideloading the newly verified image directly into Minikube...'
                bat "minikube image load %DOCKER_HUB_USER%/%APP_NAME%:%TAG%"
                
                echo 'Applying Kubernetes manifests and forcing rolling restart...'
                bat "kubectl apply -f k8s/deployment.yaml"
                bat "kubectl apply -f k8s/service.yaml"
                bat "kubectl rollout restart deployment gold-forecasting-deployment"
            }
        }
    }

    post {
        success {
            echo '✅ End-to-End CI/CD Pipeline executed successfully. Artifacts shipped!'
        }
        failure {
            echo '❌ Pipeline failed. Verification or delivery phase broken.'
        }
    }
}