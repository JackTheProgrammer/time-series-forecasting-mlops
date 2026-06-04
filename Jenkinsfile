pipeline{
    agent any
    triggers{
        // This cron expression schedules the job to run at 17:00 (5:00 PM) on the first 
        // day of every 4th month (January, May, September)
        cron('0 17 1 */4 *')
    }
    stages{
        stage('Synching to local machine the GHA performed workflow'){
            steps{
                bat 'git pull --rebase origin main'
                bat 'dvc pull'
            }
        }
    }
    post{
        always{
            echo 'Stage execution completed. Please check the logs for details.'
        }
        success{
            echo 'Workflow based synching successful.'
        }
        failure{
            echo 'Workflow based synching failed. Please investigate the issue.'
        }
    }
}