// Jenkinsfile for Jenkins CI/CD Pipeline
// Requires Jenkins with Kubernetes plugin and Docker pipeline plugin

pipeline {
    agent {
        kubernetes {
            yaml """
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: docker
    image: docker:latest
    command:
    - sleep
    args:
    - 99d
    volumeMounts:
    - name: docker-sock
      mountPath: /var/run/docker.sock
  - name: kubectl
    image: bitnami/kubectl:latest
    command:
    - sleep
    args:
    - 99d
  volumes:
  - name: docker-sock
    hostPath:
      path: /var/run/docker.sock
"""
        }
    }

    environment {
        DOCKER_IMAGE = "virality-engine"
        DOCKER_REGISTRY = "${env.DOCKER_REGISTRY}"
        KUBERNETES_NAMESPACE = "${env.KUBERNETES_NAMESPACE ?: 'virality-staging'}"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build') {
            steps {
                container('docker') {
                    script {
                        sh """
                            pip install --upgrade pip
                            pip install -r requirements.txt
                            pip install pytest pytest-cov pylint black isort
                        """
                    }
                }
            }
        }

        stage('Test') {
            steps {
                container('docker') {
                    script {
                        sh """
                            pylint src/ --exit-zero || true
                            black --check src/ tests/ || true
                            isort --check-only src/ tests/ || true
                            pytest tests/ -v --cov=src --cov-report=xml --cov-report=html
                        """
                    }
                }
            }
            post {
                always {
                    publishHTML([
                        reportDir: 'htmlcov',
                        reportFiles: 'index.html',
                        reportName: 'Coverage Report'
                    ])
                    publishTestResults testResultsPattern: 'test-results.xml'
                }
            }
        }

        stage('Security Scan') {
            steps {
                container('docker') {
                    script {
                        sh """
                            pip install safety bandit
                            safety check --file requirements.txt || true
                            bandit -r src/ -f json -o bandit-report.json || true
                        """
                    }
                }
            }
            post {
                always {
                    archiveArtifacts artifacts: 'bandit-report.json', allowEmptyArchive: true
                }
            }
        }

        stage('Build Docker Image') {
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                }
            }
            steps {
                container('docker') {
                    script {
                        def imageTag = "${DOCKER_REGISTRY}/${DOCKER_IMAGE}:${env.BUILD_NUMBER}"
                        def imageLatest = "${DOCKER_REGISTRY}/${DOCKER_IMAGE}:latest"
                        
                        sh """
                            docker build -t ${imageTag} -t ${imageLatest} .
                            docker push ${imageTag}
                            docker push ${imageLatest}
                        """
                        
                        env.DOCKER_IMAGE_TAG = imageTag
                    }
                }
            }
        }

        stage('Deploy to Staging') {
            when {
                branch 'develop'
            }
            steps {
                container('kubectl') {
                    script {
                        sh """
                            kubectl config use-context ${env.KUBECTL_CONTEXT ?: 'default'}
                            kubectl get namespace ${KUBERNETES_NAMESPACE} || kubectl create namespace ${KUBERNETES_NAMESPACE}
                            sed -i 's|image: .*|image: ${DOCKER_IMAGE_TAG}|g' k8s/deployment.yaml
                            kubectl apply -f k8s/configmap.yaml -n ${KUBERNETES_NAMESPACE}
                            kubectl apply -f k8s/service.yaml -n ${KUBERNETES_NAMESPACE}
                            kubectl apply -f k8s/deployment.yaml -n ${KUBERNETES_NAMESPACE}
                            kubectl rollout status deployment/virality-engine-api -n ${KUBERNETES_NAMESPACE} --timeout=300s
                        """
                    }
                }
            }
        }

        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                input message: 'Deploy to Production?', ok: 'Deploy'
                container('kubectl') {
                    script {
                        env.KUBERNETES_NAMESPACE = 'virality-production'
                        sh """
                            kubectl config use-context ${env.KUBECTL_CONTEXT_PRODUCTION ?: 'production'}
                            kubectl get namespace ${KUBERNETES_NAMESPACE} || kubectl create namespace ${KUBERNETES_NAMESPACE}
                            sed -i 's|image: .*|image: ${DOCKER_IMAGE_TAG}|g' k8s/rollout.yaml
                            kubectl apply -f k8s/configmap.yaml -n ${KUBERNETES_NAMESPACE}
                            kubectl apply -f k8s/service.yaml -n ${KUBERNETES_NAMESPACE}
                            kubectl apply -f k8s/rollout.yaml -n ${KUBERNETES_NAMESPACE}
                            kubectl apply -f k8s/hpa.yaml -n ${KUBERNETES_NAMESPACE}
                            kubectl apply -f k8s/pdb.yaml -n ${KUBERNETES_NAMESPACE}
                            kubectl apply -f k8s/ingress.yaml -n ${KUBERNETES_NAMESPACE}
                        """
                    }
                }
            }
        }
    }

    post {
        success {
            echo 'Pipeline succeeded!'
        }
        failure {
            echo 'Pipeline failed!'
        }
        always {
            cleanWs()
        }
    }
}

