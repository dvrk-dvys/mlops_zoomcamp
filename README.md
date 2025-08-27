# MLOps Zoomcamp

This repository contains my coursework and projects from the [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp), covering machine learning operations best practices, tools, and workflows.

## 🚀 Quick Start

To set up the project:

```bash
make setup
```

## 🔧 Development Setup

Install development dependencies:
```bash
pip install pre-commit black isort pylint pytest
pre-commit install
```

## 📊 MLflow Tracking

Start the MLflow UI to view experiment tracking:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Access the UI at: http://localhost:5001

## 📚 Course Overview

Following the comprehensive [DataTalksClub MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) curriculum, this repository covers:

- **Module 1**: Introduction to MLOps
- **Module 2**: Experiment tracking and model registry with MLflow
- **Module 3**: Orchestration and ML Pipelines
- **Module 4**: Model Deployment
- **Module 5**: Model Monitoring
- **Module 6**: Best Practices and MLOps Maturity

## 🛠 Tools & Technologies

- **MLflow** - Experiment tracking and model registry
- **Docker** - Containerization
- **GitHub Actions** - CI/CD pipelines
- **AWS** - Cloud infrastructure
- **Terraform** - Infrastructure as code
- **Monitoring** - Model and data drift detection
