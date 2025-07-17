#!/usr/bin/env python
# coding: utf-8

import os
import time
import argparse
import logging
import requests
import subprocess
import pandas as pd
import mlflow
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def wait_for_service(url, service_name, max_retries=10, retry_interval=10):
    """Wait for a service to become available by polling its URL."""
    for i in range(max_retries):
        try:
            logger.info(
                f"Attempt {i + 1}/{max_retries}: Checking if {service_name} is available at {url}..."
            )
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"{service_name} is available!")
                return True
            else:
                logger.warning(
                    f"{service_name} returned status code {response.status_code}"
                )
        except requests.RequestException as e:
            logger.warning(f"Failed to connect to {service_name}: {e}")

        if i < max_retries - 1:
            logger.info(f"Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)

    logger.error(f"Could not connect to {service_name} after {max_retries} attempts")
    return False


def ensure_data_files_exist():
    """Ensure the required taxi data files exist."""
    data_dir = Path("data")
    required_files = [
        "yellow_tripdata_2023-01.parquet",
        "yellow_tripdata_2023-02.parquet",
    ]

    missing_files = []
    for file in required_files:
        if not (data_dir / file).exists():
            missing_files.append(file)

    if missing_files:
        logger.error(f"Missing data files: {', '.join(missing_files)}")
        logger.error("Please ensure these files are available in the data directory")
        return False

    logger.info("All required data files found!")
    return True


def setup_docker_environment():
    """Start the Docker services for testing."""
    logger.info("Setting up Docker environment...")

    # Check if Docker is running
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        logger.error("Docker is not running. Please start Docker and try again.")
        return False

    # Start Docker services using our helper script
    try:
        logger.info("Starting Docker services...")
        subprocess.run(["./docker-services.sh", "start"], check=True)
        logger.info("Docker services started successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Docker services: {e}")
        return False


def check_mlflow_server():
    """Check if MLflow server is running and accessible."""
    mlflow_uri = "http://localhost:5000"
    return wait_for_service(mlflow_uri, "MLflow server")


def list_mlflow_experiments():
    """List all experiments in MLflow."""
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()

        logger.info("MLflow Experiments:")
        for exp in experiments:
            logger.info(f"  - {exp.name} (ID: {exp.experiment_id})")

        # Check if our experiment exists
        nyc_exp = client.get_experiment_by_name("nyc-taxi-experiment")
        if nyc_exp:
            logger.info(f"Found NYC Taxi experiment (ID: {nyc_exp.experiment_id})")
            return True, nyc_exp.experiment_id
        else:
            logger.warning("NYC Taxi experiment not found")
            return False, None
    except Exception as e:
        logger.error(f"Error accessing MLflow: {e}")
        return False, None


def examine_mlflow_runs(experiment_id):
    """List runs for the NYC Taxi experiment and examine metrics."""
    if not experiment_id:
        logger.error("No experiment ID provided")
        return False

    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        client = mlflow.tracking.MlflowClient()

        # Get all runs for the experiment
        runs = client.search_runs(experiment_ids=[experiment_id])

        if not runs:
            logger.warning("No runs found for the experiment")
            return False

        logger.info(f"Found {len(runs)} runs for the experiment")

        # Examine the latest run
        latest_run = runs[0]

        # Display metrics
        logger.info("Latest run metrics:")
        for key, value in latest_run.data.metrics.items():
            logger.info(f"  - {key}: {value}")

        # Check for key metrics
        if "rmse" in latest_run.data.metrics and "val_rmse" in latest_run.data.metrics:
            logger.info("Found both training and validation RMSE metrics ✓")
            return True
        else:
            logger.warning("Missing key metrics in the latest run")
            return False
    except Exception as e:
        logger.error(f"Error examining MLflow runs: {e}")
        return False


def execute_workflow_in_prefect_container():
    """Execute the workflow in the Prefect container."""
    try:
        logger.info("Executing workflow in Prefect container...")

        # Execute the command
        cmd = [
            "docker",
            "exec",
            "prefect-workflow",
            "python",
            "/app/taxi_prediction_flow.py",
        ]
        process = subprocess.run(cmd, capture_output=True, text=True, check=False)

        # Check for success
        if process.returncode == 0:
            logger.info("Workflow execution completed successfully")
            logger.info("----- Output -----")
            for line in process.stdout.splitlines():
                if "RMSE" in line or "shape" in line or "Successfully" in line:
                    logger.info(line)
            logger.info("-----------------")
            return True
        else:
            logger.error(
                f"Workflow execution failed with return code {process.returncode}"
            )
            logger.error("----- Error Output -----")
            logger.error(process.stderr)
            logger.error("-----------------------")
            return False
    except Exception as e:
        logger.error(f"Error executing workflow: {e}")
        return False


def run_end_to_end_test():
    """Run full end-to-end test of the workflow."""
    success = True

    # Step 1: Check for required data files
    if not ensure_data_files_exist():
        return False

    # Step 2: Set up Docker environment
    if not setup_docker_environment():
        return False

    # Step 3: Wait for MLflow to start
    if not check_mlflow_server():
        return False

    # Step 4: Check for existing experiments
    has_experiment, experiment_id = list_mlflow_experiments()

    # Step 5: Execute the workflow
    if not execute_workflow_in_prefect_container():
        return False

    # Give MLflow a moment to register the run
    time.sleep(5)

    # Step 6: Check for experiments again if none were found before
    if not has_experiment:
        has_experiment, experiment_id = list_mlflow_experiments()

    # Step 7: Examine the runs
    if has_experiment:
        success = examine_mlflow_runs(experiment_id) and success

    if success:
        logger.info("✨ End-to-end test completed successfully! ✨")
        logger.info("MLflow is logging experiments correctly from the Prefect workflow")
        logger.info("Visit http://localhost:5000 to view the MLflow UI")
    else:
        logger.error("❌ End-to-end test failed")

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the MLflow and Prefect workflow")
    parser.add_argument(
        "--stop-after", action="store_true", help="Stop Docker services after testing"
    )
    args = parser.parse_args()

    try:
        success = run_end_to_end_test()

        if args.stop_after:
            logger.info("Stopping Docker services...")
            subprocess.run(["./docker-services.sh", "stop"], check=False)

        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)
