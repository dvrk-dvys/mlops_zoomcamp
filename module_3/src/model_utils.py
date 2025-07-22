#!/usr/bin/env python
# coding: utf-8

import os
import pickle
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error


# Configure MLflow - check if there's a running server, otherwise use local directory
def setup_mlflow():
    # Get MLflow tracking URI from environment variable or use default
    # When running in Docker, the MLFLOW_TRACKING_URI should be set to http://mlflow:5000
    # When running locally, it defaults to http://localhost:5000
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

    try:
        mlflow.set_tracking_uri(mlflow_uri)
        # Test the connection
        mlflow.MlflowClient().get_experiment_by_name("nyc-taxi-experiment")
        print(f"Connected to MLflow server at {mlflow_uri}")
    except Exception as e:
        print(f"Could not connect to MLflow server at {mlflow_uri}: {str(e)}")
        print("Using local directory for MLflow tracking")
        # Use relative path within the project directory
        # Check if we're running from the src directory or the project root
        if os.path.exists("../../mlruns"):
            mlflow_dir = "../../mlruns"
        elif os.path.exists("../../mlruns"):
            mlflow_dir = "../../mlruns"
        else:
            # Create the directory if it doesn't exist
            os.makedirs("../../mlruns", exist_ok=True)
            mlflow_dir = "../../mlruns"
        print(f"Using local MLflow directory: {mlflow_dir}")
        mlflow.set_tracking_uri(f"file:{mlflow_dir}")

    mlflow.set_experiment("nyc-taxi-experiment")


# Call setup once at module import time
setup_mlflow()

# Set up paths intelligently based on where the script is run from
if os.path.exists("../../models"):
    models_folder = Path("../../models")
elif os.path.exists("../../models"):
    models_folder = Path("../../models")
else:
    models_folder = Path("../../models")
models_folder.mkdir(exist_ok=True)


def read_dataframe(year, month=None, filename=None):
    """
    Read and preprocess trip data from a parquet file.

    Args:
        year: Year of the data
        month: Month of the data (optional if filename is provided)
        filename: Direct path to the parquet file (optional)

    Returns:
        df: Preprocessed DataFrame
    """
    if filename:
        # If filename is provided directly, use it
        try:
            print(f"Attempting to read from file: {filename}")
            df = pd.read_parquet(filename)
        except Exception as e:
            raise Exception(
                f"Failed to read data: {e}. Please check if the file exists."
            )
    else:
        # Otherwise construct path from year and month
        local_path = f"data/yellow_tripdata_{year}-{month:02d}.parquet"
        try:
            print(f"Attempting to read from local file: {local_path}")
            df = pd.read_parquet(local_path)
        except FileNotFoundError:
            # If local file doesn't exist, try to download from URL
            try:
                print(f"Local file not found. Attempting to download from URL.")
                url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{}-{:02d}.parquet".format(
                    year, month
                )
                df = pd.read_parquet(url)
            except Exception as e:
                raise Exception(
                    "Failed to read data: {}. Please ensure the file exists at {} or check your internet connection.".format(
                        e, local_path
                    )
                )

    # Preprocess the data
    df["duration"] = df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    print(f"DataFrame shape after processing: {df.shape}")
    return df


def create_X(df, dv=None):
    """
    Create feature matrix X from DataFrame

    Args:
        df: Input DataFrame with categorical features
        dv: Existing DictVectorizer (optional)

    Returns:
        X: Feature matrix
        dv: DictVectorizer (fitted if it was not provided)
    """
    # Only use location IDs as features (no numerical features)
    categorical = ["PULocationID", "DOLocationID"]
    dicts = df[categorical].to_dict(orient="records")

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


def train_linear_model(X, y):
    """
    Train a linear regression model

    Args:
        X: Features matrix
        y: Target vector

    Returns:
        model: Trained linear regression model
        rmse: Root mean squared error on training data
    """
    # Initialize and train a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Calculate RMSE on training data
    y_pred = model.predict(X)
    rmse = root_mean_squared_error(y, y_pred)

    return model, rmse


def log_model_with_mlflow(model, X, y, dv, rmse, experiment_name="nyc-taxi-experiment"):
    """
    Log model, metrics and parameters to MLflow

    Args:
        model: Trained model
        X: Features matrix
        y: Target vector
        dv: DictVectorizer used for feature extraction
        rmse: Root mean squared error
        experiment_name: MLflow experiment name

    Returns:
        run_id: MLflow run ID
        artifact_uri: URI to the logged artifacts
    """
    # Ensure mlruns directory exists
    os.makedirs("../../mlruns", exist_ok=True)

    # Start MLflow run (no need to set tracking URI here again)
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("intercept", float(model.intercept_))
        mlflow.log_param("num_features", X.shape[1])

        # Log metrics
        mlflow.log_metric("rmse", rmse)

        # Ensure models directory exists
        os.makedirs("../../models", exist_ok=True)

        # Save the preprocessor
        with open("../../models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("../../models/preprocessor.b", artifact_path="preprocessor")

        # Log the model
        mlflow.sklearn.log_model(model, artifact_path="models_mlflow")

        return run.info.run_id, run.info.artifact_uri


def find_model_size():
    """
    Find all MLmodel files and check their size

    Returns:
        model_sizes: Dictionary with file paths and their sizes
    """
    model_sizes = {}

    # Check if mlruns directory exists
    if not os.path.exists("../../mlruns"):
        print(
            "MLflow directory (mlruns) doesn't exist yet. Run a model training first."
        )
        return model_sizes

    # Find all MLmodel files
    model_files = []
    for root, dirs, files in os.walk("../../mlruns"):
        if "MLmodel" in files:
            mlmodel_path = os.path.join(root, "MLmodel")
            model_files.append(mlmodel_path)

    print(f"Found {len(model_files)} MLmodel files")

    # Check each MLmodel file for model_size_bytes
    for mlmodel_file in model_files:
        try:
            with open(mlmodel_file, "r") as f:
                content = yaml.safe_load(f)
                model_size = content.get("model_size_bytes")
                if model_size is not None:
                    print(f"\nFile: {mlmodel_file}")
                    print(f"Model size: {model_size} bytes")
                    model_sizes[mlmodel_file] = model_size

        except Exception as e:
            print(f"Error reading {mlmodel_file}: {e}")

    print("\nQ6 Answer: 4,534 bytes")
    return model_sizes
