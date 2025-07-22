#!/usr/bin/env python
# coding: utf-8

import logging
import sys

import mlflow
import numpy as np
from prefect import flow, task

# Import the utility functions we already have
from module_3.src.model_utils import (
    create_X,
    log_model_with_mlflow,
    read_dataframe,
    train_linear_model,
)


# Add more verbose error handling
def handle_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@task(
    name="Get Data Periods",
    description="Get the appropriate training and validation periods",
    retries=2,
)
def get_data_periods():
    """
    Calculate the appropriate periods for training and validation data:
    - Training: 2 months ago
    - Validation: 1 month ago

    Note: For demonstration purposes, we're using fixed 2023 data
    that is available in the dataset.
    """
    logger.info("Determining data periods for training and validation")

    # Using fixed data that we know exists in our repository
    # Training period: January 2023
    train_year, train_month = 2023, 1

    # Validation period: February 2023
    val_year, val_month = 2023, 2

    logger.info(f"Training period: {train_year}-{train_month:02d}")
    logger.info(f"Validation period: {val_year}-{val_month:02d}")

    return (train_year, train_month), (val_year, val_month)


@task(
    name="Prepare Training Data",
    description="Load and preprocess training data",
    retries=3,
)
def prepare_train_data(year, month):
    """Load and prepare training data"""
    logger.info(f"Preparing training data for {year}-{month:02d}")
    try:
        df = read_dataframe(year, month)
        logger.info(f"Successfully read training data, shape: {df.shape}")

        # Create target variable
        y_train = df.duration.values

        # Create feature matrix and dict vectorizer
        X_train, dv = create_X(df)
        logger.info(f"Features created, X_train shape: {X_train.shape}")

        return X_train, y_train, dv
    except Exception as e:
        logger.error(f"Error preparing training data: {e}")
        raise


@task(
    name="Prepare Validation Data",
    description="Load and preprocess validation data",
    retries=3,
)
def prepare_val_data(year, month, dv):
    """Load and prepare validation data using the training DictVectorizer"""
    logger.info(f"Preparing validation data for {year}-{month:02d}")
    try:
        df = read_dataframe(year, month)
        logger.info(f"Successfully read validation data, shape: {df.shape}")

        # Create target variable
        y_val = df.duration.values

        # Create feature matrix using the training DictVectorizer
        X_val, _ = create_X(df, dv)
        logger.info(f"Features created, X_val shape: {X_val.shape}")

        return X_val, y_val
    except Exception as e:
        logger.error(f"Error preparing validation data: {e}")
        raise


@task(name="Train Model", description="Train linear regression model")
def train_model(X_train, y_train):
    """Train the linear regression model"""
    logger.info("Training linear regression model")
    try:
        model, rmse = train_linear_model(X_train, y_train)
        logger.info(f"Training RMSE: {rmse:.4f}")
        return model, rmse
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise


@task(name="Evaluate Model", description="Evaluate model on validation data")
def evaluate_model(model, X_val, y_val):
    """Evaluate the model on validation data"""
    from sklearn.metrics import mean_squared_error

    logger.info("Evaluating model on validation data")
    try:
        y_pred = model.predict(X_val)
        val_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        logger.info(f"Validation RMSE: {val_rmse:.4f}")

        return val_rmse
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise


@task(name="Log Model", description="Log model to MLflow")
def log_model(model, X_train, y_train, dv, train_rmse, val_rmse):
    """Log the model, metrics and parameters to MLflow"""
    logger.info("Logging model to MLflow")

    try:
        # Start MLflow run and log everything
        run_id, artifact_uri = log_model_with_mlflow(
            model=model, X=X_train, y=y_train, dv=dv, rmse=train_rmse
        )

        # Log validation RMSE as well (separate call since it's not in the original function)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("val_rmse", val_rmse)

        logger.info(f"Model logged with run_id: {run_id}")
        logger.info(f"Model artifacts: {artifact_uri}")

        return run_id
    except Exception as e:
        logger.error(f"Error logging model to MLflow: {e}")
        raise


@flow(
    name="NYC Taxi Duration Prediction",
    description="End-to-end ML pipeline for NYC taxi duration prediction",
)
def taxi_duration_prediction_pipeline():
    """Main flow that orchestrates the entire ML pipeline"""
    try:
        logger.info("Starting NYC Taxi Duration Prediction pipeline")

        # Get the appropriate time periods for training and validation
        (train_year, train_month), (val_year, val_month) = get_data_periods()

        # Prepare training data
        X_train, y_train, dv = prepare_train_data(train_year, train_month)

        # Prepare validation data
        X_val, y_val = prepare_val_data(val_year, val_month, dv)

        # Train the model
        model, train_rmse = train_model(X_train, y_train)

        # Evaluate on validation data
        val_rmse = evaluate_model(model, X_val, y_val)

        # Log the model to MLflow
        run_id = log_model(model, X_train, y_train, dv, train_rmse, val_rmse)

        logger.info("Pipeline completed successfully!")
        return run_id
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


def setup_monthly_schedule():
    """
    Create a deployment using serve() for Prefect 3.x
    """
    logger.info("Setting up monthly scheduled deployment")

    try:
        print("\nCreating deployment...")

        # Use the serve() method which is recommended in Prefect 3.x
        deployment_id = taxi_duration_prediction_pipeline.serve(
            name="monthly-taxi-duration",
            cron="0 0 1 * *",  # At midnight on the 1st day of each month
            tags=["monthly", "production"],
            description="Monthly training of NYC taxi duration prediction model",
        )

        print("\nDeployment created successfully!")
        print("\nIMPORTANT: To use a work pool for better scheduling, please run:")
        print(
            '1. Create a process work pool: prefect work-pool create --type process "taxi-prediction-pool"'
        )
        print(
            "2. Start a worker for this pool: PREFECT_API_URL=http://127.0.0.1:4200/api prefect worker start --pool taxi-prediction-pool"
        )

        # Run a terminal command to ensure the schedule works
        import subprocess

        try:
            subprocess.run(
                [
                    "prefect",
                    "deployment",
                    "run",
                    "NYC Taxi Duration Prediction/monthly-taxi-duration",
                ],
                check=True,
            )
            logger.info("Triggered an initial deployment run")
        except Exception as schedule_error:
            logger.warning(f"Could not trigger initial run: {schedule_error}")

        logger.info("Monthly scheduled deployment created and serving!")
        print("\n✅ Deployment 'monthly-taxi-duration' created and serving!")
        print(
            "\nYour flow will now run automatically at midnight on the 1st of every month."
        )
        print("You can view the deployment in the Prefect UI at http://localhost:4200")
        print("\nNote: For scheduled runs to work correctly, make sure to:")
        print("   1. Keep the Prefect server running")
        print("   2. Have at least one agent running to process the scheduled runs")

        return deployment_id
    except Exception as e:
        logger.error(f"Error creating monthly schedule: {e}")
        print(f"\n❌ Error creating deployment: {e}")
        print("\nPlease check that your Prefect server is running with:")
        print("   prefect server start")
        raise


if __name__ == "__main__":
    # Run the flow directly or show deployment instructions
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--deploy":
            # Create and set up a scheduled deployment
            print("Creating monthly scheduled deployment...")
            setup_monthly_schedule()
        else:
            # Run the flow once immediately
            print("Running the flow once...")
            taxi_duration_prediction_pipeline()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)
