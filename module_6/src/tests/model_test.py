import base64
import json
import os
from pathlib import Path

import model  # noqa: inspection comment #!IGNORE THIS ERROR be sure to mark src as a root


def read_text(file):
    test_directory = Path(__file__).parent
    with open(test_directory / file, "rt", encoding="utf-8") as f_in:
        return f_in.read().strip()


def test_base64_decode():
    """Test base64 decoding functionality"""
    base64_input = read_text("data.b64")

    # Decode the base64 data
    decoded_data = base64.b64decode(base64_input).decode("utf-8")
    actual_result = json.loads(decoded_data)

    expected_result = {
        "ride": {
            "PULocationID": 130,
            "DOLocationID": 205,
            "trip_distance": 3.66,
        },
        "ride_id": 256,
    }

    assert actual_result == expected_result


def test_prepare_features():
    """Test feature preparation with real S3 model"""
    PREDICTIONS_STREAM_NAME = os.getenv("PREDICTIONS_STREAM_NAME", "ride_predictions")
    TEST_RUN = os.getenv("TEST_RUN", "False") == "True"
    bucket_name = "mlflow-models-585315266445"
    s3_key = "models/GradientBoosting_pipeline_model.pkl"

    ride = {
        "PULocationID": 130,
        "DOLocationID": 205,
        "trip_distance": 3.66,
    }

    model_service = model.init(
        prediction_stream_name=PREDICTIONS_STREAM_NAME,
        test_run=TEST_RUN,
        bucket_name=bucket_name,
        s3_key=s3_key,
    )

    actual_features = model_service.prep_features(ride)

    expected_features = {
        "PU_DO": "130_205",
        "trip_distance": 3.66,
    }

    assert actual_features == expected_features


def test_prepare_features_unit():
    """Test feature preparation with mock model (unit test)"""
    model_service = model.ModelService(pipeline=None)

    ride = {
        "PULocationID": 130,
        "DOLocationID": 205,
        "trip_distance": 3.66,
    }

    actual_features = model_service.prep_features(ride)

    expected_features = {
        "PU_DO": "130_205",
        "trip_distance": 3.66,
    }

    assert actual_features == expected_features


class ModelMock:
    """Mock model for testing predictions"""

    def __init__(self, value):
        self.value = value

    def predict(self, X):
        n = len(X) if hasattr(X, "__len__") else 1
        return [self.value] * n


def test_predict():
    """Test prediction with mock model"""
    model_mock = ModelMock(10.0)
    model_service = model.ModelService(pipeline=model_mock)

    features = {
        "PU_DO": "130_205",
        "trip_distance": 3.66,
    }

    # Note: Your predict method expects (model_pipeline, features)
    actual_prediction = model_service.predict(model_mock, [features])
    expected_prediction = 10.0

    assert actual_prediction == expected_prediction


def test_lambda_handler():
    """Test complete lambda handler with mock model"""
    model_mock = ModelMock(10.0)
    model_service = model.ModelService(pipeline=model_mock, callbacks=[])

    base64_input = read_text("data.b64")

    event = {
        "Records": [
            {
                "kinesis": {
                    "data": base64_input,
                },
            }
        ]
    }

    actual_predictions = model_service.lambda_handler(event)
    expected_predictions = {
        "predictions": [
            {
                "model": "ride_duration_prediction_model",
                "version": 123,  # Your current version number
                "prediction": {
                    "ride_duration": 10.0,
                    "ride_id": 256,
                },
            }
        ]
    }

    assert actual_predictions == expected_predictions


def test_lambda_handler_with_real_model():
    """Test lambda handler with real S3 model"""
    PREDICTIONS_STREAM_NAME = os.getenv("PREDICTIONS_STREAM_NAME", "ride_predictions")
    TEST_RUN = True  # Force test mode to avoid Kinesis calls
    bucket_name = "mlflow-models-585315266445"
    s3_key = "models/GradientBoosting_pipeline_model.pkl"

    model_service = model.init(
        prediction_stream_name=PREDICTIONS_STREAM_NAME,
        test_run=TEST_RUN,
        bucket_name=bucket_name,
        s3_key=s3_key,
    )

    base64_input = read_text("data.b64")

    event = {
        "Records": [
            {
                "kinesis": {
                    "data": base64_input,
                },
            }
        ]
    }

    actual_predictions = model_service.lambda_handler(event)

    # Check structure (can't predict exact value with real model)
    assert "predictions" in actual_predictions
    assert len(actual_predictions["predictions"]) == 1
    prediction = actual_predictions["predictions"][0]
    assert prediction["model"] == "ride_duration_prediction_model"
    assert prediction["version"] == 123
    assert "ride_duration" in prediction["prediction"]
    assert prediction["prediction"]["ride_id"] == 256
    assert isinstance(prediction["prediction"]["ride_duration"], float)
