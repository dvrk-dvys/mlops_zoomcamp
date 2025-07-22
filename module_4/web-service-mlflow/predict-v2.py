import os

import mlflow
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from mlflow.tracking import MlflowClient

# Configuration
MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
RUN_ID = "988560e0855540acb04adeba7559127c"


def setup_azure_mlflow():
    """Setup Azure MLflow connection"""
    load_dotenv()

    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        print("⚠️  AZURE_STORAGE_CONNECTION_STRING not found, using local storage")
        return False

    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = connection_string
    print("✅ MLflow with Azure Blob Storage ready!")
    return True


def initialize_mlflow():
    """Initialize MLflow tracking URI"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"✅ MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")


def load_model():
    """Load the MLflow model with error handling"""
    try:
        print("📥 Loading model from MLflow...")
        pipeline = mlflow.pyfunc.load_model(
            f"runs:/{RUN_ID}/random_forest_taxi_duration_model"
        )
        print("✅ Model loaded successfully!")
        return pipeline
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


def prep_features(ride):
    """Prepare features from ride data"""
    features = {}
    features["PU_DO"] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    features["trip_distance"] = ride["trip_distance"]
    return features


def make_prediction(pipeline, features):
    """Make prediction using the pipeline"""
    try:
        preds = pipeline.predict([features])
        return float(preds[0])  # Ensure JSON serializable
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise


def validate_input(ride):
    """Validate required fields in ride data"""
    required_fields = ["PULocationID", "DOLocationID", "trip_distance"]
    missing_fields = [field for field in required_fields if field not in ride]

    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"

    return True, None


def create_app():
    """Create and configure Flask application"""
    app = Flask("taxi_duration_predictor")

    @app.route("/predict", methods=["POST"])
    def predict_endpoint():
        try:
            ride = request.get_json()

            if not ride:
                return jsonify({"error": "No JSON data provided"}), 400

            # Validate input
            is_valid, error_msg = validate_input(ride)
            if not is_valid:
                return jsonify({"error": error_msg}), 400

            # Process prediction
            features = prep_features(ride)
            pred = make_prediction(pipeline, features)

            result = {
                "duration": pred,
                "duration_minutes": round(pred, 2),
                "input_features": features,
            }

            return jsonify(result)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/health", methods=["GET"])
    def health_check():
        return jsonify(
            {
                "status": "healthy",
                "model_loaded": pipeline is not None,
                "run_id": RUN_ID,
            }
        )

    @app.route("/", methods=["GET"])
    def home():
        return jsonify(
            {
                "service": "Taxi Duration Predictor",
                "version": "1.0",
                "endpoints": {"predict": "/predict (POST)", "health": "/health (GET)"},
                "example_request": {
                    "PULocationID": 161,
                    "DOLocationID": 237,
                    "trip_distance": 5.2,
                },
            }
        )

    return app


def main():
    """Main application entry point"""
    print("🚀 Starting Taxi Duration Predictor...")

    # Setup Azure connection
    setup_azure_mlflow()

    # Initialize MLflow
    initialize_mlflow()

    # Load model globally
    # global pipeline
    # pipeline = load_model()

    # Create Flask app
    app = create_app()

    # Run the application
    print("🌐 Starting Flask server...")
    app.run(debug=True, host="0.0.0.0", port=9696)


if __name__ == "__main__":
    main()

# Production deployment:
# gunicorn --bind=0.0.0.0:9696 predict:app
#
# Test commands:
# curl http://localhost:9696/
# curl http://localhost:9696/health
# curl -X POST http://localhost:9696/predict \
#   -H "Content-Type: application/json" \
#   -d '{"PULocationID": 161, "DOLocationID": 237, "trip_distance": 5.2}'
