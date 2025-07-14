import pickle
import mlflow
from mlflow.tracking import MlflowClient
from flask import Flask, request, jsonify



#------------------------------------------
import os
from dotenv import load_dotenv
import mlflow

# Load environment variables from .env file
load_dotenv()

# Get connection string from environment
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
if not connection_string:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING not found in environment variables")

os.environ["AZURE_STORAGE_CONNECTION_STRING"] = connection_string

# Connect to MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("green-taxi-duration-azure")

print("✅ MLflow with Azure Blob Storage ready!")

CACHE_DIR = "./model_cache"
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5001'
RUN_ID = '988560e0855540acb04adeba7559127c'

model_pipeline = None

#------------------------------------------
def load_model_with_mlflow_cache():
    global model_pipeline

    """Use MLflow's built-in caching by downloading artifacts first"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_model_path = f"{CACHE_DIR}/mlflow_artifacts"

    if os.path.exists(cache_model_path):
        print("✅ Loading model from MLflow cache")
        model_pipeline = mlflow.pyfunc.load_model(cache_model_path)
    else:
        print("📥 Downloading and caching MLflow artifacts...")

        #mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Download artifacts to local directory
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        artifacts_path = client.download_artifacts(
            run_id=RUN_ID,
            path="random_forest_taxi_duration_model",
            dst_path=CACHE_DIR
        )
        print(f"📦 Downloaded to: {artifacts_path}")
        model_pipeline = mlflow.pyfunc.load_model(artifacts_path)

    print(f"✅ Model loaded successfully: {type(model_pipeline)}")
    return model_pipeline


pipeline = load_model_with_mlflow_cache()
print(f"Model type: {type(pipeline)}")

#client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

#pipeline = mlflow.pyfunc.load_model(f"runs:/{RUN_ID}/random_forest_taxi_duration_model")


def prep_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID']) #automatically turns everything into a string
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features):
    preds = pipeline.predict([features])

    #X = dv.transform(features)
    #preds = model.predict (X)
    return preds[0]

app = Flask('taxi_duration_predictor')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prep_features(ride)
    pred = predict(features)

    result = {
        'duration': pred
    }

    return jsonify(result)


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
    #lsof -i :9696
    #kill -9 12345


    #gunicorn --bind=0.0.0.0:9696 predict:app
    #cntl -c
