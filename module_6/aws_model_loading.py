import boto3
import pickle
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def upload_mlflow_pipeline_to_s3(run_id, bucket_name, s3_key):
    import mlflow.sklearn
    mlflow.set_tracking_uri("http://127.0.0.1:5001")

    model_uri = f"runs:/{run_id}/GradientBoosting_pipeline_model"
    pipeline = mlflow.sklearn.load_model(model_uri)

    s3_client = boto3.client('s3')
    model_bytes = pickle.dumps(pipeline)
    s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=model_bytes)
    print(f"✅ Uploaded pipeline to S3: s3://{bucket_name}/{s3_key}")


def download_pipeline_from_s3(bucket_name, s3_key):
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
    pipeline = pickle.loads(response['Body'].read())
    print(f"✅ Loaded pipeline from S3: {type(pipeline)}")
    return pipeline


def read_dataframe(filename):
    df = pd.read_parquet(filename)
    df['duration'] = pd.to_datetime(df.tpep_dropoff_datetime) - pd.to_datetime(df.tpep_pickup_datetime)
    df.duration = df['duration'].apply(lambda x: x.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    return df


def run_inference_test(pipeline, df_train, df_val):
    print("🔍 Starting inference test...")

    feature_columns = ['PULocationID', 'DOLocationID', 'trip_distance']

    feature_columns = [col for col in feature_columns if col in df_val.columns]

    print(f"📊 Using features: {feature_columns}")

    X_val = df_val[feature_columns]
    y_val = df_val['duration']

    print(f"📈 Validation data shape: {X_val.shape}")
    print(f"🎯 Target statistics - Mean: {y_val.mean():.2f}, Std: {y_val.std():.2f}")

    print("🚀 Making predictions...")
    X_val_dict = X_val.to_dict('records')
    y_pred = pipeline.predict(X_val_dict)

    # Calculate metrics
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    # Print results
    print("\n📊 INFERENCE TEST RESULTS:")
    print("=" * 40)
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE):      {mae:.4f}")
    print(f"R² Score:                       {r2:.4f}")
    print(f"Mean Squared Error (MSE):       {mse:.4f}")

    # Show prediction statistics
    print(f"\n🔍 Prediction Statistics:")
    print(f"Predicted - Mean: {y_pred.mean():.2f}, Std: {y_pred.std():.2f}")
    print(f"Actual    - Mean: {y_val.mean():.2f}, Std: {y_val.std():.2f}")

    # Sample predictions
    print(f"\n📋 Sample Predictions (first 10):")
    print("Actual vs Predicted:")
    for i in range(min(10, len(y_val))):
        print(f"  {y_val.iloc[i]:6.2f} vs {y_pred[i]:6.2f} (diff: {abs(y_val.iloc[i] - y_pred[i]):5.2f})")

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mse': mse,
        'predictions': y_pred,
        'actual': y_val.values
    }


if __name__ == "__main__":
    run_id = "64298b9e7868421cbf6a0505e1211879"
    bucket_name = "mlflow-models-585315266445"
    s3_key = "models/GradientBoosting_pipeline_model.pkl"

    #upload_mlflow_pipeline_to_s3(run_id, bucket_name, s3_key)


    gb_pipeline = download_pipeline_from_s3(bucket_name, s3_key)

    train_path ='/Users/jordanharris/Code/mlops_zoomcamp/data/yellow_tripdata_2023-01.parquet'
    val_path ='/Users/jordanharris/Code/mlops_zoomcamp/data/yellow_tripdata_2023-02.parquet'


    df_train = read_dataframe(train_path)
    df_val = read_dataframe(val_path)

    print(f"Training data shape: {df_train.shape}")
    print(f"Validation data shape: {df_val.shape}")

    # Run inference test
    results = run_inference_test(gb_pipeline, df_train, df_val)

    # Additional analysis
    print(f"\n✅ Inference test completed successfully!")
    print(f"Model performance: {'Good' if results['r2'] > 0.5 else 'Needs improvement'} (R² = {results['r2']:.3f})")