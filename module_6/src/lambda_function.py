import os
import model

#RUN_ID = os.getenv('RUN_ID')
#run_id = "64298b9e7868421cbf6a0505e1211879"


PREDICTIONS_STREAM_NAME = os.getenv('PREDICTIONS_STREAM_NAME', 'ride_predictions')
TEST_RUN = os.getenv('TEST_RUN', 'False') == 'True'
bucket_name = "mlflow-models-585315266445"
s3_key = "models/GradientBoosting_pipeline_model.pkl"

model_service = model.init(
    prediction_stream_name=PREDICTIONS_STREAM_NAME,
    test_run=TEST_RUN,
    bucket_name=bucket_name,
    s3_key=s3_key
)

# actual_features = model_service.prepare_features(ride)

def lambda_handler(event, context):
    return model_service.lambda_handler(event)

