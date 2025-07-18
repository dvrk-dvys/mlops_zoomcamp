import os
import json
import boto3
import base64
import pickle

#import mlflow

#kinesis_client = boto3.client('kinesis')


#PREDICTIONS_STREAM_NAME = os.getenv('PREDICTIONS_STREAM_NAME', 'ride_predictions')
#RUN_ID = os.getenv('RUN_ID')
#TEST_RUN = os.getenv('TEST_RUN', 'False') == 'True'

#run_id = "64298b9e7868421cbf6a0505e1211879"
bucket_name = "mlflow-models-585315266445"
s3_key = "models/GradientBoosting_pipeline_model.pkl"
pipeline = None


def download_pipeline_from_s3(bucket_name, s3_key):
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
    pipeline = pickle.loads(response['Body'].read())
    print(f"✅ Loaded pipeline from S3: {type(pipeline)}")
    return pipeline


def get_pipeline(bucket_name, s3_key):
    global pipeline
    if pipeline is None:
        pipeline = download_pipeline_from_s3(bucket_name, s3_key)
    return pipeline


class ModelService():
    def __init__(self, pipeline, callbacks=None):
        self.pipeline = pipeline
        self.callbacks = callbacks or []


    def prep_features(self, ride):
        features = {}
        # features['PU_DO'] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
        features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
        features['trip_distance'] = ride['trip_distance']
        return features

    def predict(self, model_pipeline, features):
        preds = model_pipeline.predict(features)
        print(preds[0])
        return preds[0]

    def lambda_handler(self, event):
        # print(json.dumps(event))

        prediction_events = []

        for record in event['Records']:
            encoded_data = record['kinesis']['data']
            decoded_data = base64.b64decode(encoded_data).decode('utf-8')
            ride_event = json.loads(decoded_data)
            print(ride_event)

            ride = ride_event['ride']
            ride_id = ride_event['ride_id']

            features = self.prep_features(ride)
            prediction = self.predict(self.pipeline, features)

            prediction_event = {
                'model': 'ride_duration_prediction_model',
                'version': 123,
                'prediction': {
                    'ride_duration': prediction,
                    'ride_id': ride_id
                }
            }

            #if not TEST_RUN:
            #    kinesis_client.put_record(
            #        StreamName=PREDICTIONS_STREAM_NAME,
            #        Data=json.dumps(prediction_event),
            #        PartitionKey=str(ride_id)
            #    )

            for callback in self.callbacks:
                callback(prediction_event)

            prediction_events.append(prediction_event)

        return {
            'predictions': prediction_events,
        }


class KinesisCallback:
    def __init__(self, kinesis_client, prediction_stream_name):
        self.kinesis_client = kinesis_client
        self.prediction_stream_name = prediction_stream_name

    def put_record(self, prediction_event):
        ride_id = prediction_event['prediction']['ride_id']

        self.kinesis_client.put_record(
            StreamName=self.prediction_stream_name,
            Data=json.dumps(prediction_event),
            PartitionKey=str(ride_id),
        )

def create_kinesis_client():
    endpoint_url = os.getenv('KINESIS_ENDPOINT_URL')

    if endpoint_url is None:
        return boto3.client('kinesis')

    return boto3.client('kinesis', endpoint_url=endpoint_url)


def init(prediction_stream_name: str, test_run: bool, bucket_name:str, s3_key: str):
    pipeline = get_pipeline(bucket_name, s3_key)

    callbacks = []

    if not test_run:
        kinesis_client = create_kinesis_client()
        kinesis_callback = KinesisCallback(kinesis_client, prediction_stream_name)
        callbacks.append(kinesis_callback.put_record)

    model_service = ModelService(pipeline=pipeline, callbacks=callbacks)

    return model_service