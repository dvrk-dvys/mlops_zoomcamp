import base64
import json
import os
import pickle

import boto3
import mlflow
from mlflow.tracking import MlflowClient

# Global variables for Lambda container reuse
kinesis_client = boto3.client("kinesis")
s3_client = boto3.client("s3")
model_pipeline = None

S3_BUCKET = "mlflow-models-585315266445"
S3_MODEL_KEY = "models/model.pkl"
PREDICTIONS_STREAM_NAME = "ride_predictions"


TEST_RUN = os.getenv("TEST_RUN", False) == True


def load_model_from_s3():
    global model_pipeline

    if model_pipeline is not None:
        return model_pipeline

    response = s3_client.get_object(Bucket=S3_BUCKET, Key=S3_MODEL_KEY)
    model_data = response["Body"].read()
    model_pipeline = pickle.loads(model_data)
    return model_pipeline


def get_model():
    global model_pipeline
    if model_pipeline is None:
        load_model_from_s3()
    return model_pipeline


def predict(features):
    model_pipeline = get_model()
    preds = model_pipeline.predict(features)
    print(preds[0])
    return preds[0]


def prep_features(ride):
    features = {}
    features["PU_DO"] = "%s_%s" % (
        ride["PULocationID"],
        ride["DOLocationID"],
    )  # automatically turns everything into a string
    features["trip_distance"] = ride["trip_distance"]
    return features


def lambda_handler(event, context):
    # print(json.dumps(event))
    prediction_events = []

    for record in event["Records"]:
        encoded_data = record["kinesis"]["data"]
        decoded_data = base64.b64decode(record["kinesis"]["data"]).decode("utf-8")
        ride_event = json.loads(decoded_data)
        print(ride_event)
        ride = ride_event["ride"]
        ride_id = ride_event["ride_id"]

        features = prep_features(ride)
        prediction = predict(features)
        # ride_id = 123

        prediction_event = {
            "model": "ride_duration_prediction_model",
            "version": "123",
            "prediction": {
                "ride_duration": prediction,
                "ride_id": ride_id,
                # Try to give the consumer as much information as possible because it has no way of knowing what produced the data.
            },
        }

        prediction_events.append(prediction_event)

        # Kinesis_client.put_records  doing this is bulk is alot cheaper but unneccessary right now
        kinesis_client.put_record(
            StreamName=PREDICTIONS_STREAM_NAME,
            Data=json.dumps(prediction_event),
            PartitionKey=str(ride_id),
        )

    return {
        "predictions": prediction_events,
    }


# ------------------------------------


"""
Status: Succeeded
Test Event Name: test_kinesis

Response:
{
  "predictions": [
    {
      "ride_duration": 10,
      "ride_id": 156
    }
  ]
}

Function Logs:
START RequestId: 4fa7284c-4097-49fc-9a7d-e1cf1caaeda6 Version: $LATEST
{'ride': {'PULocationID': 130, 'DOLocationID': 205, 'trip_distance': 3.66}, 'ride_id': 156}
END RequestId: 4fa7284c-4097-49fc-9a7d-e1cf1caaeda6
REPORT RequestId: 4fa7284c-4097-49fc-9a7d-e1cf1caaeda6	Duration: 3.87 ms	Billed Duration: 4 ms	Memory Size: 128 MB	Max Memory Used: 30 MB

Request ID: 4fa7284c-4097-49fc-9a7d-e1cf1caaeda6

"""


"""
event json:

{
    "Records": [
        {
            "kinesis": {
                "kinesisSchemaVersion": "1.0",
                "partitionKey": "1",
                "sequenceNumber": "49665177605138588436641715445750206315132565094062358530",
                "data": "eyJyaWRlIjogeyJQVUxvY2F0aW9uSUQiOiAxMzAsICJET0xvY2F0aW9uSUQiOiAyMDUsICJ0cmlwX2Rpc3RhbmNlIjogMy42Nn0sICJyaWRlX2lkIjogMTU2fQ==",
                "approximateArrivalTimestamp": 1752522563.828
            },
            "eventSource": "aws:kinesis",
            "eventVersion": "1.0",
            "eventID": "shardId-000000000000:49665177605138588436641715445750206315132565094062358530",
            "eventName": "aws:kinesis:record",
            "invokeIdentityArn": "arn:aws:iam::585315266445:role/lambda-kinesis-role",
            "awsRegion": "us-east-1",
            "eventSourceARN": "arn:aws:kinesis:us-east-1:585315266445:stream/ride_events"
        }
    ]
}


"""

"""
output

Status: Succeeded
Test Event Name: test_kinesis

Response:
{
  "predictions": [
    {
      "model": "ride_duration_prediction_model",
      "version": "123",
      "prediction": {
        "ride_duration": 10,
        "ride_id": 156
      }
    }
  ]
}

Function Logs:
START RequestId: 5f8149a4-e2fd-4b6b-af42-bcfc44f0d80f Version: $LATEST
{'ride': {'PULocationID': 130, 'DOLocationID': 205, 'trip_distance': 3.66}, 'ride_id': 156}
END RequestId: 5f8149a4-e2fd-4b6b-af42-bcfc44f0d80f
REPORT RequestId: 5f8149a4-e2fd-4b6b-af42-bcfc44f0d80f	Duration: 288.52 ms	Billed Duration: 289 ms	Memory Size: 128 MB	Max Memory Used: 77 MB	Init Duration: 458.75 ms

Request ID: 5f8149a4-e2fd-4b6b-af42-bcfc44f0d80f
"""
