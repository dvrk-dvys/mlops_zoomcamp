from lambda_function import lambda_handler


event = {
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


result = lambda_handler(event, None)
print(result)