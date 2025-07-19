#!/usr/bin/env bash

if [[ -z "${GITHUB_ACTIONS}" ]]; then
  cd "$(dirname "$0")"
fi

# Load environment variables from .env file
if [ -f "../src/.env" ]; then
    echo "Loading environment variables from ../src/.env"
    export $(grep -v '^#' ../src/.env | xargs)
elif [ -f ".env" ]; then
    echo "Loading environment variables from .env"
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: No .env file found"
fi

if [ "${LOCAL_IMAGE_NAME}" == "" ]; then
    LOCAL_TAG=`date +"%Y-%m-%d-%H-%M"`
    export LOCAL_IMAGE_NAME="stream-model-duration:${LOCAL_TAG}"
    echo "LOCAL_IMAGE_NAME is not set, building a new image with tag ${LOCAL_IMAGE_NAME}"
    # Build from the src directory where the Dockerfile is located
    docker build -t ${LOCAL_IMAGE_NAME} ../src
else
    echo "no need to build image ${LOCAL_IMAGE_NAME}"
fi

export PREDICTIONS_STREAM_NAME="ride_predictions"

# Clean up any existing containers
docker-compose down --remove-orphans

# Start services
echo "Starting services..."
docker-compose up -d

# Wait for LocalStack to be ready
echo "Waiting for LocalStack to be ready..."
MAX_ATTEMPTS=30
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:4566/_localstack/health >/dev/null 2>&1; then
        HEALTH_STATUS=$(curl -s http://localhost:4566/_localstack/health | grep -o '"kinesis": "[^"]*"' | cut -d'"' -f4)
        if [ "$HEALTH_STATUS" = "available" ]; then
            echo "✅ LocalStack Kinesis service is ready"
            break
        fi
    fi
    echo "Waiting for Kinesis service... (attempt $((ATTEMPT + 1))/$MAX_ATTEMPTS)"
    sleep 2
    ATTEMPT=$((ATTEMPT + 1))
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo "❌ Failed to start LocalStack after $MAX_ATTEMPTS attempts"
    docker-compose logs
    docker-compose down
    exit 1
fi

# Wait for backend service to be ready
echo "Waiting for backend service to be ready..."
MAX_ATTEMPTS=60
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    # First check if the port is open
    if curl -s --connect-timeout 5 http://localhost:8180 >/dev/null 2>&1; then
        echo "Port 8180 is accessible, checking Lambda function..."
        # Give it a few more seconds for Lambda to fully initialize
        sleep 5
        
        # Now test with actual payload
        if curl -s --max-time 10 http://localhost:8180/2015-03-31/functions/function/invocations \
           -H "Content-Type: application/json" \
           -d '{"Records":[]}' >/dev/null 2>&1; then
            echo "✅ Backend service is ready"
            break
        else
            echo "Lambda function not responding yet..."
        fi
    fi
    echo "Waiting for backend service... (attempt $((ATTEMPT + 1))/$MAX_ATTEMPTS)"
    sleep 3
    ATTEMPT=$((ATTEMPT + 1))
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo "❌ Backend service failed to start after $MAX_ATTEMPTS attempts"
    echo "Checking container status..."
    docker-compose ps
    echo "Backend container logs:"
    docker-compose logs backend
    docker-compose down
    exit 1
fi

# Create Kinesis stream
echo "Creating Kinesis stream..."
aws --endpoint-url=http://localhost:4566 \
    kinesis create-stream \
    --stream-name ${PREDICTIONS_STREAM_NAME} \
    --shard-count 1

# Verify stream was created
echo "Verifying Kinesis stream..."
STREAM_STATUS=""
MAX_ATTEMPTS=10
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    STREAM_STATUS=$(aws --endpoint-url=http://localhost:4566 \
                   kinesis describe-stream \
                   --stream-name ${PREDICTIONS_STREAM_NAME} \
                   --query 'StreamDescription.StreamStatus' \
                   --output text 2>/dev/null)

    if [ "$STREAM_STATUS" = "ACTIVE" ]; then
        echo "✅ Kinesis stream is active"
        break
    fi
    echo "Waiting for stream to be active... (status: $STREAM_STATUS, attempt $((ATTEMPT + 1))/$MAX_ATTEMPTS)"
    sleep 1
    ATTEMPT=$((ATTEMPT + 1))
done

if [ "$STREAM_STATUS" != "ACTIVE" ]; then
    echo "❌ Failed to create active Kinesis stream"
    docker-compose logs
    docker-compose down
    exit 1
fi

# Install dependencies if not present
if ! pipenv run python -c "import requests" 2>/dev/null; then
    echo "Installing missing dependencies..."
    pipenv install requests deepdiff
fi

# Run the tests
echo "🧪 Running Docker integration test..."
pipenv run python test_docker.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    echo "❌ Docker test failed"
    docker-compose logs
    docker-compose down
    exit ${ERROR_CODE}
fi

# Simulate sending a Kinesis event to trigger the Lambda to publish to Kinesis
echo "📤 Sending Kinesis event to Lambda function to trigger publishing..."
pipenv run python -c "
import requests
import json

with open('event.json', 'r') as f:
    event = json.load(f)

url = 'http://localhost:8180/2015-03-31/functions/function/invocations'
response = requests.post(url, json=event)
print(f'Lambda response: {response.status_code}')
print(f'Lambda output: {response.json()}')
"

# Give some time for the prediction to be published to Kinesis
echo "⏳ Waiting for Kinesis record to be published..."
sleep 3

echo "🧪 Running Kinesis integration test..."
pipenv run python test_kinesis.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    echo "❌ Kinesis test failed"
    docker-compose logs
    docker-compose down
    exit ${ERROR_CODE}
fi

echo "✅ All integration tests passed!"
docker-compose down