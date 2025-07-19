## Code snippets

### Building and running Docker images

```
# docker build -t lambda-func-duration:v1 .                                                                                             ─╯
```

```
docker run -it --rm \
    -p 8080:8080 \
    -e PREDICTIONS_STREAM_NAME="ride_predictions" \
    -e TEST_RUN="True" \
    -e AWS_DEFAULT_REGION="us-east-1" \
    -v ~/.aws:/root/.aws \
    lambda-func-duration:v1
```

```
# First time: Build and start
docker-compose up --build

# Or separately:
docker-compose build
docker-compose up -d

# Subsequent times: Just start (if no code changes)
docker-compose up -d

docker-compose down

```

```
hooks

pip install pre-commit black isort pylint pytest
pre-commit install
```
