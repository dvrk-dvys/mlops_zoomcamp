FROM public.ecr.aws/lambda/python:3.10
LABEL authors="jharr"

RUN pip install --upgrade pip

COPY ["requirements.txt", "./"]

# Add awslambdaric for local Lambda testing
RUN pip install --no-cache-dir -r requirements.txt awslambdaric

COPY ["lambda_function.py", "model.py", "./"]

CMD ["lambda_function.lambda_handler"]

# Build
#docker build -t lambda-func-duration:v1 .

# Run (choose one method for AWS creds)
#docker run -it --rm \
#    -p 8080:8080 \
#    -e PREDICTIONS_STREAM_NAME="ride_predictions" \
#    -e TEST_RUN="True" \
#    -e AWS_DEFAULT_REGION="us-east-1" \
#    -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
#    -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
#    -v ~/.aws:/root/.aws \
#    lambda-func-duration:v1


#https://gallery.ecr.aws/lambda/python
#python for lambda functions to use in docker


#!!!! maybe you need to go in and gie the lamdba fuctions that is using this docker file s3 permissions and maybe more memory
