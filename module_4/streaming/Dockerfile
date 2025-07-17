FROM public.ecr.aws/lambda/python:3.10
LABEL authors="jharr"

RUN pip install --upgrade pip

COPY ["requirements.txt", "./"]

RUN pip install --no-cache-dir -r requirements.txt

COPY ["lambda_function.py", "./"]

CMD ["lambda_function.lambda_handler"]


#❯ docker build -t stream-model-duration:v1 .                                                                                             ─╯
# docker run -it --rm -p 9696:9696 ride-duration-prediction-service:v1

#❯ docker run -it --rm \
#    -p 8080:8080 \
#    -e PREDICTIONS_STREAM_NAME="ride_predictions" \
#    -e TEST_RUN="True" \
#    -e AWS_DEFAULT_REGION="us-east-1" \
#    -v ~/.aws:/root/.aws \
#    stream-model-duration:v1

#❯ docker run -it --rm \
#    -p 8080:8080 \
#    -e PREDICTIONS_STREAM_NAME="ride_predictions" \
#    -e TEST_RUN="True" \
#    -e AWS_DEFAULT_REGION="us-east-1" \
#    -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
#    -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
#    stream-model-duration:v1

#
#docker build --platform linux/amd64 -t stream-model-duration:v1 .
#!!!! SINGLE PLATFORM BUILD FOR LAMBDA FUNCTION
#!  docker build --platform linux/amd64 --provenance=false -t stream-model-duration:v1 .

#docker tag stream-model-duration:v1 585315266445.dkr.ecr.us-east-1.amazonaws.com/duration-model:v1
#docker push 585315266445.dkr.ecr.us-east-1.amazonaws.com/duration-model:v1
#The key is making sure:
#
#Docker image is built for linux/amd64
#Lambda architecture is set to x86_64
#
#Try changing the architecture setting in the Lambda UI first before rebuilding.







#https://gallery.ecr.aws/lambda/python
#python for lambda functions to use in docker


#!!!! maybe you need to go in and gie the lamdba fuctions that is using this docker file s3 permissions and maybe more memory