FROM python:3.10.18-slim
LABEL authors="jharr"

WORKDIR /app

RUN pip install --upgrade pip

COPY ["requirements.txt", "environment.yml", "./"]

RUN pip install --no-cache-dir -r requirements.txt

COPY ["predict.py", "lin_reg.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]

#docker build -t ride-duration-prediction-service:v1 .
# docker run -it --rm -p 9696:9696 ride-duration-prediction-service:v1