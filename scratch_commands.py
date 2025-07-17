"""
if starting fresh remove the previous database:
> rm mlflow.db

conda activate mlops_zoomcamp read only!
mlflow ui --backend-store-uri sqlite:///mlflow.db

for adding experiments:
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5001

to run in background
nohup mlflow ui --backend-store-uri sqlite:///mlflow.db > mlflow.log 2>&1 &


the url
http://127.0.0.1:5000


if it wont end
❯ lsof -tiTCP:5000 -sTCP:LISTEN
❯ kill $(lsof -tiTCP:5000 -sTCP:LISTEN)



local host with blob connections

#mlflow server \
#  --backend-store-uri=sqlite:///mlflow.db \
#  --default-artifact-root=wasbs://mlflow-artifacts@lifebonderblobml.blob.core.windows.net/ \
#  --host 0.0.0.0 \
#  --port 5001

http://127.0.0.1:5001

homeworks:
https://github.com/RuiFSP/mlops-zoomcamp-2025/blob/master/02-experiment-tracking/homework02.ipynb
"""
