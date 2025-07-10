"""
if starting fresh remove the previous database:
> rm mlflow.db

conda activate mlops_zoomcamp
mlflow ui --backend-store-uri sqlite:///mlflow.db

to run in background
nohup mlflow ui --backend-store-uri sqlite:///mlflow.db > mlflow.log 2>&1 &


the url
http://127.0.0.1:5000


if it wont end
❯ lsof -tiTCP:5000 -sTCP:LISTEN
❯ kill $(lsof -tiTCP:5000 -sTCP:LISTEN)




homeworks:
https://github.com/RuiFSP/mlops-zoomcamp-2025/blob/master/02-experiment-tracking/homework02.ipynb
"""
