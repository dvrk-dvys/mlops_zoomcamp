#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

# cloud check
import platform

print("Running on:", platform.uname())
print("User:", os.getlogin())
print("Python path:", os.__file__)


# In[2]:


get_ipython().system("python3 -V")


# In[3]:


import pickle

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from mlflow import pyfunc
from mlflow.models.signature import infer_signature
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

# In[4]:


# In[5]:


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc_taxi_trip_duration_experiment")


# In[6]:


import os

print(os.getcwd())

# df = pd.read_parquet('/home/azureuser/j_harr/data/green_tripdata_2021-01.parquet')
df = pd.read_parquet("./data/green_tripdata_2021-01.parquet")


# In[7]:


df["duration"] = pd.to_datetime(df.lpep_dropoff_datetime) - pd.to_datetime(
    df.lpep_pickup_datetime
)
df.duration = df["duration"].apply(lambda x: x.total_seconds() / 60)


# In[8]:


td = df.duration.iloc[0]
td


# In[9]:


# df[df.trip_type == 2]


# In[10]:


sns.distplot(df.duration)


# In[11]:


df.duration.describe(percentiles=[0.95, 0.98, 0.99])


# In[12]:


# small filtering
df = df[(df.duration >= 1) & (df.duration <= 60)]


# In[13]:


categorical = ["PULocationID", "DOLocationID"]
numerical = ["trip_distance"]


# In[14]:


# df[categorical].astype(str).dtypes
df[categorical] = df[categorical].astype(str)
# df.dtypes
df[categorical + numerical].iloc[:10].to_dict(orient="records")


# In[15]:


train_dicts = df[categorical + numerical].to_dict(orient="records")
dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)


# In[16]:


# print(dv.feature_names_)
print(X_train.shape)
target = "duration"
y_train = df[target].values
print(y_train.shape)


# In[17]:


lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_train)
sns.distplot(y_pred, label="prediction")
sns.distplot(y_train, label="actual")
plt.legend()

# ------------------------------------------------------

mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")


# In[18]:


def read_dataframe(filename):
    df = pd.read_parquet(filename)
    df["duration"] = pd.to_datetime(df.lpep_dropoff_datetime) - pd.to_datetime(
        df.lpep_pickup_datetime
    )
    df.duration = df["duration"].apply(lambda x: x.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ["PULocationID", "DOLocationID"]
    # numerical = ['trip_distance']
    df[categorical] = df[categorical].astype(str)

    return df


# In[19]:


df_train = read_dataframe("./data/green_tripdata_2021-01.parquet")
df_val = read_dataframe("./data/green_tripdata_2021-02.parquet")


# In[20]:


len(df_train), len(df_val)


# In[21]:


df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]


# In[22]:


categorical = ["PU_DO"]  #'PULocationID', 'DOLocationID']
numerical = ["trip_distance"]

dv = DictVectorizer()

train_dicts = df_train[categorical + numerical].to_dict(orient="records")
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient="records")
X_val = dv.transform(val_dicts)


# In[23]:


target = "duration"
y_train = df_train[target].values
y_val = df_val[target].values


# In[24]:


with mlflow.start_run(run_name="linear_regression"):

    mlflow.set_tag("model_type", "linear_regression")
    mlflow.set_tag("developer", "ja_harr")

    mlflow.log_param("train-data-path", "./data/green_tripdata_2021-01.parquet")
    mlflow.log_param("valid-data-path", "./data/green_tripdata_2021-02.parquet")

    ln = LinearRegression()
    ln.fit(X_train, y_train)

    y_pred = ln.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"MSE: {mse:.3f}")
    print(f"Root MSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R²: {r2:.3f}")
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("R²", r2)

    ln_pipeline = make_pipeline(dv, ln)
    signature = infer_signature(X_train, y_train)
    # input_example = train_dicts[:5]

    mlflow.sklearn.log_model(
        sk_model=ln_pipeline,
        name="linear_model",
        signature=signature,
        # input_example=input_example,
        registered_model_name="linear_regression",  # optional, auto-register in the MLflow Model Registry
    )


# In[25]:


# with open('models/lin_reg.bin', 'wb') as f_out:
#    pickle.dump((dv, lr), f_out)


# In[26]:


with mlflow.start_run(run_name="lasso_regression"):
    mlflow.set_tag("model_type", "lasso_regression")
    mlflow.set_tag("developer", "ja_harr")

    mlflow.log_param("train-data-path", "./data/green_tripdata_2021-01.parquet")
    mlflow.log_param("valid-data-path", "./data/green_tripdata_2021-02.parquet")

    alpha = 0.01
    mlflow.log_param("alpha", alpha)

    l = Lasso(alpha=alpha)
    l.fit(X_train, y_train)

    y_pred = l.predict(X_val)

    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"MSE: {mse:.3f}")
    print(f"Root MSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R²: {r2:.3f}")
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("R²", r2)

    lasso_pipeline = make_pipeline(dv, l)
    signature = infer_signature(X_train, y_train)
    # input_example = train_dicts[:5]

    mlflow.sklearn.log_model(
        sk_model=lasso_pipeline,
        name="lasso_model",
        signature=signature,
        # input_example=input_example,
        registered_model_name="lasso_regression",  # optional, auto-register in the MLflow Model Registry
    )


# In[27]:


# with open('models/lasso.bin', 'wb') as f_out:
#    pickle.dump((dv, lr), f_out)


# In[28]:


with mlflow.start_run(run_name="ridge_regression"):
    mlflow.set_tag("model_type", "ridge_regression")
    mlflow.set_tag("developer", "ja_harr")

    mlflow.log_param("train-data-path", "./data/green_tripdata_2021-01.parquet")
    mlflow.log_param("valid-data-path", "./data/green_tripdata_2021-02.parquet")

    alpha = 0.001
    mlflow.log_param("alpha", alpha)

    r = Ridge(alpha=alpha)
    r.fit(X_train, y_train)

    y_pred = r.predict(X_val)

    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"MSE: {mse:.3f}")
    print(f"Root MSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R²: {r2:.3f}")
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("R²", r2)

    ridge_pipeline = make_pipeline(dv, r)
    signature = infer_signature(X_train, y_train)
    # input_example = train_dicts[:5]

    # example_df = df_val[categorical + numerical].iloc[:5]
    # preds = pipeline.predict(example_df)
    # signature = infer_signature(example_df, preds)

    mlflow.sklearn.log_model(
        sk_model=ridge_pipeline,
        name="ridge_model",
        signature=signature,
        # input_example=input_example,
        registered_model_name="ridge_regression",  # optional, auto-register in the MLflow Model Registry
    )

    # mlflow.log_artifact(local_path="models/ridge.bin", #artifact_path="models_pickle")


# In[29]:


# with open('models/ridge.bin', 'wb') as f_out:
#    pickle.dump((dv, lr), f_out)
# mlflow.log_artifact(local_path="models/ridge.bin", #artifact_path="models_pickle")
# Purpose

# A low‐level API for uploading any file you like (logs, plots, custom binary blobs, pickles, etc.) into the run’s artifact store.

# It does not wrap it in a standardized “MLmodel” directory or register it—it just copies the file.


# In[30]:


import sys

import sklearn
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope

# In[31]:


print("sklearn version:", sklearn.__version__)
print("python executable:", sys.executable)


# In[32]:


train = xgb.DMatrix(X_train, label=y_train)
valid = xgb.DMatrix(X_val, label=y_val)


# In[33]:


def objective(params):
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, "validation")],
            early_stopping_rounds=50,
        )
        y_pred = booster.predict(valid)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("R²", r2)

    return {"loss": rmse, "status": STATUS_OK}


# In[34]:


search_space = {
    "max_depth": scope.int(hp.quniform("max_depth", 4, 100, 1)),
    "learning_rate": hp.loguniform(
        "learning_rate", -3, 0
    ),  # exp(-3), exp(0) - [0.05, 1]
    "reg_alpha": hp.loguniform("reg_alpha", -5, -1),
    "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
    "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
    "objective": "reg:linear",
    "seed": 42,
}
# The ranges in which we want hyperopt to explore the hyperparameters
# https://hyperopt.github.io/hyperopt/getting-started/search_spaces/

# best_result = fmin(
#    fn=objective, #fmin will try to optimize the given objective by minimizing the output
#    space=search_space,
#    algo=tpe.suggest, #algorithm to run the optimization
#    max_evals=50,
#    trials=Trials()# informatiuon for each run stored in tirals
# )


# In[35]:


mlflow.xgboost.autolog(disable=True)


# In[38]:


with mlflow.start_run():

    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    best_params = {
        "learning_rate": 0.09585355369315604,
        "max_depth": 30,
        "min_child_weight": 1.060597050922164,
        "objective": "reg:linear",
        "reg_alpha": 0.018060244040060163,
        "reg_lambda": 0.011658731377413597,
        "seed": 42,
    }

    mlflow.log_params(best_params)

    booster = xgb.train(
        params=best_params,
        dtrain=train,
        num_boost_round=1000,
        evals=[(valid, "validation")],
        early_stopping_rounds=50,
    )

    y_pred = booster.predict(valid)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("R²", r2)

    with open("models/preprocessor.b", "wb") as f_out:
        pickle.dump(dv, f_out)
mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

mlflow.xgboost.log_model(booster, artifact_path="xgboost_model")


# In[41]:


from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.svm import LinearSVR

if mlflow.active_run():
    mlflow.end_run()

mlflow.sklearn.autolog()

for model_class in (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    LinearSVR,
):

    with mlflow.start_run():

        mlflow.log_param("train-data-path", "./data/green_tripdata_2021-01.csv")
        mlflow.log_param("valid-data-path", "./data/green_tripdata_2021-02.csv")
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlmodel = model_class()
        mlmodel.fit(X_train, y_train)

        y_pred = mlmodel.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("R²", r2)


# In[108]:


run_id = "c340236dba7c4903bdb8995e2bbebf9b"
artifact = "models_mlflow"  # <— the folder you just confirmed

model_uri = f"runs:/{run_id}/{artifact}"
pyfunc_model = mlflow.pyfunc.load_model(model_uri)


# In[ ]:


loaded_model


# In[107]:


run_id = "9ae4346c55764400b2b5ee7a6cac56d2"
model_uri = f"runs:/{run_id}/models_mlflow"
booster = mlflow.xgboost.load_model(model_uri)

# now you can call booster.predict on a DMatrix
import xgboost as xgb

dm = xgb.DMatrix(X_val)  # or new data


# In[109]:


y_pred = booster.predict(dm)
y_pred[:10]
