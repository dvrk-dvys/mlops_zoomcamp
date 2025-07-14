import predict
import requests


ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 50
    #"fare_amount": "10.0",
    #"extra": "1.0",
    #"mta_tax": "1.0",
    #"tip_amount": "1.0",
    #"tolls_amount": "1.0",
    #"improvement_surcharge": "1.0",
    #"total_amount": "10.0",
}

response = requests.post(url='http://127.0.0.1:9696/predict', json=ride)
print(response.json())


#features = predict.prep_features(ride)
#pred = predict.predict(ride)
#print(pred)

