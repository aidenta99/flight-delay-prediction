import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler

airports = pd.read_csv('airports.csv')
flights_train_validation = pd.read_csv('flights_train.csv')
flights_test = pd.read_csv('flights_test.csv')

def preprocessing(df):
    airports_origin = airports[['IATA_CODE','LATITUDE','LONGITUDE']].rename(columns = {'IATA_CODE' : 'ORIGIN_AIRPORT'})
    airports_arrive = airports[['IATA_CODE','LATITUDE','LONGITUDE']].rename(columns = {'IATA_CODE' : 'DESTINATION_AIRPORT'})

    df = df.merge(airports_origin, on = 'ORIGIN_AIRPORT').rename(columns = {'LATITUDE' : 'LATITUDE_origin', 'LONGITUDE' : 'LONGITUDE_origin'})
    df = df.merge(airports_arrive, on = 'DESTINATION_AIRPORT').rename(columns = {'LATITUDE' : 'LATITUDE_arrival', 'LONGITUDE' : 'LONGITUDE_arrival'})

    # Drop unuseful cols: "YEAR" (year = 2015 for every row), "id", "TAIL_NUMBER" (high cardinality),
    # "ORIGIN_AIRPORT", "DESTINATION_AIRPORT" (high cardinality, already have longitude and latitude)
    df.drop(["YEAR", "TAIL_NUMBER", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"], axis=1, inplace=True)
    df = df.sort_values(by="id").set_index("id")
    return df

flights_train_validation = preprocessing(flights_train_validation)
flights_test = preprocessing(flights_test)

# Feature engineering
# 1. One-hot encoding
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train_val = pd.DataFrame(OH_encoder.fit_transform(flights_train_validation[["AIRLINE"]]))
OH_cols_train_val.columns=OH_encoder.get_feature_names(["AIRLINE"])

OH_cols_test = pd.DataFrame(OH_encoder.transform(flights_test[["AIRLINE"]]))
OH_cols_test.columns=OH_encoder.get_feature_names(["AIRLINE"])

# One-hot encoding removed index; put it back
OH_cols_train_val.index = flights_train_validation.index
OH_cols_test.index = flights_test.index

# Add one-hot encoded columns to numerical features
flights_train_validation = pd.concat([flights_train_validation, OH_cols_train_val], axis=1)
flights_test = pd.concat([flights_test, OH_cols_test], axis=1)

# Drop AIRLINE col
flights_train_validation.drop("AIRLINE", axis=1, inplace=True)
flights_test.drop("AIRLINE", axis=1, inplace=True)

# 2. Robust scaler
scaler = RobustScaler()
transformed_cols = ['DEPARTURE_TIME', 'WHEELS_OFF', 'SCHEDULED_TIME', 'DISTANCE']
transformed_cols_train_val = pd.DataFrame(scaler.fit_transform(flights_train_validation[transformed_cols]), columns=["scaled_"+col for col in transformed_cols])
transformed_cols_test = pd.DataFrame(scaler.transform(flights_test[transformed_cols]), columns=["scaled_"+col for col in transformed_cols])

transformed_cols_train_val.index = flights_train_validation.index
transformed_cols_test.index = flights_test.index

flights_train_validation = pd.concat([flights_train_validation, transformed_cols_train_val], axis=1)
flights_test = pd.concat([flights_test, transformed_cols_test], axis=1)

flights_train_validation.drop(transformed_cols, axis=1, inplace=True)
flights_test.drop(transformed_cols, axis=1, inplace=True)

flights_train_validation.to_csv("final_train_val.csv")
flights_test.to_csv("final_test.csv")