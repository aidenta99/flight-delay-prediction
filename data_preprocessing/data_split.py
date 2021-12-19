import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

def split_data():
    
    #CHANGE INPUT LOCATION
    airports = pd.read_csv('input/airports.csv')
    flights_train = pd.read_csv('input/flights_train.csv')
    
    airports_origin = airports[['IATA_CODE','LATITUDE','LONGITUDE']].rename(columns = {'IATA_CODE' : 'ORIGIN_AIRPORT'})
    airports_arrive = airports[['IATA_CODE','LATITUDE','LONGITUDE']].rename(columns = {'IATA_CODE' : 'DESTINATION_AIRPORT'})
    flights_train = flights_train.merge(airports_origin, on = 'ORIGIN_AIRPORT').rename(columns = {'LATITUDE' : 'LATITUDE_origin', 'LONGITUDE' : 'LONGITUDE_origin'})
    flights_train = flights_train.merge(airports_arrive, on = 'DESTINATION_AIRPORT').rename(columns = {'LATITUDE' : 'LATITUDE_arrival', 'LONGITUDE' : 'LONGITUDE_arrival'})
    
    train, test_val = train_test_split(flights_train, test_size=0.20, random_state = 42)
    test, val = train_test_split(test_val, test_size=0.6, random_state = 42)
    print('splitted data into parts with sizes: {} train, {} test and {} validation'.format(len(train), len(test), len(val)))
    
    #CHANGE OUTPUT NAMES
    train.to_csv('flights_training.csv')
    test.to_csv('flights_testing.csv')
    val.to_csv('flights_validation.csv')

split_data()