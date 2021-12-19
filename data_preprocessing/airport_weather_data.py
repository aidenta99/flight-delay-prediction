#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests as r
import geopandas as gpd
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from shapely.geometry import Point


# In[2]:


airports = pd.read_csv("./airports.csv")


# In[3]:


airports[airports.LATITUDE.isna()]


# In[4]:


# Fix ECP
airports.loc[96,["LATITUDE", "LONGITUDE"]] = (30.3548543,-85.8017021)
# Fix PBG
airports.loc[234,["LATITUDE", "LONGITUDE"]] = (44.6520597,-73.470109)
# Fix UST
airports.loc[313,["LATITUDE", "LONGITUDE"]] = (29.95439,-81.3450803)


# In[5]:


try:
    weather = pd.read_csv("./weather_data.csv")  # pd.DataFrame()
except:
    df = pd.DataFrame(columns=[
        "station",
        "valid",
        "tmpc",
        "sknt",
        "p01m",
        "vsby",
        "gust",
        "skyc1",
        "skyc2",
        "skyc3",
        "wxcodes",
        "ice_accretion_6hr",
        "snowdepth"
    ])

    for code in airports.IATA_CODE:
        url = f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={code}&data=tmpc&data=sknt&data=p01m&data=vsby&data=gust&data=skyc1&data=skyc2&data=skyc3&data=wxcodes&data=ice_accretion_6hr&data=snowdepth&year1=2015&month1=1&day1=1&year2=2015&month2=8&day2=1&tz=Etc%2FUTC&format=onlycomma&latlon=no&elev=no&missing=empty&trace=T&direct=no&report_type=1&report_type=2"
        df = pd.read_csv(url)
        weather = weather.append(df)
    weather.to_csv("./weather_data.csv")


# In[6]:


weather


# In[7]:


weather["valid"] = pd.to_datetime(weather["valid"])
weather


# In[8]:


weather.gust = weather.gust.fillna(0)
weather.ice_accretion_6hr = weather.ice_accretion_6hr.fillna(0)
weather.loc[weather.ice_accretion_6hr=='T',"ice_accretion_6hr"].ice_accretion_6hr = 0
weather = weather.join(weather.wxcodes.str.split(' ', expand=True))
weather = weather.drop(columns="snowdepth")


# In[9]:


# Creating a geometry column for all airports
geometry = [Point(xy) for xy in zip(airports['LONGITUDE'], airports['LATITUDE'])]
# Creating a Geographic data frame for all airports
gpd1 = gpd.GeoDataFrame(airports, geometry=geometry).reset_index(drop=True)

# Get all the airports for which we couldn't retrieve any data
missing_airports = set(airports.IATA_CODE.unique()) - set(weather.station.unique())
# Create geo data frame for the missing airports
gpd2 = gpd1[gpd1["IATA_CODE"].isin(missing_airports)].reset_index(drop=True)
# Change the column name to be identifiable later
gpd2["MISSING_IATA_CODE"] = gpd2["IATA_CODE"]
gpd2 = gpd2.drop(columns="IATA_CODE")


# In[10]:


def ckdnearest(gdA, gdB):
    """
    Function to compute pairwise distances between all points in gdA and gdB
    Found in: https://gis.stackexchange.com/a/301935
    """
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ], 
        axis=1)

    return gdf


# In[11]:


# Find the closest airport to the missing ones
distance_matrix = ckdnearest(gpd2, gpd1[~gpd1.IATA_CODE.isin(missing_airports)]).sort_values('dist')
airport_mapping = distance_matrix[["MISSING_IATA_CODE", "IATA_CODE"]]
airport_mapping


# In[12]:


df_missing_airports = pd.DataFrame()
for airports in airport_mapping.itertuples():
    missing_airport = airports[1]
    closest_airport = airports[2]
    closest_airport = weather[weather.station==closest_airport]
    closest_airport.station = missing_airport
    df_missing_airports = df_missing_airports.append(closest_airport)


# In[13]:


weather = weather.append(df_missing_airports)


# In[14]:


weather.to_csv("weather_for_all_airports.csv")


# In[15]:


weather.valid = weather.valid.dt.round('1h')


# # Collocate with Flights

# In[16]:


flights_train = pd.read_csv("./train_emiel_v5.csv")
flights_test = pd.read_csv("./test_emiel_v5.csv")


# In[17]:


flights_train.SCHEDULED_DEPARTURE_DATETIME = pd.to_datetime(flights_train.SCHEDULED_DEPARTURE_DATETIME)
flights_test.SCHEDULED_DEPARTURE_DATETIME = pd.to_datetime(flights_test.SCHEDULED_DEPARTURE_DATETIME)


# In[18]:


weather


# In[19]:


flights_train.SCHEDULED_DEPARTURE_DATETIME = flights_train.SCHEDULED_DEPARTURE_DATETIME.dt.round('1h') 
flights_test.SCHEDULED_DEPARTURE_DATETIME = flights_test.SCHEDULED_DEPARTURE_DATETIME.dt.round('1h') 


# In[ ]:


flights_train.merge(weather, how="left", left_on="SCHEDULED_DEPARTURE_DATETIME", right_on="valid", suffixes=("_departure","_departure"))


# In[ ]:




