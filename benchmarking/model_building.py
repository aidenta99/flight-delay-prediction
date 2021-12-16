import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from mlxtend.regressor import StackingCVRegressor

flights_training = pd.read_csv("../input/split/flights_training.csv", index_col="Unnamed: 0")
