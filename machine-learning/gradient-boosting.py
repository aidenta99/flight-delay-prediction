import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

# Load the data
df_train_validation = pd.read_csv("final_train_val.csv", low_memory=False)
df_test = pd.read_csv("final_test.csv", low_memory=False)

df_train, df_validation = train_test_split(df_train_validation, test_size=0.20, random_state = 42)
X_train, y_train = df_train.drop("ARRIVAL_DELAY", axis=1), df_train["ARRIVAL_DELAY"]
X_val, y_val = df_validation.drop("ARRIVAL_DELAY", axis=1), df_validation["ARRIVAL_DELAY"]
X_test = df_test

# Hyperparameter tuning
space={
    "n_estimators": 500,
    "learning_rate": hp.loguniform('learning_rate', -0.3, 0.3),
    "max_depth": hp.quniform('max_depth', 5, 12, 1),
    "min_samples_leaf": hp.quniform('min_samples_leaf', 30, 70, 10),
    'seed': 0
    }

def objective(space):
    reg = GradientBoostingRegressor(
                n_estimators =space['n_estimators'],
                learning_rate= space['learning_rate'],
                max_depth= int(space['max_depth']),
                min_samples_leaf= int(space['min_samples_leaf'])
                )

    reg.fit(X_train, y_train)
    val_pred = reg.predict(X_val)
    mse = mean_squared_error(y_val, val_pred)
    return{'loss':mse, 'status': STATUS_OK }

trials = Trials()
best_hyperparams = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials)

print(best_hyperparams)

print("The best hyperparameters are : ","\n")
print(best_hyperparams)