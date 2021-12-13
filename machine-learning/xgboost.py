import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_squared_error

# Load the data
df_train_validation = pd.read_csv("final_train_val.csv", low_memory=False)
df_test = pd.read_csv("final_test.csv", low_memory=False)

df_train, df_validation = train_test_split(df_train_validation, test_size=0.20, random_state = 42)
X_train, y_train = df_train.drop("ARRIVAL_DELAY", axis=1), df_train["ARRIVAL_DELAY"]
X_val, y_val = df_validation.drop("ARRIVAL_DELAY", axis=1), df_validation["ARRIVAL_DELAY"]
X_test = df_test

# Hyperparameter tuning
space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
    }

def objective(space):
    reg = xgb.XGBRegressor(n_estimators =space['n_estimators'], 
                           max_depth = int(space['max_depth']),
                           gamma = space['gamma'], 
                           reg_alpha = int(space['reg_alpha']),
                           min_child_weight=int(space['min_child_weight']),
                           colsample_bytree=int(space['colsample_bytree']))

    eval_set  = [(X_train, y_train), (X_val, y_val)]

    reg.fit(X_train, y_train, eval_set=eval_set, eval_metric = 'rmse',
						early_stopping_rounds=10,verbose=False)
    val_pred = reg.predict(X_val)
    mse = mean_squared_error(y_val, val_pred)
    return{'loss':mse, 'status': STATUS_OK }

trials = Trials()
best_hyperparams = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print(best_hyperparams)

print("The best hyperparameters are : ","\n")
print(best_hyperparams)

# Fit the model
xg_reg = xgb.XGBRegressor(**best_hyperparams)
xg_reg.fit(X_train,y_train)
joblib.dump(xg_reg, "xgreg.pkl")
val_pred = xg_reg.predict(X_val)
print(mean_squared_error(y_val, val_pred))
eval_pred = xg_reg.predict(X_test)
pd.DataFrame(eval_pred, columns=['ARRIVAL_DELAY']).to_csv("flight_result.csv", index_label='id')